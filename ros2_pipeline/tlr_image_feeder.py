#!/usr/bin/env python3
"""Feed a directory of images into the live TLR recognition graph, one frame at a
time, so the production detector+classifier nodes autolabel them.

The frame index is encoded in the ROS timestamp (stamp.sec = index+1, nanosec=0).
Every downstream node preserves the image header stamp, so the collector can map
each output back to its source frame by stamp alone. A frame_map.json (sec ->
image path) is also written for filename recovery.

Publish rate is deliberately modest: the classifier ExactTime-syncs image+rois
(queue 10), so a frame must still be buffered when its ROIs come back around the
detector->adapter->classifier chain. Slower = safer against drops.

Usage (after sourcing the workspace):
  python3 tlr_image_feeder.py --image-dir <dir> --topic /tlr_autolabel/image \
      --rate 3 --frame-id camera6 --map-out /tmp/frame_map.json
"""
import argparse
import glob
import json
import os
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def list_images(d):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    return sorted(sum([glob.glob(os.path.join(d, e)) for e in exts], []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--topic", default="/tlr_autolabel/image")
    ap.add_argument("--rate", type=float, default=3.0, help="publish rate (Hz)")
    ap.add_argument("--frame-id", default="camera")
    ap.add_argument("--map-out", default="/tmp/tlr_frame_map.json")
    ap.add_argument("--start-index", type=int, default=1, help="first stamp.sec (avoid 0)")
    ap.add_argument("--wait-subs", type=int, default=3,
                    help="min subscriber count on the image topic before feeding "
                         "(detector + car classifier + ped classifier = 3)")
    ap.add_argument("--grace", type=float, default=5.0,
                    help="seconds to keep spinning after the last frame")
    ap.add_argument("--limit", type=int, default=0, help="only feed first N images (0=all)")
    args = ap.parse_args()

    rclpy.init()
    node = Node("tlr_image_feeder")
    pub = node.create_publisher(Image, args.topic, QoSProfile(depth=10))
    bridge = CvBridge()

    paths = list_images(args.image_dir)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        node.get_logger().error(f"no images under {args.image_dir}")
        return
    # write the frame map UP FRONT (all filenames are known) so the collector,
    # which starts before we finish feeding, can name outputs by source image.
    frame_map = {str(args.start_index + i): os.path.realpath(p) for i, p in enumerate(paths)}
    with open(args.map_out, "w") as f:
        json.dump(frame_map, f)

    node.get_logger().info(f"{len(paths)} images; waiting for >= {args.wait_subs} subscriber(s) on {args.topic}")

    # wait for the detector/classifier to subscribe (they may still be building engines)
    t0 = time.time()
    while pub.get_subscription_count() < args.wait_subs:
        rclpy.spin_once(node, timeout_sec=0.2)
        if time.time() - t0 > 600:
            node.get_logger().error("timed out waiting for subscribers")
            return
    node.get_logger().info("subscriber(s) connected; start feeding")

    period = 1.0 / args.rate if args.rate > 0 else 0.0
    for i, p in enumerate(paths):
        sec = args.start_index + i
        img = cv2.imread(p)
        if img is None:
            node.get_logger().warn(f"skip unreadable {p}")
            continue
        msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = args.frame_id
        pub.publish(msg)
        frame_map[str(sec)] = os.path.realpath(p)
        if (i + 1) % 50 == 0 or i + 1 == len(paths):
            node.get_logger().info(f"fed {i + 1}/{len(paths)}")
        end = time.time() + period
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=max(0.0, end - time.time()))

    with open(args.map_out, "w") as f:
        json.dump(frame_map, f)
    node.get_logger().info(f"done feeding; frame_map -> {args.map_out}; grace {args.grace}s")
    t_end = time.time() + args.grace
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
