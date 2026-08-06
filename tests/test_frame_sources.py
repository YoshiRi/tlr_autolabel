"""Frame source contract.

The identity a source assigns (`frame_id`, `frame_index`, `channel`, `rel_path`)
is what ends up in Tier A and what the comparison layer joins on, so it is
pinned here — in particular that `ImageDirSource` still produces exactly the
identity the L1 CLI produced when it owned the glob.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from tlr_autolabel.frames import build_frame_source
from tlr_autolabel.frames.cache import materialize
from tlr_autolabel.frames.images import ImageDirSource, MaterializedSource
from tlr_autolabel.frames.rosbag import channel_of_topic, decode_image_msg
from tlr_autolabel.frames.t4 import T4DatasetSource


def write_image(path: Path, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((8, 12, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return img


class ImageDirSourceTest(unittest.TestCase):
    def test_frame_identity_matches_the_historical_l1_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "CAM_FRONT" / "00000.png")
            write_image(root / "CAM_FRONT" / "00007.jpg")
            write_image(root / "CAM_FRONT" / "named.png")

            src = ImageDirSource(path=str(root / "CAM_FRONT"), image_root=str(root))
            frames = list(src)

            self.assertEqual([f.frame_id for f in frames],
                             ["00000", "00007", "named"],
                             "one sorted() over the concatenated per-extension globs, "
                             "so order is by path, not by extension")
            by_id = {f.frame_id: f for f in frames}
            self.assertEqual(by_id["00000"].frame_index, 0)
            self.assertEqual(by_id["00007"].frame_index, 7,
                             "numeric stem wins over sequence position")
            self.assertEqual(by_id["named"].frame_index, 2,
                             "non-numeric stem falls back to its position in the list")
            self.assertEqual(by_id["00000"].channel, "CAM_FRONT")
            self.assertEqual(by_id["00000"].rel_path, "CAM_FRONT/00000.png")
            self.assertTrue(by_id["00000"].realpath.endswith("CAM_FRONT/00000.png"))
            self.assertIsNone(by_id["00000"].source, "file sources add no source block")
            self.assertIsNone(by_id["00000"].timestamp_us)

    def test_single_file_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            write_image(path)
            frames = list(ImageDirSource(path=str(path)))
            self.assertEqual([f.frame_id for f in frames], ["frame"])
            self.assertEqual(frames[0].rel_path, "frame.png")

    def test_skip_callback_runs_before_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "00000.png")
            write_image(root / "00001.png")
            asked = []

            def skip(frame_id):
                asked.append(frame_id)
                return frame_id == "00000"

            frames = list(ImageDirSource(path=str(root)).iter_frames(skip=skip))
            self.assertEqual([f.frame_id for f in frames], ["00001"])
            self.assertEqual(asked, ["00000", "00001"])

    def test_unreadable_file_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "00000.png")
            (root / "00001.png").write_text("not an image")
            frames = list(ImageDirSource(path=str(root)))
            self.assertEqual([f.frame_id for f in frames], ["00000"])

    def test_t4_dataset_fills_sample_data_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "data" / "CAM_FRONT" / "00000.png")
            (root / "annotation").mkdir(parents=True)
            (root / "annotation" / "sample_data.json").write_text(json.dumps([
                {"token": "sd-token-0", "filename": "data/CAM_FRONT/00000.png"}]))
            frames = list(ImageDirSource(path=str(root / "data" / "CAM_FRONT"),
                                        t4_dataset=str(root)))
            self.assertEqual(frames[0].sample_data_token, "sd-token-0")
            self.assertEqual(frames[0].rel_path, "data/CAM_FRONT/00000.png",
                             "with --t4-dataset the dataset root is the image root")


class VideoSourceTest(unittest.TestCase):
    def _write_video(self, path: Path, n=6):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 10.0, (32, 24))
        if not writer.isOpened():
            self.skipTest("no OpenCV video encoder available")
        for i in range(n):
            writer.write(np.full((24, 32, 3), i * 10, dtype=np.uint8))
        writer.release()

    def test_stride_keeps_source_frame_numbering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            self._write_video(path)
            frames = list(build_frame_source(
                {"kind": "video", "uri": str(path), "stride": 2}))
            self.assertEqual([f.frame_id for f in frames][:3],
                             ["000000", "000002", "000004"])
            self.assertEqual([f.frame_index for f in frames][:3], [0, 2, 4])
            self.assertEqual(frames[0].source["kind"], "video")
            self.assertEqual(frames[0].source["stride"], 2)
            self.assertIsNone(frames[0].realpath, "a video frame is not a file")
            self.assertEqual(frames[0].rel_path, "clip.avi#000000")

    def test_max_frames_and_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            self._write_video(path)
            frames = list(build_frame_source(
                {"kind": "video", "uri": str(path), "start": 2, "max_frames": 2}))
            self.assertEqual([f.frame_id for f in frames], ["000002", "000003"])

    def test_materialize_gives_every_config_the_same_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            self._write_video(path, n=3)
            src = build_frame_source({"kind": "video", "uri": str(path)})
            out = Path(tmp) / "frames"
            mat = materialize(src, str(out), verbose=False)
            self.assertIsInstance(mat, MaterializedSource)
            first = list(mat)
            second = list(MaterializedSource(path=str(out)))
            self.assertEqual([f.frame_id for f in first], ["000000", "000001", "000002"])
            for a, b in zip(first, second):
                self.assertTrue(np.array_equal(a.image, b.image))
                self.assertTrue(a.realpath.endswith(f"{a.frame_id}.png"))
                self.assertEqual(a.source["origin_ref"], f"clip.avi#{a.frame_id}")
            # a second materialize of the same source reuses the extraction
            again = materialize(build_frame_source({"kind": "video", "uri": str(path)}),
                                str(out), verbose=False)
            self.assertEqual(len(list(again)), 3)


class T4SourceTest(unittest.TestCase):
    def _dataset(self, root: Path, channels=("CAM_FRONT", "CAM_FRONT_FAR")):
        ann = root / "annotation"
        ann.mkdir(parents=True)
        sensors, calibs, sample_data = [], [], []
        for ci, ch in enumerate(channels):
            sensors.append({"token": f"sensor-{ci}", "channel": ch, "modality": "camera"})
            calibs.append({"token": f"calib-{ci}", "sensor_token": f"sensor-{ci}"})
            for i in range(2):
                rel = f"data/{ch}/{i:05d}.png"
                write_image(root / rel, value=i)
                sample_data.append({"token": f"sd-{ch}-{i}", "filename": rel,
                                    "calibrated_sensor_token": f"calib-{ci}",
                                    "timestamp": 1000 + i})
        sensors.append({"token": "sensor-lidar", "channel": "LIDAR_TOP",
                        "modality": "lidar"})
        calibs.append({"token": "calib-lidar", "sensor_token": "sensor-lidar"})
        sample_data.append({"token": "sd-lidar", "filename": "data/LIDAR_TOP/0.pcd",
                            "calibrated_sensor_token": "calib-lidar", "timestamp": 1000})
        (ann / "sensor.json").write_text(json.dumps(sensors))
        (ann / "calibrated_sensor.json").write_text(json.dumps(calibs))
        (ann / "sample_data.json").write_text(json.dumps(sample_data))

    def test_camera_rows_only_and_per_channel_frame_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._dataset(root)
            frames = list(T4DatasetSource(root=str(root)))
            self.assertEqual([f.frame_id for f in frames],
                             ["CAM_FRONT/00000", "CAM_FRONT/00001",
                              "CAM_FRONT_FAR/00000", "CAM_FRONT_FAR/00001"],
                             "multi-channel runs namespace frame ids by channel")
            self.assertEqual(frames[0].sample_data_token, "sd-CAM_FRONT-0")
            self.assertEqual(frames[0].channel, "CAM_FRONT")
            self.assertEqual(frames[0].timestamp_us, 1000)
            self.assertEqual(frames[0].rel_path, "data/CAM_FRONT/00000.png")

    def test_single_channel_keeps_plain_frame_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._dataset(root)
            frames = list(T4DatasetSource(root=str(root), channels="CAM_FRONT"))
            self.assertEqual([f.frame_id for f in frames], ["00000", "00001"])


class RosbagHelpersTest(unittest.TestCase):
    """The bag reader needs a sourced ROS 2, but its decode/naming rules do not."""

    def test_channel_of_topic(self):
        self.assertEqual(
            channel_of_topic("/sensing/camera/camera6/image_rect_color"), "camera6")
        self.assertEqual(channel_of_topic("/camera/CAM_FRONT/image_raw"), "CAM_FRONT")
        self.assertEqual(channel_of_topic("/my_images"), "my_images")

    def test_decode_bgr8_and_rgb8(self):
        h, w = 2, 3
        rgb = np.dstack([np.full((h, w), 10, np.uint8),
                         np.full((h, w), 20, np.uint8),
                         np.full((h, w), 30, np.uint8)])
        msg = SimpleNamespace(height=h, width=w, encoding="rgb8", data=rgb.tobytes())
        out = decode_image_msg(msg, "sensor_msgs/msg/Image")
        self.assertEqual(out.shape, (h, w, 3))
        self.assertEqual(tuple(out[0, 0]), (30, 20, 10), "rgb8 must come back as BGR")

        bgr_msg = SimpleNamespace(height=h, width=w, encoding="bgr8", data=rgb.tobytes())
        out = decode_image_msg(bgr_msg, "sensor_msgs/msg/Image")
        self.assertEqual(tuple(out[0, 0]), (10, 20, 30))

    def test_decode_mono8_expands_to_three_channels(self):
        msg = SimpleNamespace(height=2, width=2, encoding="mono8",
                              data=np.full((2, 2), 7, np.uint8).tobytes())
        out = decode_image_msg(msg, "sensor_msgs/msg/Image")
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertEqual(tuple(out[0, 0]), (7, 7, 7))

    def test_decode_compressed(self):
        img = np.full((4, 4, 3), 128, np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        self.assertTrue(ok)
        msg = SimpleNamespace(data=buf.tobytes())
        out = decode_image_msg(msg, "sensor_msgs/msg/CompressedImage")
        self.assertEqual(out.shape, (4, 4, 3))

    def test_unsupported_encoding_is_explicit(self):
        msg = SimpleNamespace(height=1, width=1, encoding="yuv422", data=b"\x00\x00")
        with self.assertRaises(SystemExit) as ctx:
            decode_image_msg(msg, "sensor_msgs/msg/Image")
        self.assertIn("unsupported image encoding", str(ctx.exception))


class FactoryTest(unittest.TestCase):
    def test_unknown_kind_lists_options(self):
        with self.assertRaises(SystemExit) as ctx:
            build_frame_source({"kind": "hdf5"})
        self.assertIn("unknown frame source kind", str(ctx.exception))

    def test_missing_kind(self):
        with self.assertRaises(SystemExit):
            build_frame_source({})


if __name__ == "__main__":
    unittest.main()
