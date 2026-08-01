// Minimal TensorRT engine runner for the tlr_autolabel scripts.
//
//   one-shot: build/trt_run <engine> <input.f32> <output.f32>
//   serve   : build/trt_run <engine> serve
//     prints "INPUT d0 d1 ..." / "OUTPUT d0 d1 ..." / "READY" on stdout, then
//     reads "<in.f32> <out.f32>" lines from stdin, answering "DONE" (or "ERR")
//     per line. Keeps the engine deserialized between calls, which matters:
//     deserialization takes ~1-2 s while one inference is ~10 ms.
//
// Build: g++ -O2 tools/trt_run.cpp -o build/trt_run -I/usr/local/cuda/include \
//        -L/usr/local/cuda/lib64 -lnvinfer -lcudart
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
  }
} gLogger;

static size_t volume(const Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
  return v;
}

static bool has_dynamic_dim(const Dims& d) {
  for (int i = 0; i < d.nbDims; ++i) {
    if (d.d[i] < 0) return true;
  }
  return false;
}

static std::string dims_string(const Dims& d) {
  std::ostringstream dims;
  for (int k = 0; k < d.nbDims; ++k) dims << d.d[k] << (k + 1 < d.nbDims ? " " : "");
  return dims.str();
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s engine (in.f32 out.f32 | serve)\n", argv[0]); return 1; }
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (blob.empty()) { fprintf(stderr, "cannot read engine %s\n", argv[1]); return 1; }

  auto* runtime = createInferRuntime(gLogger);
  auto* engine = runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (!engine) { fprintf(stderr, "deserialize failed\n"); return 1; }
  auto* ctx = engine->createExecutionContext();

  int nio = engine->getNbIOTensors();
  std::vector<std::string> names(nio);
  std::vector<bool> is_input(nio);
  for (int i = 0; i < nio; ++i) {
    const char* name = engine->getIOTensorName(i);
    names[i] = name;
    is_input[i] = engine->getTensorIOMode(name) == TensorIOMode::kINPUT;
    if (is_input[i]) {
      Dims d = engine->getTensorShape(name);
      if (has_dynamic_dim(d)) {
        Dims opt = engine->getProfileShape(name, 0, OptProfileSelector::kOPT);
        if (has_dynamic_dim(opt) || !ctx->setInputShape(name, opt)) {
          fprintf(stderr, "cannot resolve dynamic input shape for %s\n", name);
          return 1;
        }
      }
    }
  }

  std::vector<void*> devbuf(nio);
  int in_idx = -1, out_idx = -1;
  size_t in_count = 0, out_count = 0;
  for (int i = 0; i < nio; ++i) {
    const char* name = names[i].c_str();
    Dims d = ctx->getTensorShape(name);
    if (d.nbDims < 0 || has_dynamic_dim(d)) d = engine->getTensorShape(name);
    if (d.nbDims < 0 || has_dynamic_dim(d)) {
      fprintf(stderr, "unresolved tensor shape for %s: %s\n", name, dims_string(d).c_str());
      return 1;
    }
    size_t count = volume(d);
    bool is_in = is_input[i];
    printf("%s %s\n", is_in ? "INPUT" : "OUTPUT", dims_string(d).c_str());
    cudaMalloc(&devbuf[i], count * sizeof(float));
    ctx->setTensorAddress(name, devbuf[i]);
    if (is_in) { in_idx = i; in_count = count; }
    else { out_idx = i; out_count = count; }
  }
  fflush(stdout);
  if (in_idx < 0 || out_idx < 0) { fprintf(stderr, "need 1 input + 1 output\n"); return 1; }

  cudaStream_t stream; cudaStreamCreate(&stream);
  std::vector<float> in(in_count), out(out_count);

  auto run_one = [&](const std::string& in_path, const std::string& out_path) -> bool {
    std::ifstream fi(in_path, std::ios::binary);
    fi.read(reinterpret_cast<char*>(in.data()), in_count * sizeof(float));
    if ((size_t)fi.gcount() != in_count * sizeof(float)) {
      fprintf(stderr, "input %s: wrong size\n", in_path.c_str()); return false;
    }
    cudaMemcpy(devbuf[in_idx], in.data(), in_count * sizeof(float), cudaMemcpyHostToDevice);
    if (!ctx->enqueueV3(stream)) { fprintf(stderr, "enqueue failed\n"); return false; }
    cudaStreamSynchronize(stream);
    cudaMemcpy(out.data(), devbuf[out_idx], out_count * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream fo(out_path, std::ios::binary);
    fo.write(reinterpret_cast<char*>(out.data()), out_count * sizeof(float));
    return true;
  };

  if (std::strcmp(argv[2], "serve") == 0) {
    printf("READY\n"); fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      std::istringstream ss(line);
      std::string a, b; ss >> a >> b;
      printf("%s\n", run_one(a, b) ? "DONE" : "ERR"); fflush(stdout);
    }
    return 0;
  }
  if (argc != 4) { fprintf(stderr, "one-shot needs out path\n"); return 1; }
  return run_one(argv[2], argv[3]) ? 0 : 1;
}
