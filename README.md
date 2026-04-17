# ggml-ane

Apple Neural Engine (ANE) backend for [ggml](https://github.com/ggml-org/ggml). Offloads `MUL_MAT` operations to the ANE via private CoreML APIs, reaching 3.5-4 TFLOPS on M-series chips for large prefill-size matmuls.

## How it works

1. Weight matrices are compiled into MIL (Machine Learning Intermediate Language) programs at runtime
2. Each MIL program wraps a `conv1x1` operation (equivalent to matmul) with weights baked in
3. Input/output tensors are passed through IOSurface shared memory
4. The ANE hardware executes the compiled model with no CPU involvement

## Requirements

- macOS 15+ (Sequoia) or iOS 18+
- Apple Silicon (M1/M2/M3/M4)
- Must be built inside a ggml source tree (uses ggml's CMake build system)

## Build

```bash
# Clone ggml and place this backend inside it
cd ggml/ggml/src/
ln -s /path/to/ggml-ane-clean/ggml/src/ggml-ane ggml-ane
cp /path/to/ggml-ane-clean/ggml/include/ggml-ane.h ../include/

# Add to ggml's CMakeLists.txt:
#   add_subdirectory(ggml-ane)

# Build
cmake -B build && cmake --build build
```

## Limitations

- **MUL_MAT only** — other ops fall through to CPU/Metal
- **2D tensors only** — batched matmul (ne[2] > 1) not supported
- **fp16 precision** — all computation is in fp16; weights are converted on first use
- **Minimum dimensions** — M, K, N >= 64 (ANE is inefficient for small matmuls)
- **Compile time** — first inference is slow (~50ms per unique weight shape) due to MIL compilation; subsequent calls use cached kernels
- **Private API dependency** — uses `_ANEModel`, `_ANERequest`, `_ANEIOSurfaceObject` which are undocumented Apple frameworks; may break across macOS updates

## Architecture

```
ggml-ane.mm      ggml backend interface (graph_compute, supports_op, device)
ane_bridge.m     Low-level ANE access (compile MIL, create IOSurfaces, eval)
ane_bridge.h     C API for the bridge
ggml-ane.h       Public ggml backend header
```

## License

MIT
