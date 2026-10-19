# Build from source

The standalone project has four CMake presets: `cpu-diar`, `vulkan-diar`,
`cuda-diar`, and `metal-diar`. All configure the Nemotron 3 Diarization
runtime, public C API, and `nemo-speech` CLI.

## Common requirements

- Git and the `llama.cpp` submodule, which provides ggml/GGUF;
- CMake 3.26 or newer and Ninja;
- C and C++17 compilers;
- the SDK/toolkit for the selected accelerator backend.

No SentencePiece, protobuf, gRPC, ASR, TTS, NMT, or Python package is needed
to configure or build the runtime.

```bash
git submodule update --init llama.cpp
scripts/configure.sh cpu-diar
cmake --build --preset cpu-diar
```

## Backend requirements

| Preset | Requirements | Build option |
|---|---|---|
| `cpu-diar` | Standard C/C++ compiler | ggml CPU backend |
| `vulkan-diar` | Vulkan headers and loader, `glslc`, SPIR-V headers | `GGML_VULKAN=ON` |
| `cuda-diar` | CUDA toolkit with `nvcc` and a compatible driver | `GGML_CUDA=ON`; CUDA graphs enabled |
| `metal-diar` | macOS and Xcode Command Line Tools | `GGML_METAL=ON` |

CUDA architecture selection is available through CMake, for example:

```bash
scripts/configure.sh cuda-diar -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build --preset cuda-diar
```

The CPU and CUDA presets use the repository's pinned ggml patch series. Vulkan
and Metal presets build the pinned ggml source without that patch series.
`GGML_LLAMAFILE` is enabled by all presets. CPU is the only backend validated
on every host; the other presets require their native build environment.

## Tests

The presets default `BUILD_TESTING` to `OFF`. Enable tests when configuring:

```bash
scripts/configure.sh cpu-diar -DBUILD_TESTING=ON
cmake --build --preset cpu-diar
ctest --test-dir build/cpu-diar --output-on-failure
```

The `diar_state` and `mel_features` tests do not need a model. If
`models/Nemotron-3-Diarization.q8_0.gguf` exists at configure time, CMake also
registers `diar_c_api_model`, which loads it through the public C API. Test WAV
fixtures live in `test_files/diar/`.

## Local model and command line

Model conversion and downloading are outside this runtime. Put a compatible
Nemotron 3 Diarization GGUF on disk and pass its path to the CLI:

```bash
nemo-speech diarize test_files/diar/jfk.wav \
  --model models/Nemotron-3-Diarization.q8_0.gguf \
  --format json
```

See [CLI guide](cli.md) for streaming, offline, output, and segmentation
options. GGUF files and build directories are ignored by `.gitignore`.
