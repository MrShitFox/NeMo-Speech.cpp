<div align="center">

# NeMo-Speech.cpp

**A standalone C++ runtime for NVIDIA Nemotron 3 Diarization (Sortformer V3).**

</div>

This repository builds only the Nemotron 3 speaker diarization engine, its
public C API, and a small command-line client. It accepts local GGUF model files
and runs on CPU, Vulkan, CUDA, or Metal builds. The project does not download
models or require SentencePiece, ASR, TTS, NMT, or gRPC dependencies.

## Build

Initialize the ggml source submodule and choose one backend preset:

```bash
git submodule update --init llama.cpp
scripts/configure.sh cpu-diar
cmake --build --preset cpu-diar
```

The available presets are `cpu-diar`, `vulkan-diar`, `cuda-diar`, and
`metal-diar`. CUDA needs a CUDA toolkit; Vulkan needs Vulkan development files,
`glslc`, and SPIR-V headers; Metal needs macOS and Xcode Command Line Tools.
See [Build](docs/build.md) for prerequisites and backend details. Only report a
backend as built after compiling it with that backend's toolchain.

To build the model-free tests, configure with `-DBUILD_TESTING=ON`, then build
and run CTest:

```bash
scripts/configure.sh cpu-diar -DBUILD_TESTING=ON
cmake --build --preset cpu-diar
ctest --test-dir build/cpu-diar --output-on-failure
```

The model-backed C API test is registered when the local Q8_0 model exists at
`models/Nemotron-3-Diarization.q8_0.gguf`. GGUF files are ignored by Git.

## Run diarization

Place a Nemotron 3 Diarization GGUF at a local path, then run:

```bash
nemo-speech diarize test.wav \
  --model models/Nemotron-3-Diarization.q8_0.gguf
```

The CLI reads WAV files or directories and supports text, JSON, and RTTM output.
Streaming is the default; use `--offline` for full-attention processing of
shorter recordings. `--preset`, `--onset`, `--offset`, `--pad-onset`,
`--pad-offset`, `--min-duration-on`, and `--min-duration-off` expose the
existing geometry and segmentation controls. Run `nemo-speech diarize --help`
for all options.

The public C interface is in [`include/nemo_speech/diar.h`](include/nemo_speech/diar.h).
It supports model loading, offline inference, streaming audio, retained frame
probabilities, and speaker segments. See [C API guide](docs/sdk.md).

## Project scope and future HTTP API

The old HTTP and WebSocket server depended on ASR engine registries and
VoiceChat pipelines, so it was removed with those subsystems. A future VoiceID
REST service should be a separate optional executable that wraps the public
diarization C API and owns its own transport and request lifecycle. No network
library is part of this project today.

## Documentation

- [Build guide](docs/build.md)
- [CLI guide](docs/cli.md)
- [C API guide](docs/sdk.md)
- [Configuration example](config/diar.example.yaml)
- [Task 02 baseline](docs/VOICEID_TASK02_REPORT.md)
- [Task 03 cleanup report](docs/VOICEID_TASK03_REPORT.md)

## License

NVIDIA-authored project code is licensed under [Apache 2.0](LICENSE), with
copyright details in [NOTICE](NOTICE). See
[third-party notices](THIRD_PARTY_NOTICES.md) for ggml, retained derived code,
and the sample audio attribution.
