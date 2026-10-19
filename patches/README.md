# ggml patch series

The runtime builds ggml from the pinned `llama.cpp` submodule. CPU and CUDA
presets apply the NVIDIA ggml patch series in this directory; Vulkan and Metal
presets use the pinned upstream ggml source unchanged. No llama.cpp inference
target is built.

CMake applies the series during configuration. `scripts/llama-patches.sh` is
maintainer tooling for editing, exporting, and checking the pinned patch set.
It is not needed to build Vulkan or Metal configurations.

The patch series retains ggml changes required by Nemotron 3's transformer,
convolution, normalization, and CUDA execution paths. A patch is removed only
when its affected operation has been verified as outside the V3 build/runtime
dependency graph. Patch files and their order are listed in `series`.

To verify that all listed patches apply to the pinned submodule:

```bash
git submodule update --init llama.cpp
scripts/llama-patches.sh check
```

For patch development, use `scripts/llama-patches.sh edit`, build against the
printed `NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR`, then run `export` and `check` as
described by the script's help.
