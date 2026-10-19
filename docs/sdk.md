# Public C API

The installed SDK exposes the standalone Nemotron 3 Diarization C API in
`nemo_speech/diar.h`, together with shared status and error declarations in
`nemo_speech/common.h`. The library keeps the established `nemo_speech_diar_*`
symbols and the legacy ASR-prefixed status, error, and version symbols required
by the ABI.

The model config takes a local GGUF path. Set `gpu` to `-1` for CPU or to the
selected device index for a backend-enabled build. Model handles outlive their
streams. Each stream is driven by one thread; separate streams may be used
concurrently.

```c
#include <stdio.h>
#include "nemo_speech/diar.h"

int main(void) {
    nemo_speech_diar_model_config config = {0};
    config.size = sizeof(config);
    config.model_path = "models/Nemotron-3-Diarization.q8_0.gguf";
    config.gpu = -1;

    nemo_speech_diar_model *model = NULL;
    if (nemo_speech_diar_create(&config, &model) != NEMO_SPEECH_ASR_OK) {
        fprintf(stderr, "%s\n", nemo_speech_asr_last_error());
        return 1;
    }

    nemo_speech_diar_stream *stream = NULL;
    nemo_speech_asr_status status = nemo_speech_diar_stream_open(model, &stream);
    /* Push mono float32 samples with nemo_speech_diar_stream_push_f32(). */
    if (status == NEMO_SPEECH_ASR_OK)
        status = nemo_speech_diar_stream_finish(stream);
    if (status != NEMO_SPEECH_ASR_OK)
        fprintf(stderr, "%s\n", nemo_speech_asr_last_error());

    nemo_speech_diar_stream_close(stream);
    nemo_speech_diar_destroy(model);
    return status == NEMO_SPEECH_ASR_OK ? 0 : 1;
}
```

`nemo_speech_diar_offline_f32()` runs full-attention inference on a complete
audio buffer. Streaming calls accept mono float32 input, resample supported
rates to the model's rate, and expose native-cadence frame probabilities and
speaker segments. See the declarations and ownership comments in
[`include/nemo_speech/diar.h`](../include/nemo_speech/diar.h).

## CMake package

After installing the project, a consumer can locate the diarization library
with:

```cmake
find_package(NeMoSpeech REQUIRED COMPONENTS Diarization)
target_link_libraries(my_app PRIVATE NeMoSpeech::Diarization)
```

The package installs `nemo_speech/diar.h`, `nemo_speech/common.h`, and the
shared libraries needed by the CLI and C API.
