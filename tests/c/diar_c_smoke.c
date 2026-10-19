// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "nemo_speech/diar.h"

int
main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: diar_c_smoke <model.gguf>\n");
        return 2;
    }

    nemo_speech_diar_model_config cfg = {0};
    cfg.size = sizeof(cfg);
    cfg.model_path = argv[1];
    cfg.gpu = -1;

    nemo_speech_diar_model* model = NULL;
    if (nemo_speech_diar_create(NULL, &model) != NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT ||
        !nemo_speech_asr_last_error()[0]) {
        fprintf(stderr, "diar C ABI did not report invalid arguments\n");
        return 1;
    }
    if (nemo_speech_diar_create(&cfg, &model) != NEMO_SPEECH_ASR_OK) {
        fprintf(stderr, "diar C ABI model load failed: %s\n", nemo_speech_asr_last_error());
        return 1;
    }
    if (nemo_speech_diar_num_speakers(model) != 8 ||
        fabs(nemo_speech_diar_seconds_per_frame(model) - 0.01) > 1e-6) {
        fprintf(stderr, "unexpected Sortformer V3 model geometry\n");
        nemo_speech_diar_destroy(model);
        return 1;
    }

    nemo_speech_diar_stream* stream = NULL;
    if (nemo_speech_diar_stream_open(model, &stream) != NEMO_SPEECH_ASR_OK || !stream) {
        fprintf(stderr, "diar C ABI stream open failed: %s\n", nemo_speech_asr_last_error());
        nemo_speech_diar_destroy(model);
        return 1;
    }
    nemo_speech_diar_stream_close(stream);
    nemo_speech_diar_destroy(model);
    return 0;
}
