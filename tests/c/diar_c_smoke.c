// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <math.h>
#include <stdlib.h>
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

    size_t state_size = 0;
    if (nemo_speech_diar_stream_state_size(stream, &state_size) != NEMO_SPEECH_ASR_OK ||
        state_size < 108) {
        fprintf(stderr, "state size query failed\n");
        nemo_speech_diar_stream_close(stream);
        nemo_speech_diar_destroy(model);
        return 1;
    }
    void* snapshot = malloc(state_size);
    size_t written = 0;
    if (!snapshot || nemo_speech_diar_stream_save_state(
                         stream, snapshot, state_size - 1, &written) !=
                         NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT ||
        written != 0 ||
        nemo_speech_diar_stream_save_state(stream, snapshot, state_size, &written) !=
                         NEMO_SPEECH_ASR_OK ||
        written != state_size) {
        fprintf(stderr, "state save failed\n");
        free(snapshot);
        nemo_speech_diar_stream_close(stream);
        nemo_speech_diar_destroy(model);
        return 1;
    }
    nemo_speech_diar_stream* restored = NULL;
    if (nemo_speech_diar_stream_open(model, &restored) != NEMO_SPEECH_ASR_OK || !restored ||
        nemo_speech_diar_stream_load_state(restored, snapshot, written) != NEMO_SPEECH_ASR_OK) {
        fprintf(stderr, "state load failed\n");
        free(snapshot);
        nemo_speech_diar_stream_close(restored);
        nemo_speech_diar_stream_close(stream);
        nemo_speech_diar_destroy(model);
        return 1;
    }
    if (nemo_speech_diar_stream_load_state(restored, snapshot, written) !=
        NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "active stream accepted a second state import\n");
        free(snapshot);
        nemo_speech_diar_stream_close(restored);
        nemo_speech_diar_stream_close(stream);
        nemo_speech_diar_destroy(model);
        return 1;
    }
    free(snapshot);
    nemo_speech_diar_stream_close(restored);
    nemo_speech_diar_stream_close(stream);
    nemo_speech_diar_destroy(model);
    return 0;
}
