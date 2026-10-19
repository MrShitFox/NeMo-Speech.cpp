// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared C ABI status, error, and version declarations. The ASR-prefixed
// names remain for compatibility with the original diarization ABI.
#ifndef NEMO_SPEECH_COMMON_H
#define NEMO_SPEECH_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(NEMO_SPEECH_ASR_BUILD) || defined(NEMO_SPEECH_DIAR_BUILD)
#define NEMO_SPEECH_ASR_API __declspec(dllexport)
#else
#define NEMO_SPEECH_ASR_API __declspec(dllimport)
#endif
#else
#define NEMO_SPEECH_ASR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nemo_speech_asr_status {
    NEMO_SPEECH_ASR_OK = 0,
    NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT = 1,
    NEMO_SPEECH_ASR_ERROR_OUT_OF_MEMORY = 2,
    NEMO_SPEECH_ASR_ERROR_RUNTIME = 3,
    NEMO_SPEECH_ASR_ERROR_CANCELLED = 4,
} nemo_speech_asr_status;

// Thread-local message for the most recent failed diarization C ABI call on
// this thread. Valid until the next call into the API.
NEMO_SPEECH_ASR_API const char* nemo_speech_asr_last_error(void);
NEMO_SPEECH_ASR_API const char* nemo_speech_asr_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NEMO_SPEECH_COMMON_H
