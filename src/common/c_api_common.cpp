// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "c_api_internal.h"

namespace nemo_speech::c_api {
namespace {
thread_local std::string last_error_message;
}  // namespace

void
set_last_error(const std::string& message) {
    last_error_message = message;
}

const char*
last_error() {
    return last_error_message.c_str();
}

const char*
version() {
#ifndef NEMO_SPEECH_VERSION_STR
#define NEMO_SPEECH_VERSION_STR "0.0.0"
#endif
    return "nemo-speech-asr " NEMO_SPEECH_VERSION_STR;
}
}  // namespace nemo_speech::c_api
