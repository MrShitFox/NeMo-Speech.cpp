// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

namespace nemo_speech::c_api {

void set_last_error(const std::string& message);
const char* last_error();
const char* version();

}  // namespace nemo_speech::c_api
