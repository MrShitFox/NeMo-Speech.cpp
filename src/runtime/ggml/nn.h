// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#pragma once

#include "runtime.h"

namespace ggml_runtime {

class Conv1D : public Module {
   public:
    Conv1D(
        const std::string& name, int in_channels, int out_channels, int kernel_size, int stride = 1,
        int padding = 0, int dilation = 1, bool use_bias = true)
        : name(name), in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation),
          use_bias(use_bias) {
        weight_name = this->name + ".weight";
        bias_name = this->name + ".bias";
    }

    void define_tensors(Session* session) override;
    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;
    void set_data(Session* session) override;

   private:
    std::string name;
    std::string weight_name;
    std::string bias_name;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    bool use_bias;
};

class Linear : public Module {
   public:
    Linear(const std::string& name, int in_features, int out_features, bool use_bias = true)
        : name(name), in_features(in_features), out_features(out_features), use_bias(use_bias) {
        weight_name = this->name + ".weight";
        bias_name = this->name + ".bias";
    }

    void define_tensors(Session* session) override;
    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;
    void set_data(Session* session) override;

   private:
    std::string name;
    std::string weight_name;
    std::string bias_name;
    int in_features;
    int out_features;
    bool use_bias;
    ggml_bf_tensor weight = ggml_bf_tensor(nullptr, nullptr);
    ggml_bf_tensor bias = ggml_bf_tensor(nullptr, nullptr);
};

class LayerNorm : public Module {
   public:
    LayerNorm(const std::string& name, const int64_t (&input_shape)[GGML_MAX_DIMS]) : name(name) {
        for (int i = 0; i < GGML_MAX_DIMS; i++)
            this->input_shape[i] = input_shape[i];
        weight_name = this->name + ".weight";
        bias_name = this->name + ".bias";
    }

    void define_tensors(Session* session) override;
    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;
    void set_data(Session* session) override;

   private:
    std::string name;
    std::string weight_name;
    std::string bias_name;
    int64_t input_shape[GGML_MAX_DIMS] = {0};
};

}  // namespace ggml_runtime
