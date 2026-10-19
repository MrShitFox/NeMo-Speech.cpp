// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#include "nn.h"

#include "runtime.h"

namespace ggml_runtime {

void
Conv1D::define_tensors(Session* session) {
    session->model_tensor_container->create_tensor_3d(
        weight_name, GGML_TYPE_F16, kernel_size, in_channels, out_channels);
    if (use_bias) {
        session->model_tensor_container->create_tensor_1d(bias_name, GGML_TYPE_F32, out_channels);
    }
}

TensorBag
Conv1D::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);

    auto weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
    auto out_tensor = conv_1d(
        bf_ctx.ctx, weight_tensor.tensor, input_tensor.tensor, stride, padding, dilation);
    if (use_bias) {
        auto bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        // Reshape (OC) to (1, OC, 1) for broadcast over output length and batch.
        auto bias_reshape =
            ggml_reshape_3d(bf_ctx.ctx, bias_tensor.tensor, 1, bias_tensor.tensor->ne[0], 1);
        out_tensor = ggml_add(bf_ctx.ctx, out_tensor, bias_reshape);
    }
    TensorBag output_tensors;
    output_tensors.add_tensor(ggml_bf_tensor(out_tensor, bf_ctx.buft));
    return output_tensors;
}

void
Conv1D::set_data(Session* session) {
    session->load_weight(weight_name);
    if (use_bias) {
        session->load_weight(bias_name);
    }
}


void
Linear::define_tensors(Session* session) {
    // Honor the stored quantization type so quantized GGUFs load with
    // matching byte sizes. Bias always stays at the stored type too
    // (typically F32 or F16 — we never quantize biases in the converter).
    ggml_type weight_type = session->gguf_loader->get_tensor_type(weight_name);
    this->weight = session->model_tensor_container->create_tensor_4d(
        weight_name, weight_type, in_features, out_features, 1, 1);
    if (use_bias) {
        ggml_type bias_type = session->gguf_loader->get_tensor_type(bias_name);
        this->bias = session->model_tensor_container->create_tensor_4d(
            bias_name, bias_type, out_features, 1, 1, 1);
    }
}

TensorBag
Linear::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
    ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);

    ggml_tensor* matmul_ret = ggml_mul_mat(bf_ctx.ctx, weight_tensor.tensor, input_tensor.tensor);

    ggml_tensor* output_tensor = nullptr;
    if (use_bias) {
        ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        const bool fuse_bf16_bias = session->params.use_gpu &&
                                    weight_tensor.tensor->type == GGML_TYPE_BF16 &&
                                    input_tensor.tensor->ne[2] * input_tensor.tensor->ne[3] > 1;
        output_tensor = fuse_bf16_bias
                            ? ggml_add(bf_ctx.ctx, matmul_ret, bias_tensor.tensor)
                            : ggml_add_inplace(bf_ctx.ctx, matmul_ret, bias_tensor.tensor);
    } else {
        output_tensor = matmul_ret;
    }
    auto output_tensor_bag = TensorBag();
    output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
    return output_tensor_bag;
}

void
Linear::set_data(Session* session) {
    session->load_weight(weight_name);
    if (use_bias) {
        // Bias must be F32: the graph adds it to an F32 activation.
        ggml_type bias_tensor_type = session->gguf_loader->get_tensor_type(bias_name);
        if (bias_tensor_type != GGML_TYPE_F32) {
            throw std::runtime_error(
                "linear(" + name + ") bias " + bias_name + " is not f32 (got type " +
                std::to_string(int(bias_tensor_type)) + ")");
        }
        session->load_weight(bias_name);
    }
}


void
LayerNorm::define_tensors(Session* session) {
    session->model_tensor_container->create_tensor_4d(
        weight_name, GGML_TYPE_F32, input_shape[0], input_shape[1], input_shape[2], 1);
    session->model_tensor_container->create_tensor_4d(
        bias_name, GGML_TYPE_F32, input_shape[0], 1, 1, 1);
}

TensorBag
LayerNorm::build_graph(
    Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) {
    auto input_tensor = input_tensors.get_tensor(0);
    auto weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
    auto bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);

    ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);
    // In-place views prevent CUDA from fusing NORM+MUL+ADD.
    auto norm_tensor = ggml_norm(bf_ctx.ctx, input_tensor.tensor, 1e-5);
    auto scaled_tensor = ggml_mul(bf_ctx.ctx, norm_tensor, weight_tensor.tensor);
    auto output_tensor = ggml_add(bf_ctx.ctx, scaled_tensor, bias_tensor.tensor);
    auto output_tensor_bag = TensorBag();
    output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
    return output_tensor_bag;
}

void
LayerNorm::set_data(Session* session) {
    // γ and β must be F32: the graph multiplies/adds them against an F32 norm.
    ggml_type weight_tensor_type = session->gguf_loader->get_tensor_type(weight_name);
    ggml_type bias_tensor_type = session->gguf_loader->get_tensor_type(bias_name);
    if (weight_tensor_type != GGML_TYPE_F32 || bias_tensor_type != GGML_TYPE_F32) {
        throw std::runtime_error(
            "layer norm(" + name + ") weight " + weight_name + " or bias " + bias_name +
            " is not f32 (got types " + std::to_string(int(weight_tensor_type)) + "/" +
            std::to_string(int(bias_tensor_type)) + ")");
    }
    session->load_weight(weight_name);
    session->load_weight(bias_name);
}

}  // namespace ggml_runtime
