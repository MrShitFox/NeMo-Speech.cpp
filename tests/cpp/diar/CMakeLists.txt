# Standalone Sortformer regression tests. ASR and decoder tests remain in the
# separate asr/ directory and are not configured for diar-only builds.
add_executable(test_sortformer_parity ../asr/test_sortformer_parity.cpp)
target_link_libraries(test_sortformer_parity PRIVATE nemo_speech_diar)

add_executable(test_diar_streaming ../asr/test_diar_streaming.cpp)
target_link_libraries(test_diar_streaming PRIVATE nemo_speech_diar)

add_executable(test_diar_state ../asr/test_diar_state.cpp)
target_link_libraries(test_diar_state PRIVATE nemo_speech_diar)
add_test(NAME diar_state COMMAND test_diar_state)

add_executable(test_mel_features ../asr/test_mel_features.cpp)
target_link_libraries(test_mel_features PRIVATE nemo_speech_audio_frontend)
add_test(NAME mel_features COMMAND test_mel_features)

# This C source verifies that the installed public header is C-compatible and
# that the diar C ABI loads the local V3 model without the ASR C API library.
add_executable(diar_c_smoke ${CMAKE_SOURCE_DIR}/tests/c/diar_c_smoke.c)
target_link_libraries(diar_c_smoke PRIVATE nemo_speech_diar_c)
if(GGML_CUDA)
    target_link_libraries(diar_c_smoke PRIVATE CUDA::cuda_driver)
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/models/Nemotron-3-Diarization.q8_0.gguf")
    add_test(NAME diar_c_api_model
        COMMAND diar_c_smoke "${CMAKE_SOURCE_DIR}/models/Nemotron-3-Diarization.q8_0.gguf")
endif()
