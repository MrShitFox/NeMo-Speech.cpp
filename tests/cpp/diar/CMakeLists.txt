# Standalone Nemotron 3 / Sortformer V3 regression tests.
add_executable(test_sortformer_parity test_sortformer_parity.cpp)
target_link_libraries(test_sortformer_parity PRIVATE nemo_speech_diar)

add_executable(test_diar_streaming test_diar_streaming.cpp)
target_link_libraries(test_diar_streaming PRIVATE nemo_speech_diar)

add_executable(test_diar_state test_diar_state.cpp)
target_include_directories(test_diar_state PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_diar_state PRIVATE nemo_speech_diar)
add_test(NAME diar_state COMMAND test_diar_state)

add_executable(test_mel_features test_mel_features.cpp)
target_include_directories(test_mel_features PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_mel_features PRIVATE nemo_speech_audio_frontend)
add_test(NAME mel_features COMMAND test_mel_features)

add_executable(diar_c_smoke ${CMAKE_SOURCE_DIR}/tests/c/diar_c_smoke.c)
target_link_libraries(diar_c_smoke PRIVATE nemo_speech_diar_c)
if(GGML_CUDA)
    target_link_libraries(diar_c_smoke PRIVATE CUDA::cuda_driver)
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/models/Nemotron-3-Diarization.q8_0.gguf")
    add_test(NAME diar_c_api_model
        COMMAND diar_c_smoke "${CMAKE_SOURCE_DIR}/models/Nemotron-3-Diarization.q8_0.gguf")
endif()
