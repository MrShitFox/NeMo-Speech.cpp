# Standalone Nemotron 3 / Sortformer V3 regression tests.
add_executable(test_sortformer_parity test_sortformer_parity.cpp)
target_link_libraries(test_sortformer_parity PRIVATE nemo_speech_diar)

add_executable(test_diar_streaming test_diar_streaming.cpp)
target_link_libraries(test_diar_streaming PRIVATE nemo_speech_diar)

add_executable(test_diar_snapshot test_diar_snapshot.cpp)
target_link_libraries(test_diar_snapshot PRIVATE nemo_speech_diar)

add_executable(diar_enrollment_probe diar_enrollment_probe.cpp)
target_link_libraries(diar_enrollment_probe PRIVATE nemo_speech_diar)

add_executable(diar_representation_probe diar_representation_probe.cpp)
target_link_libraries(diar_representation_probe PRIVATE nemo_speech_diar)

add_executable(diar_native_matching_probe diar_native_matching_probe.cpp)
target_link_libraries(diar_native_matching_probe PRIVATE nemo_speech_diar)

add_executable(diar_vctk_voiceid_probe diar_vctk_voiceid_probe.cpp)
target_include_directories(diar_vctk_voiceid_probe PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/vendor)
target_link_libraries(diar_vctk_voiceid_probe PRIVATE nemo_speech_diar)

add_executable(test_voiceid_mapping_research test_voiceid_mapping_research.cpp)
target_include_directories(test_voiceid_mapping_research PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME voiceid_mapping_research COMMAND test_voiceid_mapping_research)

add_executable(test_aosc_profile_intervention test_aosc_profile_intervention.cpp)
target_include_directories(test_aosc_profile_intervention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_aosc_profile_intervention PRIVATE nemo_speech_diar)
add_test(NAME aosc_profile_intervention COMMAND test_aosc_profile_intervention)
if(UNIX)
    add_executable(diar_snapshot_cli diar_snapshot_cli.cpp)
    target_link_libraries(diar_snapshot_cli PRIVATE nemo_speech_diar)
endif()

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
    if(EXISTS "${CMAKE_SOURCE_DIR}/test_files/diar/jfk.wav" AND
       EXISTS "${CMAKE_SOURCE_DIR}/test_files/diar/ami_en2002d_2132.wav")
        add_test(NAME diar_snapshot_continuity
            COMMAND test_diar_snapshot
                "${CMAKE_SOURCE_DIR}/models/Nemotron-3-Diarization.q8_0.gguf"
                "${CMAKE_SOURCE_DIR}/test_files/diar/jfk.wav"
                "${CMAKE_SOURCE_DIR}/test_files/diar/ami_en2002d_2132.wav")
    endif()
endif()
