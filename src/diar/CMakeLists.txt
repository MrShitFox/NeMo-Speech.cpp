# Nemotron 3 / Sortformer V3 and its stable public C API.
add_library(nemo_speech_diar SHARED
    aosc_state.cpp
    diar_enrollment_profile.cpp
    diar_pipeline.cpp
    diarizer.cpp
    diar_snapshot.cpp
    sortformer_model.cpp
    encoder/rope_transformer.cpp)
target_include_directories(nemo_speech_diar PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/encoder
    ${CMAKE_SOURCE_DIR}/src/audio_frontend)
target_link_libraries(nemo_speech_diar PUBLIC
    nemo_speech_audio
    nemo_speech_audio_frontend
    nemo_speech_common
    nemo_speech_runtime_ggml)
if(WIN32)
    set_target_properties(nemo_speech_diar PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
install(TARGETS nemo_speech_diar
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_library(nemo_speech_c_api_common STATIC ../common/c_api_common.cpp)
target_include_directories(nemo_speech_c_api_common PRIVATE ../common)
set_target_properties(nemo_speech_c_api_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

add_library(nemo_speech_diar_c SHARED c_api.cpp)
target_link_libraries(nemo_speech_diar_c PRIVATE nemo_speech_diar nemo_speech_c_api_common)
target_include_directories(nemo_speech_diar_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(nemo_speech_diar_c PRIVATE ${CMAKE_SOURCE_DIR}/src/common)
target_compile_definitions(nemo_speech_diar_c PRIVATE
    NEMO_SPEECH_DIAR_BUILD
    NEMO_SPEECH_DIAR_C_API_STANDALONE)
set_target_properties(nemo_speech_diar_c PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    SOVERSION 1)
if(UNIX AND NOT APPLE)
    target_link_options(nemo_speech_diar_c PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/c_api.map")
endif()
install(TARGETS nemo_speech_diar_c
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
