# Nemotron 3's shared NeMo mel frontend.
add_library(nemo_speech_audio_frontend SHARED fe.cpp)
target_include_directories(nemo_speech_audio_frontend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common)
target_link_libraries(nemo_speech_audio_frontend PUBLIC nemo_speech_runtime_ggml)
if(WIN32)
    set_target_properties(nemo_speech_audio_frontend PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
install(TARGETS nemo_speech_audio_frontend
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
