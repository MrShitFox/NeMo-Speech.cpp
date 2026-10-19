# JSON and configuration helpers used by the CLI and diarization runtime.
add_library(nemo_speech_common STATIC json.cpp parameter_parser.cpp)
target_include_directories(nemo_speech_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(nemo_speech_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Audio file I/O and sample-rate conversion shared by the CLI and public API.
add_library(nemo_speech_audio SHARED audio_file.cpp audio_resampler.cpp)
target_include_directories(nemo_speech_audio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(WIN32)
    set_target_properties(nemo_speech_audio PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
install(TARGETS nemo_speech_audio
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
