# Shared ASR/Sortformer building blocks. These targets own the sources that
# genuinely serve both pipelines; neither pipeline compiles them a second time.
add_library(nemo_speech_audio_frontend SHARED features/fe.cpp)
target_include_directories(nemo_speech_audio_frontend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/features
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(nemo_speech_audio_frontend PUBLIC nemo_speech_runtime_ggml)
if(WIN32)
    set_target_properties(nemo_speech_audio_frontend PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Sortformer V2 and ASR both use the NeMo FastConformer implementation. V3
# diarization uses RopeTransformer below, but keeping the V2 path preserves
# existing model support without pulling in the ASR recognizer/decoder stack.
add_library(nemo_speech_encoder SHARED
    encoder/fastconformer.cpp
    encoder/rel_pos_attention.cpp)
target_include_directories(nemo_speech_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/encoder)
target_link_libraries(nemo_speech_encoder PUBLIC nemo_speech_runtime_ggml)
if(WIN32)
    set_target_properties(nemo_speech_encoder PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Shared C ABI error/version storage. It is linked into exactly one public C
# ABI library in each build mode, so ASR and diarization see the same thread-
# local error when they are packaged together.
add_library(nemo_speech_c_api_common STATIC ../common/c_api_common.cpp)
target_include_directories(nemo_speech_c_api_common PRIVATE ../common)
set_target_properties(nemo_speech_c_api_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

if(NEMO_SPEECH_BUILD_DIAR OR NEMO_SPEECH_BUILD_ASR)
    add_library(nemo_speech_diar SHARED
        diar/aosc_state.cpp
        diar/diar_pipeline.cpp
        diar/diarizer.cpp
        diar/sortformer_model.cpp
        encoder/rope_transformer.cpp
        encoder/transformer_encoder.cpp)
    target_include_directories(nemo_speech_diar PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/diar
        ${CMAKE_CURRENT_SOURCE_DIR}/encoder
        ${CMAKE_CURRENT_SOURCE_DIR}/features)
    target_link_libraries(nemo_speech_diar PUBLIC
        nemo_speech_audio
        nemo_speech_audio_frontend
        nemo_speech_common
        nemo_speech_encoder
        nemo_speech_runtime_ggml)
    if(WIN32)
        set_target_properties(nemo_speech_diar PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    install(TARGETS nemo_speech_diar
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(TARGETS nemo_speech_audio_frontend nemo_speech_encoder
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(NEMO_SPEECH_BUILD_ASR)
    # Explicit ASR-only implementation list. Diarization and the genuinely
    # shared frontend/encoder sources are owned by their own targets above.
    add_library(nemo_speech_asr SHARED
        model.cpp
        recognizer.cpp
        runner.cpp
        decoders/context_biasing.cpp
        decoders/flashlight_decoder.cpp
        decoders/greedy_ctc_decoder.cpp
        decoders/rnnt_greedy_decoder.cpp
        decoders/rnnt_modules.cpp
        encoder/cache_aware_encoder.cpp
        pnc/pnc_model.cpp
        pnc/pnc_runner.cpp
        pnc/wordpiece.cpp
        postproc/itn.cpp
        postproc/itn_align.cpp
        postproc/pipeline.cpp
        postproc/profanity.cpp
        vad/silero_vad.cpp
        vad/vad_endpointer.cpp
        vad/vad_masker.cpp)

    if(WIN32)
        set_target_properties(nemo_speech_asr PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    target_include_directories(nemo_speech_asr PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/features
        ${CMAKE_CURRENT_SOURCE_DIR}/encoder
        ${CMAKE_CURRENT_SOURCE_DIR}/decoders
        ${CMAKE_CURRENT_SOURCE_DIR}/vad
        ${CMAKE_CURRENT_SOURCE_DIR}/diar
        ${CMAKE_CURRENT_SOURCE_DIR}/postproc
        ${CMAKE_CURRENT_SOURCE_DIR}/pnc)
    target_link_libraries(nemo_speech_asr PUBLIC
        nemo_speech_audio
        nemo_speech_audio_frontend
        nemo_speech_common
        nemo_speech_diar
        nemo_speech_encoder
        nemo_speech_runtime_ggml)

    if(NEMO_SPEECH_WITH_NORM)
        target_compile_definitions(nemo_speech_asr PUBLIC NEMO_SPEECH_WITH_NORM=1)
        target_link_libraries(nemo_speech_asr PRIVATE nemo_speech_text_normalization)
    endif()

    # SentencePiece is required by ASR context-biasing decoders, but is not
    # used by Sortformer or the standalone diarization C ABI.
    if(UNIX AND NOT APPLE)
        set(_NEMO_SPEECH_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
        find_library(SENTENCEPIECE_STATIC_LIB sentencepiece
            HINTS "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/lib")
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${_NEMO_SPEECH_LIBRARY_SUFFIXES}")
        unset(_NEMO_SPEECH_LIBRARY_SUFFIXES)
    elseif(APPLE)
        find_library(SENTENCEPIECE_STATIC_LIB libsentencepiece.a
            PATHS "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/lib" NO_DEFAULT_PATH)
    endif()
    if(SENTENCEPIECE_STATIC_LIB)
        find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h
            HINTS "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/include")
        if(NOT SENTENCEPIECE_INCLUDE_DIR)
            message(FATAL_ERROR "SentencePiece archive found without sentencepiece_processor.h")
        endif()
        target_include_directories(nemo_speech_asr PRIVATE ${SENTENCEPIECE_INCLUDE_DIR})
        if(APPLE)
            target_link_options(nemo_speech_asr PRIVATE
                "LINKER:-load_hidden,${SENTENCEPIECE_STATIC_LIB}")
        else()
            target_link_libraries(nemo_speech_asr PRIVATE ${SENTENCEPIECE_STATIC_LIB})
            target_link_options(nemo_speech_asr PRIVATE
                "LINKER:--exclude-libs,libsentencepiece.a")
        endif()
    elseif(NEMO_SPEECH_WITH_NORM AND UNIX AND NOT APPLE)
        message(FATAL_ERROR
            "normalization requires private static SentencePiece; "
            "run scripts/build_sentencepiece_static.sh")
    else()
        find_package(unofficial-sentencepiece CONFIG QUIET)
        if(TARGET unofficial::sentencepiece::sentencepiece)
            target_link_libraries(nemo_speech_asr PRIVATE
                unofficial::sentencepiece::sentencepiece)
        else()
            find_library(SENTENCEPIECE_LIB sentencepiece REQUIRED)
            find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h REQUIRED)
            target_link_libraries(nemo_speech_asr PRIVATE ${SENTENCEPIECE_LIB})
            target_include_directories(nemo_speech_asr PRIVATE ${SENTENCEPIECE_INCLUDE_DIR})
            file(STRINGS "${SENTENCEPIECE_INCLUDE_DIR}/sentencepiece_processor.h"
                _NEMO_SPEECH_SENTENCEPIECE_ABSL REGEX "#include [<\"](third_party/)?absl/")
            if(_NEMO_SPEECH_SENTENCEPIECE_ABSL)
                find_package(absl CONFIG QUIET)
                if(absl_FOUND)
                    target_link_libraries(nemo_speech_asr PRIVATE absl::status absl::strings)
                else()
                    message(WARNING
                        "sentencepiece_processor.h includes abseil but no abseil CMake package "
                        "was found; linking nemo_speech_asr may fail")
                endif()
            endif()
            unset(_NEMO_SPEECH_SENTENCEPIECE_ABSL)
        endif()
    endif()

    if(WIN32)
        find_package(absl CONFIG QUIET)
        if(absl_FOUND)
            target_link_libraries(nemo_speech_asr PRIVATE
                absl::flat_hash_map absl::flat_hash_set absl::strings absl::str_format absl::hash)
        endif()
        find_package(protobuf CONFIG QUIET)
        if(TARGET protobuf::libprotobuf)
            target_link_libraries(nemo_speech_asr PRIVATE protobuf::libprotobuf)
        endif()
    endif()

    if(NEMO_SPEECH_WITH_FLASHLIGHT)
        target_compile_definitions(nemo_speech_asr PUBLIC NEMO_SPEECH_WITH_FLASHLIGHT=1)
        target_link_libraries(nemo_speech_asr PRIVATE flashlight-text flashlight-text-kenlm)
    endif()

    install(TARGETS nemo_speech_asr
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# Keep the established ASR ABI DSO and its direct exports when both APIs are
# requested. In diar-only builds the same diar C API source gets its own DSO,
# with no ASR target or SentencePiece dependency.
if(NEMO_SPEECH_BUILD_ASR)
    # ASR's established C ABI also exported the standalone diarization
    # functions. Keep those symbols for existing ASR-only consumers.
    set(_NEMO_SPEECH_C_API_SOURCES c_api.cpp diar/c_api.cpp)
    add_library(nemo_speech_asr_c SHARED ${_NEMO_SPEECH_C_API_SOURCES})
    target_link_libraries(nemo_speech_asr_c PRIVATE nemo_speech_asr nemo_speech_c_api_common)
    target_include_directories(nemo_speech_asr_c PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_include_directories(nemo_speech_asr_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
    target_compile_definitions(nemo_speech_asr_c PRIVATE NEMO_SPEECH_ASR_BUILD)
    if(NEMO_SPEECH_BUILD_DIAR)
        target_compile_definitions(nemo_speech_asr_c PRIVATE NEMO_SPEECH_DIAR_BUILD)
        add_library(nemo_speech_diar_c ALIAS nemo_speech_asr_c)
    endif()
    set_target_properties(nemo_speech_asr_c PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        SOVERSION 1)
    if(UNIX AND NOT APPLE)
        target_link_options(nemo_speech_asr_c PRIVATE
            "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/c_api.map")
    endif()
    install(TARGETS nemo_speech_asr_c
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
elseif(NEMO_SPEECH_BUILD_DIAR)
    add_library(nemo_speech_diar_c SHARED diar/c_api.cpp)
    target_link_libraries(nemo_speech_diar_c PRIVATE nemo_speech_diar nemo_speech_c_api_common)
    target_include_directories(nemo_speech_diar_c PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_include_directories(nemo_speech_diar_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
    target_compile_definitions(nemo_speech_diar_c PRIVATE
        NEMO_SPEECH_DIAR_BUILD NEMO_SPEECH_DIAR_C_API_STANDALONE)
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

    # Before source-level separation the diar functions were also exported
    # from libnemo_speech_asr_c.so. Keep that runtime name usable by existing
    # diar-only ELF consumers while the new target/library name is adopted.
    add_library(nemo_speech_asr_c ALIAS nemo_speech_diar_c)
    if(UNIX AND NOT APPLE)
        add_custom_command(TARGET nemo_speech_diar_c POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -f
                "$<TARGET_FILE_DIR:nemo_speech_diar_c>/libnemo_speech_asr_c.so.1"
                "$<TARGET_FILE_DIR:nemo_speech_diar_c>/libnemo_speech_asr_c.so"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "$<TARGET_FILE_NAME:nemo_speech_diar_c>"
                "$<TARGET_FILE_DIR:nemo_speech_diar_c>/libnemo_speech_asr_c.so.1"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "$<TARGET_FILE_NAME:nemo_speech_diar_c>"
                "$<TARGET_FILE_DIR:nemo_speech_diar_c>/libnemo_speech_asr_c.so"
            VERBATIM)
        install(CODE
            "set(_asr_compat_dir \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
file(REMOVE \"\${_asr_compat_dir}/libnemo_speech_asr_c.so.1\" \"\${_asr_compat_dir}/libnemo_speech_asr_c.so\")
file(CREATE_LINK \"libnemo_speech_diar_c.so.1\" \"\${_asr_compat_dir}/libnemo_speech_asr_c.so.1\" SYMBOLIC)
file(CREATE_LINK \"libnemo_speech_diar_c.so.1\" \"\${_asr_compat_dir}/libnemo_speech_asr_c.so\" SYMBOLIC)")
    endif()
endif()

if(NEMO_SPEECH_BUILD_ASR)
    set(_NEMO_SPEECH_SENTENCEPIECE_LICENSE_DIR
        "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/share/licenses/nemo-speech/third_party/sentencepiece")
    if(EXISTS "${_NEMO_SPEECH_SENTENCEPIECE_LICENSE_DIR}/LICENSE")
        install(DIRECTORY "${_NEMO_SPEECH_SENTENCEPIECE_LICENSE_DIR}/"
            DESTINATION ${NEMO_SPEECH_LICENSE_INSTALL_DIR}/third_party/sentencepiece)
    endif()
endif()
