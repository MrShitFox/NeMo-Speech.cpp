# VoiceID cleanup plan: Nemotron 3 Diarization

Аудит привязан к исходному commit `caa3f47c7c75fa2e9e6340272ebad57c1647dd44`. Исходные наблюдения относятся к baseline Task 01. Task 02 выполнил source/target split, сохранив продуктовые исходники; подробности сборки и регрессий находятся в `docs/VOICEID_TASK02_REPORT.md`.

## Текущее состояние после Task 02

- `cpu-diar` собирает `nemo_speech_diar` и `nemo_speech_diar_c`; ASR implementation target не создаётся.
- CMake использует явные списки исходников для ASR и diarization. Decoder, VAD, PnC, postproc, ASR model/recognizer/runner, audio decoder и subtitles не компилируются в standalone diar build.
- SentencePiece находится только в ветке ASR CMake и standalone diar configure/build не ищет и не линкует её.
- Общие audio I/O, mel frontend, FastConformer/V2 encoder, GGML runtime и C ABI error/version storage принадлежат отдельным общим targets без повторной компиляции.
- `diar.h` включает нейтральный `common.h`; `asr.h` по-прежнему включает его для совместимости. При сборке ASR сохраняется исходная библиотека `nemo_speech_asr_c` со всеми прежними ASR и diar API exports. В diar-only ELF build имя `libnemo_speech_asr_c.so` оставлено как symlink к `libnemo_speech_diar_c.so.1` для старых клиентов, использовавших только diar symbols.
- Физическое удаление ASR/TTS/NMT/S2S исходников в Task 02 не выполнялось.

## Что сейчас собирает `cpu-diar`

Исходный preset отключал отдельные ASR-команды CLI, TTS, NMT, HTTP/gRPC, examples, tools и tests. До Task 02 корневой `CMakeLists.txt` всё равно добавлял `src/asr`, а широкие `file(GLOB ...)` собирали ASR и diarization в `nemo_speech_asr` и `nemo_speech_asr_c`. Теперь `src/asr/CMakeLists.txt` задаёт компоненты явными source lists и создаёт ASR implementation только при `NEMO_SPEECH_BUILD_ASR=ON`.

Фактическая цепочка текущего CLI:

```text
nemo-speech
  -> nemo_speech_engine_registry
  -> nemo_speech_diar
       -> nemo_speech_audio / nemo_speech_audio_frontend / nemo_speech_encoder
       -> nemo_speech_common
       -> nemo_speech_runtime_ggml
            -> ggml / ggml-base / ggml-cpu
  -> app/diarize.cpp + app/bench_diarize.cpp
```

После Task 02 `src/asr/c_api.cpp` содержит ASR API, а `src/asr/diar/c_api.cpp` — diar API. Общие статусы, export macro, version и last-error declarations находятся в `include/nemo_speech/common.h`, который включают оба публичных заголовка. В ASR+DIAR конфигурации обе реализации остаются в существующем `nemo_speech_asr_c`, сохраняя его exports.

До Task 02 объединённый `nemo_speech_asr` требовал SentencePiece при ASR выключенном. Теперь SentencePiece обнаруживается и линкуется только внутри `if(NEMO_SPEECH_BUILD_ASR)`. Sortformer V3 inference tokenizer не использует; standalone configure/build проходит без SentencePiece prefix.

## Что действительно общее между ASR и diarization

- `src/runtime/ggml` и ggml backend targets обслуживают обе модели и GGUF загрузку.
- `src/common` разделён на `nemo_speech_audio` (WAV/audio file и resampling) и `nemo_speech_common` (JSON и config helpers). Audio decoder и subtitles не входят в standalone runtime; decoder test target исключён из обычной сборки.
- `src/asr/features/fe.{h,cpp}` — общий NeMo mel frontend для ASR и Sortformer V3.
- `src/asr/batching.h` и engine registry обеспечивают общие batching/dispatch contracts.
- `src/asr/encoder/fastconformer.{h,cpp}` и `rel_pos_attention.{h,cpp}` нужны ASR, а также оставшейся V2 Sortformer ветке; V3 использует `rope_transformer.*` вместо них.
- `nemo_speech_diar` является отдельной библиотекой для Sortformer/AOSC/streaming. `nemo_speech_asr`, если он включён, зависит от diar target для интегрированного speaker sidecar; обратной зависимости нет. Общие frontend/encoder/audio targets не дублируются.

## KEEP

| Файлы и targets | Причина |
|---|---|
| `src/asr/diar/sortformer_model.{h,cpp}` | Читает Sortformer metadata из GGUF, строит граф V3 и выдаёт coarse embeddings/predictions для AOSC и native 10-мс predictions. В текущем файле также остаётся ветка V2. |
| `src/asr/encoder/rope_transformer.{h,cpp}` | RoPE encoder для high-resolution Sortformer V3. |
| `src/asr/features/fe.{h,cpp}` | NeMo-compatible log-mel frontend; имеет CPU и ggml GPU paths. |
| `src/asr/diar/aosc_state.{h,cpp}` | Arrival-Order Speaker Cache: in-memory embeddings, cached predictions, silence state и compression. Это текущая stream identity state machine. |
| `src/asr/diar/diar_pipeline.{h,cpp}` | Offline path, streaming chunk schedule, per-stream state, probability timeline и segmentation. `DiarStream` владеет `AoscState`. |
| `src/asr/diar/diarizer.{h,cpp}` | Общий model lifetime, resampling, offline и streaming C++ интерфейс. |
| `src/asr/batching.h` | Batching config/metrics используются Sortformer model и C API. |
| `include/nemo_speech/diar.h` и `src/asr/diar/c_api.cpp` | Отдельный публичный C ABI target `nemo_speech_diar_c`: model load, число speakers, stream push/finish, offline и результаты. Существующие функции и структуры сохранены. |
| `include/nemo_speech/common.h` и `src/common/c_api_common.cpp` | Общие статусы, ABI export macro, version и last-error storage. `asr.h` по-прежнему включает `common.h` для сохранения обычного include path. |
| `src/runtime/ggml/*`, `src/runtime/ggml/CMakeLists.txt` | `GGUFLoader`, Session, backend manager, tensor container и общие ggml NN blocks. |
| `llama.cpp/ggml` submodule и `cmake/llama_cpp.cmake` | Источник ggml, включая GGUF и CPU/GPU backend targets. Сборка корневой llama.cpp library для этого не нужна, но сам submodule нужен. |
| `patches/series` и патчи для выбранного backend | CPU и CUDA presets применяют патченный ggml; Vulkan и Metal presets сейчас отключают этот слой. Не удалять без повторной backend-проверки. |
| `src/common/audio_file.{h,cpp}`, `audio_resampler.{h,cpp}` | CLI читает WAV; C++ diarizer преобразует допустимые частоты в частоту модели. |
| `src/common/json.{h,cpp}`, `parameter_parser.{h,cpp}` | JSON CLI output и текущая diar config/YAML path. При выделении более узкого API проверить фактических потребителей до удаления. |
| `src/core/engine_registry.{h,cpp}`, `app/main.cpp`, `app/diarize.cpp`, `app/cli_util.{h,cpp}`, `app/commands.h` | Нужны текущей unified CLI и команде `diarize`. Для будущего меньшего бинарника часть registry/CLI можно заменить отдельным thin frontend. |
| `tests/cpp/asr/test_diar_state.cpp`, `test_diar_streaming.cpp`, `test_sortformer_parity.cpp`, `test_mel_features.cpp` | Регрессия AOSC, stream scheduler, графа и frontend. `test_sortformer_parity` требует внешние NeMo reference dumps, но сам тест и контракт полезны. |
| `test_files/diar/*` | Локальный AMI audio, RTTM и transcript для функциональных и будущих accuracy проверок. |

## REMOVE

Ниже указаны компоненты, которые не нужны самостоятельному Nemotron 3 runtime по назначению. Большинство из них уже исключено из compile graph `cpu-diar`, но исходники всё ещё обслуживают ASR и другие presets проекта. Физическое удаление не входит в Task 02; рассматривать его можно в отдельном VoiceID fork после проверки оставшихся CMake/API потребителей.

| Кандидаты | Targets и зависимости | Условие безопасного удаления |
|---|---|---|
| ASR model/pipeline: `src/asr/model.{h,cpp}`, `recognizer.{h,cpp}`, `runner.{h,cpp}` | `nemo_speech_asr`, `nemo_speech_asr_c`; SentencePiece, VAD/decoder/postproc/PnC компоненты | Уже исключены из standalone compile graph. Физически удалять только в отдельном VoiceID fork после отказа от ASR product/API и проверки остальных presets. |
| ASR-only `src/asr/decoders/*`, `src/asr/vad/*`, `src/asr/postproc/*`, `src/asr/pnc/*` | `nemo_speech_asr`; SentencePiece остаётся ASR-only build dependency | Уже исключены из standalone compile graph. Перед физическим удалением проверить ASR и text normalization consumers в основном проекте. |
| ASR-integrated diar sidecar test: `tests/cpp/asr/test_diar_recognizer.cpp` | Проверяет word-level speaker tags через `Recognizer` и требует отдельную ASR GGUF | Удалить или перенести только если standalone VoiceID больше не поддерживает ASR sidecar; оставить при сохранении совместной ASR+diarization функциональности. |
| TTS: `src/tts/*`, `include/nemo_speech/tts.h`, TTS examples/tests | Target `nemo_speech_tts` (алиасы `nemo_speech_tts_magpietts`, `nemo_speech_tts_nanocodec`, `nemo_speech_tts_c`) | После удаления TTS preset/API из форка. Опциональные `open_jtalk` и `cppjieba` submodules и tokenizer данные оставлять только если другие поддерживаемые цели их используют. |
| NMT: `src/nmt/*`, `include/nemo_speech/nmt.h`, `app/translate.cpp`, `app/bench_translate.cpp` | `nemo_speech_nmt`, `nemo_speech_nmt_c`; root llama.cpp library | После удаления NMT presets/API. Сохранить `llama.cpp/ggml` submodule: он нужен diarization runtime независимо от llama.cpp inference library. |
| ASR+NMT translation pipeline: `src/speech/*`, `examples/speech_translate_file.cpp` | `nemo_speech_speech`; требует ASR и NMT | После отключения соответствующего public product path. |
| Full duplex voice chat: `src/s2s/*` | `nemo_speech_s2s`; включает ASR, ggml patch requirements и llama.cpp | После удаления VoiceChat целей; это не Sortformer stream state. |
| HTTP/gRPC server: `server/http/*`, `proto/*`, `src/services/*`, `src/server/*` | `nemo_speech_http_server`, `riva_proto`, `nemo_speech_grpc_asr`, `nemo_speech_grpc_tts`, `nemo_speech_grpc_nmt`, `riva_server`; cpp-httplib и riva-common submodules | Если отдельный REST/Riva транспорт не нужен в этом форке. Текущий `cpu-diar` их не строит. Будущий REST endpoint лучше оставить отдельным необязательным target. |
| V2 Sortformer graph files: `src/asr/encoder/transformer_encoder.{h,cpp}` и V2 ветка `sortformer_model.cpp` | V2 graph path; сейчас V2/V3 разделены runtime условием в одном target | Только после явного отказа от V2 и удаления ветки V2 из source list/graph. |
| `src/asr/encoder/fastconformer.{h,cpp}`, `rel_pos_attention.{h,cpp}` | Нужны ASR FastConformer и V2 Sortformer | Только после удаления обоих потребителей и проверки V3 source list. Сейчас удалять нельзя. |
| Общие `src/common/audio_decoder.{h,cpp}`, `subtitles.{h,cpp}` | Audio decoder существует как ASR test-only `EXCLUDE_FROM_ALL` target; subtitles собираются только при ASR | Уже не компилируются в standalone runtime. Физически удалять только после удаления ASR/transcript consumers и соответствующих tests. |

Python conversion scripts и Python CI helpers не входят в C++ runtime dependency chain; на этом этапе их лучше оставить как development tooling, пока зафиксирован конвертер GGUF и точность модели.

## REFACTOR

| Узел | Что упростить | Ограничение |
|---|---|---|
| `src/asr/CMakeLists.txt` | Выполнено: явные `nemo_speech_diar` и `nemo_speech_asr` lists, shared frontend/encoder targets, ASR создаётся только при включённом ASR. | Source lists должны обновляться вместе с будущими новыми потребителями; полный список и результаты — в Task 02 report. |
| C ABI | Выполнено: diar функции перенесены в `src/asr/diar/c_api.cpp`; общие declarations/storage вынесены в `common.h` и `c_api_common.cpp`. | Сохранить публичные сигнатуры, структуры, статусы и ownership. При изменении API обновлять обе сборочные конфигурации и export map. |
| Common runtime | Выполнено: audio I/O/resampler отделены от JSON/config; subtitles ASR-only, audio decoder test-only и исключён из default build. | Физическое удаление этих файлов требует отдельного аудита ASR/transcript tests. |
| CLI graph | Отделить `app/diarize.cpp` от ASR transcription, `model_store` и универсального `engine_registry`, если VoiceID binary должен работать только с локальным GGUF. | Сначала сохранить текущую команду, форматы text/json/rttm, config options и local path behavior. HF resolution не считать обязательной частью GGUF inference. |
| Sortformer V2/V3 | После отдельного target при желании зафиксировать V3-only contract, чтобы V3 graph не компилировал FastConformer и Post-LN V2 ветки. | Не менять текущую V3 математику, preprocessing, AOSC или вывод при source separation. |
| AOSC | Оставить полностью неизменным в cleanup этапе. В будущем отдельным API/форматом обсуждать сохранение состояния и регистрацию голоса. | В текущем коде cache — in-memory stream state; сериализация и voice enrollment не существуют. |
| Backend wiring | Оставить один ggml source и подключать только нужный backend target из CMake preset. | Отдельно проверять CPU, CUDA, Vulkan и Apple Metal; CPU build не подтверждает корректность GPU kernels. |

## Зависимости backend

| Backend | Текущая CMake связь и необходимые инструменты |
|---|---|
| CPU | `ggml`, `ggml-base`, `ggml-cpu`; C/C++17, CMake, Ninja и threads. Проект включает `GGML_LLAMAFILE=ON`; ggml включает OpenMP (на этом хосте найден OpenMP 4.5). Standalone diarization больше не требует SentencePiece; она остаётся ASR dependency. |
| Vulkan | `GGML_VULKAN=ON` добавляет `ggml-vulkan`; upstream требует `find_package(Vulkan COMPONENTS glslc REQUIRED)` и `find_package(SPIRV-Headers CONFIG REQUIRED)`. Нужны Vulkan SDK/headers/loader, `glslc`, SPIRV-Headers и runtime Vulkan driver/device. На текущем хосте SDK не обнаружен. |
| CUDA | `GGML_CUDA=ON` добавляет `ggml-cuda`; корневой CMake требует `find_package(CUDAToolkit REQUIRED)`, CUDA compiler/toolkit и driver API. Включаются CUDA Graphs. По умолчанию проект использует toolkit cuBLAS; in-tree cuBLAS shim опционален и выключен. На хосте виден RTX 4060 и driver, но `nvcc` отсутствует; build не проверялся. |
| Metal | `GGML_METAL=ON` добавляет `ggml-metal`; upstream находит Foundation, Metal и MetalKit frameworks и компилирует `.metal` kernels через Apple toolchain/SDK. Это Apple-only путь; Linux checkout его проверить не может. |

Для CPU и CUDA preset в `CMakePresets.json` включён `NEMO_SPEECH_GGML_PATCHED`; Vulkan и Metal preset его выключают. Не переносить эти настройки между backend без parity/build проверки.

## Безопасная последовательность коммитов

1. **Выполнено в Task 02 — разделение runtime и C ABI.** Добавлены явные source lists и отдельные diar/ASR targets; общий ABI вынесен без изменения публичных diar функций, структур и статусов. ASR+DIAR сохраняет существующую ASR C ABI library, `cpu-diar` использует самостоятельную diar library. Ни один продуктовый исходник физически не удалён.
2. **Выполнено в Task 02 — убрать лишнюю build dependency.** Standalone diar configure/build не находит и не линкует SentencePiece; ASR сохраняет свою зависимость. Проверены чистый dependency prefix, C API model smoke и baseline inference comparisons.
3. **Удалить необязательные продуктовые targets отдельными коммитами:** сначала NMT и связанные `src/speech`/llama.cpp inference target; отдельно TTS и tokenizer submodules; отдельно S2S и HTTP/gRPC. Каждый коммит обновляет root CMake/presets/examples/tests и проверяет оставшиеся обещанные продуктовые builds.
4. **Удалить ASR implementation отдельным коммитом** после удаления/переноса `nemo_speech_asr_c` функций, общих статусов и headers. Удалить ASR model, decoders, VAD, PnC и postproc только если новая diar library больше не содержит ссылок.
5. **Удалить V2 Sortformer и лишние общие файлы отдельными коммитами.** Сначала убрать V2 ветку из графа и проверить V3 output, затем исключить FastConformer/V2/common audio decoder/subtitles из CMake; физически удалять файлы после успешного чистого build.
6. **Сузить CLI и backend matrix.** Отвязать model registry/store, если продукт требует только локальный GGUF; CPU gate обязателен. CUDA/Vulkan/Metal gates запускать на подходящих SDK/device runner, а не засчитывать по configure flags.

После каждого шага повторять критерии из `docs/VOICEID_BASELINE.md`, проверять `git diff --check`, C ABI symbols и то, что GGUF остаётся ignored. Не менять AOSC state, feature math и Sortformer graph в target-splitting commits.
# Historical plan. Task 03 has been completed; see `VOICEID_TASK03_REPORT.md`
# for the implemented cleanup and remaining follow-up.
