# VoiceID baseline: Nemotron 3 Diarization

Снимок сделан 2026-10-19 до очистки подсистем. Этот документ фиксирует поведение исходного checkout, а не заявляет о готовности регистрации голосов или выпуска production runtime.

## Исходное состояние

- Исходный commit: `caa3f47c7c75fa2e9e6340272ebad57c1647dd44` (`fix(diar): keep frames for words not yet tagged`). На момент аудита tracked-файлы были чистыми; локальная GGUF-модель была untracked.
- ОС: NixOS 26.05, Linux x86_64; CPU AMD Ryzen 5 1600, 6 ядер / 12 логических потоков.
- Инструменты: GCC/G++ 15.3.0, CMake 4.1.6, Ninja 1.13.2.
- Устройство: NVIDIA GeForce RTX 4060, драйвер 595.71.05, 8188 MiB. GPU не использовался в CPU-проверках.
- Подмодуль `llama.cpp`: `bd4f514db14d87fded667787a7a963bfbaa98e89` (ggml 0.25.1). CPU preset применяет 19 локальных патчей проекта.

## Модель и защита от добавления весов в Git

Использован файл `models/Nemotron-3-Diarization.q8_0.gguf`:

- размер: 107,012,128 байт;
- SHA-256: `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`;
- GGUF version 3, заголовок сообщает 360 tensors и 57 metadata entries;
- проверка через C API успешно создала модель и вернула 8 speaker channels и 0.010000 секунды на выходной кадр — параметры соответствуют V3.

Веса исключены точечным правилом `/models/*.gguf` в `.gitignore`. Проверка `git check-ignore -v models/Nemotron-3-Diarization.q8_0.gguf` подтвердила правило. Модель осталась в рабочем каталоге и не добавлялась в Git.

Загрузка подтверждена не только разбором заголовка: модель прошла streaming и offline forward на локальных WAV. `ldd build/cpu-diar/bin/nemo-speech` не показывает зависимостей от Python или динамической SentencePiece-библиотеки. В текущем объединённом ASR target SentencePiece всё же нужна при сборке; она собиралась проектным скриптом в `.deps` и статически включилась в библиотеку.

## Конфигурация и сборка

Проверены команды:

```bash
git submodule update --init --recursive
scripts/configure.sh cpu-diar
cmake --build --preset cpu-diar
```

Первая попытка configure завершилась сообщением `Could not find SENTENCEPIECE_LIB`: preset выставляет `NEMO_SPEECH_BUILD_ASR=OFF` и убирает transcribe sources из CLI, но существующий `nemo_speech_asr` target всё равно собирает ASR и diarization вместе и требует SentencePiece. После этого выполнен штатный проектный скрипт, без установки системных пакетов:

```bash
scripts/build_sentencepiece_static.sh
scripts/configure.sh cpu-diar
cmake --build --preset cpu-diar
```

Повторная конфигурация прошла за 0.19 с, сборка — за 39.77 с, с кодом возврата 0. Собралось 92 шага. Это Release/C++17-сборка с `NEMO_SPEECH_BUILD_ASR=OFF`, `NEMO_SPEECH_BUILD_DIAR=ON`, `NEMO_SPEECH_BUILD_CLI=ON`, `NEMO_SPEECH_BUILD_TTS=OFF`, `NEMO_SPEECH_BUILD_NMT=OFF`, HTTP/gRPC/examples/tools/tests выключены, `NEMO_SPEECH_GGML_PATCHED=ON`, `GGML_LLAMAFILE=ON`, `GGML_OPENMP=ON`; CUDA, Metal и Vulkan выключены. Сборка выдала несколько non-fatal предупреждений компилятора, ошибок компиляции не было.

В логах сборки `cpu-diar` присутствовали также ASR `model.cpp`, `recognizer.cpp`, `runner.cpp`, decoders, VAD, PnC, postproc и C API. Это подтверждает, что текущий preset не является source-level минимальной diarization-сборкой.

## Проверки inference

В каталоге `test_files` найдены два пригодных WAV:

- `test_files/asr/wav/test/jfk.wav` — известная односпикерная речь JFK; matching RTTM-разметки в репозитории нет.
- `test_files/diar/ami_en2002d_2132.wav` — 60 секунд, 16 kHz mono PCM, AMI meeting. README и `test_files/diar/ami_en2002d_2132.rttm` описывают три основных голоса, короткую речь четвёртого и перекрытия. В эталоне есть пауза без speech между 19.51 и 26.75 с.

Все прогоны выполнены с локальным файлом модели и `--backend cpu`. Времена ниже — wall time на указанной машине, не обещание throughput:

| Проверка | Результат |
|---|---|
| C API загрузка модели | Успешно; 8 channels, 10 мс/кадр; около 0.10 с |
| JFK, streaming CLI | Успешно за 2.44 с; 3 сегмента, все `speaker 1`: 0.091–2.319, 3.061–4.599, 5.171–10.699 с |
| JFK, offline CLI | Успешно за 0.52 с; 3 сегмента, все `speaker 1`: 0.071–2.349, 3.051–4.629, 5.151–10.699 с |
| AMI, streaming CLI | Успешно за 44.11 с; 25 сегментов и 3 выданных слота (1: 7 сегментов, 2: 10, 3: 8). Между 19.579 и 26.611 с сегмента нет — эталонная длинная пауза не была заполнена. Эталон содержит четыре speaker ID, поэтому этот прогон подтверждает исполнение модели, но не полное восстановление всех голосов и не проходит точностную оценку встречи. |
| AMI, two-speaker turn crop | Кроп 48.3–57.5 с содержит два размеченных последовательных голоса: MEE071 и MEE073. Streaming CLI за 1.74 с выдал два слота: первый 0.031–2.669 с, второй 2.701–5.069, 5.411–7.809 и 8.211–9.109 с. Смена говорящего обнаружена. |
| AMI, короткий дополнительный crop | Кроп 40.4–48.5 с с короткими репликами MEE073 выдал только один слот. Короткую/перекрытую речь нельзя считать успешно различённой. |
| Искусственная тишина | 5 секунд mono PCM, streaming CLI за 0.64 с; `segments: []`. |
| Повторяемость JFK | Два одинаковых streaming запуска дали побайтно одинаковый JSON; SHA-256 обоих результатов `41c76d7d6cceb8cf841b12f138412761ad6d27d4104809e756c739441fcd09e6`. |
| Настоящий chunked streaming на AMI | `test_diar_streaming` подавал аудио порциями по 160 мс, завершился с кодом 0 за 43.24 с. Его 25 RTTM-сегментов и speaker IDs совпали с результатом CLI после нормализации формата и нумерации слотов. |

Один из воспроизводимых streaming вызовов:

```bash
build/cpu-diar/bin/nemo-speech diarize test_files/diar/ami_en2002d_2132.wav \
  --model models/Nemotron-3-Diarization.q8_0.gguf \
  --backend cpu --format json \
  --output build/cpu-diar/ami-stream.json --force
```

Для отдельного двухспикерного участка и тишины использовались локальные временные WAV в игнорируемом `build/`:

```bash
ffmpeg -hide_banner -loglevel error -ss 48.3 \
  -i test_files/diar/ami_en2002d_2132.wav -t 9.2 \
  -ac 1 -ar 16000 -c:a pcm_s16le -y build/cpu-diar/ami-2speaker-long-turn.wav

ffmpeg -hide_banner -loglevel error -f lavfi \
  -i anullsrc=r=16000:cl=mono -t 5 \
  -c:a pcm_s16le -y build/cpu-diar/silence-5s.wav
```

Проверены существующие state/frontend тесты. Чтобы не менять preset по умолчанию, тесты конфигурировались отдельно поверх того же build tree:

```bash
cmake --preset cpu-diar -DBUILD_TESTING=ON
cmake --build --preset cpu-diar \
  --target test_diar_streaming test_diar_state test_mel_features
ctest --test-dir build/cpu-diar --output-on-failure \
  -R 'diar_state|mel_features'
```

Оба CTest прошли: `diar_state` (0.09 с) и `mel_features` (0.04 с). `test_sortformer_parity` не запускался: для него нужны NeMo reference dumps, которых среди локальных fixtures нет. DER для AMI не рассчитывался; проверялась работа локальной GGUF и потоковое выполнение, а не точность разметки. Ни один альтернативный checkpoint не загружался.

## Backend и ограничения среды

- **CPU:** собран и проверен на WAV выше.
- **Vulkan:** GPU присутствует, но SDK-компонентов нет: не найдены `vulkaninfo`, `glslc` и Vulkan headers. `vulkan-diar` не конфигурировался и не собирался.
- **CUDA:** драйвер и RTX 4060 доступны, но `nvcc` не найден в PATH; CUDA preset не запускался. Наличие полного CUDAToolkit этим аудитом не подтверждено.
- **Metal:** этот backend предназначен для Apple платформ; текущая ОС — Linux/NixOS.
- Не проверялись MSVC/Windows, другие GPU, длинные real-time сессии и производительность на полном потоке.
- AOSC speaker cache является только in-memory состоянием `DiarStream`. Сохранения/восстановления AOSC, voice enrollment и cross-session идентификации зарегистрированных лиц в текущем публичном API не найдено.

## Regression gates для последующих коммитов

Для сравнения с этим снимком используйте тот же файл модели (SHA-256 выше) и те же WAV:

1. `scripts/configure.sh cpu-diar` и `cmake --build --preset cpu-diar` должны проходить без удаления ASR-исходников ради обхода зависимости.
2. C API должен загружать GGUF как V3, возвращать 8 channels и 10 мс/кадр.
3. CLI должен обрабатывать JFK streaming/offline, AMI streaming, двухспикерный turn crop и тишину; ожидаемые количества и интервалы приведены в таблице.
4. `test_diar_state`, `test_mel_features` и chunked `test_diar_streaming --push-ms 160` должны проходить; chunked и CLI сегментация должна совпадать после нормализации RTTM.
5. Два CPU streaming запуска JFK должны оставаться детерминированными на том же toolchain и backend.
6. Полный AMI вывод нельзя объявлять accuracy-pass только по числу сегментов: для этого нужен отдельный DER regression против локального RTTM и явная политика к перекрытиям.
7. Проверять отсутствие runtime Python и случайного попадания GGUF: `ldd build/cpu-diar/bin/nemo-speech` и `git check-ignore -v models/Nemotron-3-Diarization.q8_0.gguf`.
# Historical baseline. Current repository scope and validation are recorded in
# `VOICEID_TASK03_REPORT.md`; this document describes the earlier baseline.
