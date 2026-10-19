# Command-line guide

`nemo-speech` exposes `diarize`, `help`, and `--version` commands. It accepts
local WAV files or directories and requires a local Nemotron 3 Diarization
GGUF. The CLI does not download models.

```bash
nemo-speech diarize test.wav --model models/Nemotron-3-Diarization.q8_0.gguf
```

Run `nemo-speech --help` or `nemo-speech diarize --help` for the options in the
build. Global `--json`, `--quiet`, and `--verbose` flags control result and
diagnostic output.

## Streaming and offline modes

Streaming is the default and preserves the model's AOSC speaker cache and
streaming state. `--offline` selects the full-attention path for shorter audio.
The `--preset` option accepts `v3-streaming` and `v3-offline`; preset geometry
and the `--offline` flag use the same inference implementation and model.

## Output

Choose `--format text`, `--format json`, or `--format rttm`. JSON output may be
written with `--output PATH`; `--output-dir DIR` preserves directory layout for
batch input. RTTM output accepts an explicit `--recording-id NAME`. Existing
files are kept unless `--force` is supplied.

```bash
nemo-speech diarize meeting.wav --model model.gguf --format rttm \
  --recording-id meeting --output meeting.rttm
```

Directories process WAV inputs. Use `--recursive` to descend into nested
directories and `--concurrency N` to process several files while sharing the
loaded model. `--no-batching` disables compatible dynamic GPU batching.

## Segmentation thresholds

The CLI keeps the runtime's existing defaults. Optional controls are
`--onset`, `--offset`, `--pad-onset`, `--pad-offset`, `--min-duration-on`, and
`--min-duration-off`. These tune speaker activity hysteresis, segment padding,
short speech removal, and short silence filling.

A YAML file may set diarization and batching fields. See
[`config/diar.example.yaml`](../config/diar.example.yaml). CLI flags override
environment and YAML values.
