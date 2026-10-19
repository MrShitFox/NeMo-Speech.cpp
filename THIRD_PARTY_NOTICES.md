# Third-Party Notices

This project retains the third-party notices for code and data used by the
Nemotron 3 Diarization runtime.

## ggml

The project builds ggml from the pinned `llama.cpp` submodule at `llama.cpp/ggml`.
Only ggml is added to the project build; llama.cpp inference targets are not
built. ggml and the upstream source are Copyright (c) 2023-2026 The ggml
Authors and distributed under the MIT License. See [`llama.cpp/LICENSE`](llama.cpp/LICENSE).
The project patch series contains NVIDIA-authored modifications to ggml; the
patches carry their Apache 2.0 notices and do not replace ggml's MIT terms.

## parakeet.cpp-derived runtime code

Parts of `src/runtime/ggml/` derive from
[`jason-ni/parakeet.cpp`](https://github.com/jason-ni/parakeet.cpp), revision
`1675ee5b612b629645e3bb4b42f671da27a76d5b`.

Copyright (c) 2025 Jason Ni. Licensed under the MIT License.

## Sample audio

`test_files/diar/jfk.wav` is copied from
[`ggml-org/whisper.cpp`](https://github.com/ggml-org/whisper.cpp), revision
`23ee03506a91ac3d3f0071b40e66a430eebdfa1d`.

Copyright (c) 2023-2026 The ggml Authors. Licensed under the MIT License.

The AMI meeting excerpt and speaker transcript in `test_files/diar/` are from
the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), licensed
under CC BY 4.0. The excerpt reference RTTM is adapted from
[`BUTSpeechFIT/AMI-diarization-setup`](https://github.com/BUTSpeechFIT/AMI-diarization-setup),
licensed under Apache 2.0. Fixture-specific details and attribution are in
[`test_files/diar/README.md`](test_files/diar/README.md).

## MIT License

The MIT-licensed components and derived material identified above are subject
to the following terms. Component-specific copyright notices remain listed
above.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
