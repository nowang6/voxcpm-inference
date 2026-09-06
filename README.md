# inference

VoxCPM AudioVAE 纯 CPU 推理引擎（基于 GGML），仅保留 AudioVAE encoder/decoder，
不含 VoxCPM 主模型（MiniCPM / LocEnc / LocDiT / tokenizer 等）。

内置 ggml v0.22.0（`third_party/ggml` vendored 子树，已剪裁非 CPU backend 目录）。
如需启用 `GGML_CUDA` 等其他 backend，需先从 ggml 上游补回对应的
`src/ggml-<backend>/` 子目录。

模型（`--model-path` 切换，均由 `../quantization/` 生成）：

- `models/voxcpm-0.5b-audio-vae-fp16.gguf`：181 × F16（精度基线）
- `models/voxcpm-0.5b-audio-vae-q4_0.gguf`：34 × Q4_0 + 147 × F16（-71.4%）

架构 `voxcpm-audio-vae`。加载端布局约定：量化产物的 3 维卷积核 `[K, Cin, Cout]`
已折叠为 2 维 `[K*Cin, Cout]`（纯 reshape，行内下标 `j = cin*K + k`，k 最快），
转置卷积核折叠为 `[K*Cout, Cin]`（行内 `j = cout*K + k`），折叠规则见
`../quantization/README.md`。

## 1. 构建（CPU）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

构建产物（`./build/examples/`）：

- `voxcpm-vae-decode-mock`：AudioVAE 解码重放工具（latent → 波形）
- `voxcpm-vae-dump-tensor-check`：GGUF 张量加载校验小工具

## 2. 重放 mock 捕获（latent → WAV）

输入为流式 mock 捕获（`../quantization/mock/`，参考姊妹仓 `voxcpm-vae` 的
`tools/decode_mock.py`）：每步捕获 decoder 输入 `z=[1, 64, 6]`，独立解码后
取尾部 `patch_len_samples` 个采样拼接成完整语音，并与
`mock/reference_streaming.wav` 对拍 `max|Δ|`。

```bash
./build/examples/voxcpm-vae-decode-mock \
  --model-path ./models/voxcpm-0.5b-audio-vae-q4_0.gguf \
  --mock-dir ../quantization/mock \
  --output ./assets/out/output_streaming.wav \
  --threads 4
```

`--model-path` 换成 `q4_0.gguf` / `fp16.gguf` 即可对拍其他精度（捕获目录在
姊妹目录 `../quantization/mock`）。

一键脚本：`./build_and_run.sh`（默认用 `models/voxcpm-0.5b-audio-vae-q4_0.gguf`，
可传参切换：`./build_and_run.sh models/voxcpm-0.5b-audio-vae-fp16.gguf`）

## 3. 接口概览

核心路径：`AudioVAE::load_from_store()`（加载 GGUF）→
`AudioVAE::decode()`（搭建解码计算图，latent `ne=[T, 64]`）→
`VoxCPMBackend::compute()` 执行，输出波形（每 latent patch 640 采样 @ 16 kHz）。
