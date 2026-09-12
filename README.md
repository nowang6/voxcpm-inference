# voxcpm-inference

VoxCPM-0.5B 纯 CPU TTS 推理引擎（基于 GGML）：完整 文本→语音 管线
（MiniCPM BaseLM/ResidualLM + LocEnc/LocDiT + CFM/FSQ + AudioVAE + BPE tokenizer），

内置 ggml v0.22.0（`third_party/ggml` vendored 子树，已剪裁非 CPU backend 目录，
含 Hexagon/HTP 后端源码）。

## 目录结构

```
├── CMakeLists.txt          # 顶层构建（voxcpm 库 + 3 个可执行目标）
├── voxcpm_tts.cpp          # TTS CLI 入口
├── vae_decode_mock.cpp     # AudioVAE 解码重放工具（latent → WAV）
├── dump_tensor_check.cpp   # GGUF 张量加载校验小工具
├── include/voxcpm/         # 公共头（模型/基础设施）
├── src/                    # 引擎实现（16 个编译单元）
├── third_party/ggml/       # vendored ggml v0.22.0（CPU + Hexagon）
└── third_party/miniaudio/  # 参考音频解码（声音克隆输入）
```

## 模型

位于 `/data/models/VoxCPM-GGUF/`：

- `voxcpm-0.5b-q8_0.gguf`：Q8_0 × 310 + F32 × 216 + F16 × 32（766 MiB）
- `voxcpm-0.5b-q4_k.gguf`：Q4_K × 237 + Q5_K × 46 + Q8_0 × 14 + F32 × 216 + F16 × 45（477 MiB）

量化类型由 GGUF 逐张量自动检测。加载端布局约定：量化产物的 3 维卷积核
`[K, Cin, Cout]` 已折叠为 2 维 `[K*Cin, Cout]`（纯 reshape，行内下标
`j = cin*K + k`，k 最快），转置卷积核折叠为 `[K*Cout, Cin]`（行内
`j = cout*K + k`），折叠规则见 `../quantization/README.md`。K=7 深度卷积
（`block.1`）在量化产物中保持 F16/F32。

## 1. 构建（CPU）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

构建产物（`./build/examples/`）：

- `voxcpm_tts`：VoxCPM-0.5B 文本转语音 CLI（Q8_0/Q4_K，含克隆与流式）
- `voxcpm-vae-decode-mock`：AudioVAE 解码重放工具（捕获目录 `../quantization/mock/`）
- `voxcpm-vae-dump-tensor-check`：GGUF 张量加载校验小工具

## 2. 语音合成

```bash
# 纯文本合成（--seed 固定噪声，输出可复现）
./build/examples/voxcpm_tts \
  --model-path /data/models/VoxCPM-GGUF/voxcpm-0.5b-q8_0.gguf \
  --text "你好，欢迎使用语音合成。" \
  --output out.wav --seed 42 --threads 8

# 声音克隆（AudioVAE encode 参考音频）
./build/examples/voxcpm_tts \
  --model-path /data/models/VoxCPM-GGUF/voxcpm-0.5b-q8_0.gguf \
  --prompt-audio prompt.wav --prompt-text "参考音频文本" \
  --text "要合成的文本" --output out.wav --threads 8

# 流式输出（解码循环中逐块落盘）
./build/examples/voxcpm_tts ... --stream --stream-dir stream_out/
```

`--model-path` 换成 q4_k 即可。与源仓库 VoxCPM.cpp（ggml v0.9.7）同参数对拍，
Q8_0/Q4_K × {纯文本, 声音克隆} 四组输出均 bit-exact（PCM16 逐样本 max|Δ|=0）。

