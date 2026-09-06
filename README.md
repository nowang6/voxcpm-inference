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

## 3. 构建（Hexagon NPU 后端，可选）

已合入 `../ggml-hexagon`（zhouwg fork）的高通 Hexagon/HTP 后端到 vendored ggml
（backport 到 v0.22.0，`third_party/ggml/src/ggml-hexagon/`，含官方 dspqueue 变体
与 FastRPC/mempool 变体，由 `GGML_HEXAGON_USE_MEMPOOL` 切换）。依赖 Hexagon SDK
（本机 `/home/niwang/hexagon-sdk6.6`，6.6.0.0 + Tools 19.0.07）：

```bash
# 首次需生成 SDK 的 IDL 编译器（build_idl 硬编码该路径）
make -C "$HOME/hexagon-sdk6.6/ipc/fastrpc/qaic" bin/qaic

# 官方 dspqueue 变体
cmake -B build-hexagon -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HEXAGON=ON -DGGML_HEXAGON_USE_MEMPOOL=OFF \
      -DHEXAGON_SDK_ROOT="$HOME/hexagon-sdk6.6"
cmake --build build-hexagon -j

# FastRPC/mempool 变体
cmake -B build-hexagon-mempool -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HEXAGON=ON -DGGML_HEXAGON_USE_MEMPOOL=ON \
      -DHEXAGON_SDK_ROOT="$HOME/hexagon-sdk6.6"
cmake --build build-hexagon-mempool -j
```

产物：AP 侧 `libggml-hexagon.so`（x86_64 宿主上仅编译冒烟；真机需 aarch64 交叉
构建）+ DSP skeleton `libggml-htp-v73/v75/v79/v81.so`（hexagon-clang 交叉编译，
QUALCOMM DSP6 ELF，部署到设备 cDSP）。无设备时运行会因 dlopen 失败而静默回退
CPU，输出与纯 CPU 构建逐位一致。

注意：应用层（`src/backend.cpp`）目前仍固定 CPU backend，NPU 混合执行
（`ggml_backend_sched` CPU/HTP 分片）为后续任务。

## 4. 接口概览

核心路径：`AudioVAE::load_from_store()`（加载 GGUF）→
`AudioVAE::decode()`（搭建解码计算图，latent `ne=[T, 64]`）→
`VoxCPMBackend::compute()` 执行，输出波形（每 latent patch 640 采样 @ 16 kHz）。
