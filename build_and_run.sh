#!/bin/bash
# 一键脚本：重新编译 -> 用 mock 捕获重放 AudioVAE 解码并对拍
set -e  # 任一步失败立即退出

# 固定到脚本所在目录，保证 models/ assets/ 等相对路径与本目录一致
cd "$(dirname "$0")"

# 量化模型路径：默认 q4_0，可用第一个参数覆盖（如 ./build_and_run.sh models/voxcpm-0.5b-audio-vae-q8_0.gguf）
MODEL_PATH="${1:-models/voxcpm-0.5b-audio-vae-q4_0.gguf}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model GGUF does not exist: $MODEL_PATH" >&2
  exit 1
fi

# 1. 编译项目
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target voxcpm-vae-decode-mock -j

# 2. 重放 mock 捕获（默认 q4_0 模型，输出 assets/out/output_streaming.wav，
#    并与 mock/reference_streaming.wav 对拍 max|delta|）
#    捕获数据目前放在姊妹目录 quantization/mock，需显式传入
./build/examples/voxcpm-vae-decode-mock --mock-dir assets/mock --model-path "models/voxcpm-0.5b-audio-vae-q4_0.gguf"

echo "完成，输出文件：./assets/out/output_streaming.wav"
