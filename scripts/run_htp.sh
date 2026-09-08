#!/usr/bin/env bash
# 以 CPU+HTP 异构模式运行本仓构建产物（voxcpm-htp-smoke / voxcpm-vae-decode-mock）：
#   LD_LIBRARY_PATH   - 解析 host 侧 libggml*.so
#   ADSP_LIBRARY_PATH - FastRPC 搜索 DSP 侧 skel（libggml-htp-v73.so）。
#                       注意 FastRPC 的路径分隔符是 ';' 而非 ':'。
#
# 用法: scripts/run_htp.sh <程序名> [参数...]
#   例: scripts/run_htp.sh voxcpm-htp-smoke
#       scripts/run_htp.sh voxcpm-vae-decode-mock models/... --compare-wav assets/out/cpu_baseline.wav
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD_DIR:-$ROOT/build-htp}

# host 侧 ggml 动态库与 DSP skel（dspqueue 变体输出在 <build>/bin/，mempool 变体在
# <build>/ggml/src/ggml-hexagon/，两处都探测）
GGML_LIB="$BUILD/ggml/src"
SKEL_DIR=""
for d in "$BUILD/bin" "$BUILD/ggml/src/ggml-hexagon"; do
    if ls "$d"/libggml-htp-v*.so >/dev/null 2>&1; then
        SKEL_DIR=$d
        break
    fi
done
if [ -z "$SKEL_DIR" ]; then
    echo "run_htp.sh: 未找到 DSP skel（libggml-htp-v*.so），检查 build-htp 是否为 -DGGML_HEXAGON=ON 构建" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$GGML_LIB:$SKEL_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$SKEL_DIR;/dsp/cdsp${ADSP_LIBRARY_PATH:+;$ADSP_LIBRARY_PATH}"

BIN="$BUILD/examples/$1"
shift
if [ ! -e "$BIN" ]; then
    echo "run_htp.sh: $BIN 不存在" >&2
    exit 1
fi
exec "$BIN" "$@"
