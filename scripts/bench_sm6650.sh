#!/usr/bin/env bash
# SM6650 (volcano) 受控基准协议：
#   governor=performance + 大核 taskset + 多轮取中位 + 温度记录
#
# 用法: scripts/bench_sm6650.sh [htp|cpu] [runs]
# 环境变量:
#   THREADS=4          decode-mock 线程数（默认 4）
#   TASKSET_MASK=auto  大核簇自动识别（默认）| none 不绑核 | 十六进制掩码（如 f0）
#   COOLDOWN=45        每轮间冷却秒数（严格对比建议 ≥60）
#   PROFILE_SEGS=1     透传 VOXCPM_PROFILE_SEGS（每段计时）
#   HEXAGON_ENV        附加 HTP 侧环境（如 "GGML_HEXAGON_MM_SELECT=2"）
#
# 前置：adb 已连接且 adb root 可用；/data/local/tmp/voxcpm-mp/ 已部署
#   （bin/lib/skel/model/mock/cpu_baseline.wav）。
set -euo pipefail

MODE=${1:-htp}
RUNS=${2:-5}
DEV_DIR=/data/local/tmp/voxcpm-mp
THREADS=${THREADS:-4}
COOLDOWN=${COOLDOWN:-45}
ADB=${ADB:-adb}

echo "== bench_sm6650: mode=$MODE runs=$RUNS threads=$THREADS =="

# ---- 1) root + governor=performance（恢复原状见文末说明：interactive 更省电）----
$ADB root >/dev/null 2>&1 || true
$ADB shell 'for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $c; done 2>/dev/null' || true

# ---- 2) 大核簇识别（cpuinfo_max_freq ≥ 90% 全局最大者；SM6650: 4×A55@1.8G + 3×A78@2.2G + 1×A78@2.3G → cpu4-7）----
if [ "${TASKSET_MASK:-auto}" = "auto" ]; then
    TASKSET_MASK=$($ADB shell 'FMAX=$(cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq 2>/dev/null | sort -rn | head -1); TH=$((FMAX * 9 / 10)); m=0; for c in /sys/devices/system/cpu/cpu[0-9]*; do n=${c##*cpu}; f=$(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0); if [ "$f" -ge "$TH" ]; then m=$((m | (1<<n))); fi; done; printf %x $m' | tr -d '\r\n')
fi
TS_PREFIX=""
if [ -n "${TASKSET_MASK:-}" ] && [ "$TASKSET_MASK" != "none" ]; then
    TS_PREFIX="taskset $TASKSET_MASK"
    echo "== big-core mask: $TASKSET_MASK =="
else
    echo "== no taskset pinning =="
fi

# ---- 3) 温度采样（top3 zone，毫摄氏度）----
temp_now() {
    $ADB shell 'for t in /sys/class/thermal/thermal_zone*/temp; do cat $t 2>/dev/null; done | sort -rn | head -3 | tr "\n" " "' | tr -d '\r' || true
}

run_once() {
    $ADB shell "cd $DEV_DIR && LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH='skel;/vendor/dsp/cdsp' \
        VOXCPM_BACKEND=$MODE VOXCPM_PROFILE_SEGS=${PROFILE_SEGS:-0} ${HEXAGON_ENV:-} \
        $TS_PREFIX ./bin/voxcpm-vae-decode-mock \
        --model-path model/voxcpm-0.5b-audio-vae-q4_0.gguf --mock-dir mock \
        --compare-wav cpu_baseline.wav --threads $THREADS" 2>&1 |
        grep -E "Decoded|Compare with cpu_baseline|\[l2plan\] (profile|segs)"
}

# ---- 4) warmup（丢弃）+ N 轮全量 ----
echo "== warmup (--max-steps 32, discarded) =="
$ADB shell "cd $DEV_DIR && LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH='skel;/vendor/dsp/cdsp' \
    VOXCPM_BACKEND=$MODE $TS_PREFIX ./bin/voxcpm-vae-decode-mock \
    --model-path model/voxcpm-0.5b-audio-vae-q4_0.gguf --mock-dir mock \
    --max-steps 32 --threads $THREADS" >/dev/null 2>&1 || true

RESULTS_FILE=$(mktemp /tmp/bench_sm6650.XXXXXX)
for r in $(seq 1 "$RUNS"); do
    sleep "$COOLDOWN"
    T0=$(temp_now)
    OUT=$(run_once) || OUT=""
    T1=$(temp_now)
    LINE=$(printf '%s\n' "$OUT" | grep -oE 'in [0-9.]+s \([0-9.]+s/call, RTF [0-9.]+' | grep -oE '[0-9.]+$' | head -1)
    SNR=$(printf '%s\n' "$OUT" | grep 'Compare with cpu_baseline' | grep -oE 'SNR = [0-9.]+' | grep -oE '[0-9.]+$')
    printf 'run %d: RTF=%s SNR=%s temp %s -> %s\n' "$r" "${LINE:-parse-fail}" "${SNR:-?}" "$T0" "$T1"
    [ -n "$LINE" ] && echo "$LINE" >>"$RESULTS_FILE"
    printf '%s\n' "$OUT" | grep -E '\[l2plan\]' | tail -2 | sed 's/^/    /' || true
done

# ---- 5) 汇总（中位/最小/最大）----
if [ -s "$RESULTS_FILE" ]; then
    sort -g "$RESULTS_FILE" | awk -v n="$(wc -l <"$RESULTS_FILE")" '
        { v[NR] = $1 }
        END {
            med = (n % 2) ? v[int(n/2)+1] : (v[n/2] + v[n/2+1]) / 2;
            printf "== median RTF = %.3f  min = %.3f  max = %.3f  (n=%d) ==\n", med, v[1], v[n], n;
            printf "    每步中位 = %.1f ms（RTF 1.0 目标 = 80.0 ms）\n", med * 80;
        }'
else
    echo "== no results parsed =="
fi
rm -f "$RESULTS_FILE"
# 备注：结束后可手动恢复 governor：
#   adb shell 'for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo schedutil > $c; done'
