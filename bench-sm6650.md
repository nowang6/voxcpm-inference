# SM6650 (volcano) HTP 异构解码基准

设备：SM6650P，Hexagon v73（2 HVX + 1 HMX），Android 16，adb root。
负载：`voxcpm-vae-decode-mock` 159 步全量（12.72s 音频，q4_0 GGUF，mempool 变体）。
协议：`scripts/bench_sm6650.sh`（performance governor + 大核 taskset f0 + 45s 冷却 + 5 轮中位）。
线程拓扑：cpu0-3 A55@1.8G / cpu4-6 A78@2.2G / cpu7 A78@2.3G；taskset f0 = cpu4-7。

## 基线里程碑

| 里程碑 | 配置 | 每步中位 | 分解 (ms) | RTF |
|---|---|---|---|---|
| M0 移植（含 TEMP 诊断） | 无绑核默认 | 305.9 | htp=28.5 cpu=272.0 copy=5.4 | 3.86 |
| 1a 剥离诊断 | 无绑核默认 | 213.6 | htp=57.8 cpu=153.8 copy=2.2 | 2.74 |
| **1c 受控（稳定态）** | governor+绑核 | **~100** | **htp=23 cpu=75 copy=2.2** | **1.30** |
| 1c 受控（热降频后） | governor+绑核 | 216.0 | htp=58 cpu=156 copy=2.3 | 2.25–2.77 |

注：M0 的 htp=28.5 与 1a 的 htp=57.8 差异来自 AP 侧诊断 readback 原本计入 htp 段计时；
剥离后 htp 段显出真实成本，但 CPU 段同时 -118ms（诊断日志的管道背压拖慢全流程）。

## 受控稳定态每段分解（ms/call，32 步均值）

```
C0=0.1  H1=3.1  C2=4.8  H3=6.4  C4=13.0  H5=8.5  C6=27.5  H7=5.5  C8=30.0   copy=2.2
```

- H1/H3/H5/H7（HTP convT 双 GEMM）：合计 ~23ms，T 越大越贵（H5@T=1920 最重）。
- CPU 段合计 ~75ms，**C6+C8=57ms（T=1920/3840 大段）**——snake（sin）、dw conv、
  pw k1 GEMM 均 ∝ C×T，是剩余瓶颈主体。
- 段间搬运 copy=2.2ms（scratch 预分配后近乎免费）。

## 热行为

- 5 轮中 3 轮稳定 RTF 1.30，2 轮（第 3/4 轮）退化至 216ms/步；
  退化与机身温度 >~45°C 相关（thermal zone 47.7°C 时仍可达 1.30，46.5°C 时曾退化——
  降频滞后于温度，具体阈值待系统采样）。
- 结论：**性能对比必须报告温度 + 多轮中位**；连续推理场景需要热管理策略
  （间歇/降频预算），基准协议 45s 冷却仍不足以完全避免热影响。

## 环境读数（GGML_HEXAGON_VERBOSE=1）

- `thread_counts on HTP: 2`（DSP 侧 readback：hw_threads-2）
- `dsp_cache_mode: 5`、`enable graph_optimize: 1`、`enable op_fusion: 1`
- mempool 变体；`ADSP_LIBRARY_PATH='skel;/vendor/dsp/cdsp'`（分隔符 `;`）

## 已知问题

1. **host x86 smoke 崩溃（M0 既有）**：CPU 自检模式下 T9b（F16 GEMM w@HTP 变体）
   在 `ggml_vec_dot_f16` 触发 NaN 断言；真机不崩（仅 T9b/T11 数值 FAIL 属已知）。
   不阻塞设备路径，待查。
2. **T1 Q4_0 MUL_MAT rel_rms=0（本会话新观察，0908 时为 PASS 0.0038）**：疑似
   Q4_0 GEMM 未落 HTP（首要嫌疑：SM6650 soc 表项缺失 → VTCM 硬编码 8MB 偏大 →
   预算检查拒绝 → sched 落 CPU）。当前 decode 主路径用 Q8_0 派生权重不受影响；
   列入 2E 排查。
