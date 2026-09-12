# SM6650 (volcano) HTP 异构解码基准

设备：SM6650P，Hexagon v73（2 HVX，无 HMX），Android 16，adb root。
负载：`voxcpm-vae-decode-mock` 159 步全量（12.72s 音频，q4_0 GGUF，mempool 变体）。
协议：`scripts/bench_sm6650.sh`（performance governor + 大核 taskset f0 + 45s 冷却 + 5 轮中位）。
线程拓扑：cpu0-3 A55@1.8G / cpu4-6 A78@2.2G / cpu7 A78@2.3G；taskset f0 = cpu4-7。

## 基线里程碑

| 里程碑 | 配置 | 每步中位 | 分解 (ms) | RTF |
|---|---|---|---|---|
| M0 移植（含 TEMP 诊断） | 无绑核默认 | 305.9 | htp=28.5 cpu=272.0 copy=5.4 | 3.86 |
| 1a 剥离诊断 | 无绑核默认 | 213.6 | htp=57.8 cpu=153.8 copy=2.2 | 2.74 |
| 1c 受控（稳定态） | governor+绑核 | ~100 | htp=23 cpu=75 copy=2.2 | 1.30 |
| 1c 受控（热降频后） | governor+绑核 | 216.0 | htp=58 cpu=156 copy=2.3 | 2.25–2.77 |
| +B1 融合 snake | 受控稳定窗口 | ~91 | htp=23 cpu=66 copy=2.2 | ~1.14 |
| **+B2/B4（最终）** | 受控 5 轮 bench | **68.0** | **htp=22.5 cpu=36.7 copy=2.1** | **0.850 ✅** |

最终 bench（scripts/bench_sm6650.sh htp 5）：RTF = 0.84 / 0.85 / 0.85 / 0.87 / 0.88，
**中位 0.850**，5 轮零热退化（负载降低后 45s 冷却已足够），SNR 恒 33.0dB。

### 最终验证（Phase 3）

- **HTP vs CPU 输出一致性**：SNR 75.6dB、max|Δ|=3.07e-05（同 L2 图两后端，远超
  35dB 门槛）。
- **真机纯 CPU 模式**（B 系列内核共享）：RTF 0.87 —— 与 HTP 0.85 几乎打平；
  A78 dotprod Q8_0 GEMM ≈ SM6650 HTP（2 HVX），HTP 的现实价值是卸载
  CPU（流式生产中 CPU 可跑其它负载），而非净加速。
- **smoke**：T2/T3/T4/T6/T8/T14 全绿；FAIL 项均为已知（T1 Q4_0 未落 HTP、
  T5/T7/T11 weights-buffer 输入、T10 direct、T9-F16 v73）。
- **回退路径**：ADSP_LIBRARY_PATH 失效 → "HTP device unavailable, falling back
  to CPU-only" + 正常运行。
- host x86：33.4dB / RTF 0.34（标量回退路径同数值门）。

注：M0 的 htp=28.5 与 1a 的 htp=57.8 差异来自 AP 侧诊断 readback 原本计入 htp 段计时；
剥离后 htp 段显出真实成本，但 CPU 段同时 -118ms（诊断日志的管道背压拖慢全流程）。

## 受控稳定态每段分解（ms/call，32 步均值）

1c 基线（B 系列优化前）：
```
C0=0.1  H1=3.1  C2=4.8  H3=6.4  C4=13.0  H5=8.5  C6=27.5  H7=5.5  C8=30.0   copy=2.2
```

当前（B1 融合 snake + B2 Neon dw + B4 final conv L2 核）：
```
C0=0.1  H1=3.1  C2=3.5  H3=6.3  C4=7.9   H5=8.4  C6=13.6  H7=5.1  C8=12.3   copy=2.2
```

- B1（snake 融合）：C 段 -9ms；实测 sin 并非 CPU 大头（pw GEMM 才是）。
- B2（dw Neon + F32 派生权重）：C2/C4/C6 合计 -14ms。
- B4（final conv L2 核，消 im2col ~10MB/步）：C8 -18ms（30→12）。
- H 段（HTP convT 双 GEMM）不变 ~23ms；当前剩余大头为 HTP GEMM 23ms + CPU pw k1
  Q4_0 GEMM（估 25-30ms，C2-C8 内）→ 后续 2C2/2A 空间。

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
