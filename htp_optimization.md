# VoxCPM AudioVAE CPU + HTP 异构优化说明

> 目标设备：QCS8550（SM8550，Hexagon v73 cDSP/HTP）
> 测试负载：AudioVAE 声码器全量流式解码（159 步 mock 重放，12.72 s 音频）
> 数值基准：`assets/out/cpu_baseline.wav`（原 L1 图、纯 CPU 输出）

## 1. 性能基线与结果

| 指标 | 基线（原 L1 图，纯 CPU） | **HTP 拆段异构（当前默认 HTP 模式）** | CPU 回退 |
|---|---|---|---|
| 总耗时（159 步） | 39.9 s | **10.29 s** | 19.4 s |
| 单步耗时 | 0.251 s/call | **0.065 s/call** | 0.122 s/call |
| RTF（越低越好） | 3.13 | **0.81（实时）** | 1.52 |
| 加速比 | 1× | **3.89×** | 2.06× |
| SNR vs CPU 基线 | — | **38.2 dB**（门槛 35） | 38.7 dB |
| max\|Δ\| vs CPU 基线 | — | 8.93e-03（门槛 ≤1e-2） | 9.74e-03 |

对拍对象：`--compare-wav assets/out/cpu_baseline.wav`（线程数导致的浮点归约顺序差异约为
1e-5 量级，可视为同源基线；参考流 `reference_streaming.wav` 与基线本身差 0.171）。

## 2. 优化方式

### 2.1 L2 激活布局（Stage 2c）
原 decoder 激活为 L1 布局（时间在 ne[0]），每层 conv 后需要
`im2col → mul_mat → permute → cont` 四步。decoder 专用路径统一切到 **L2 布局
（通道在 ne[0]，内存按时间分块）**：

- **pw k1 卷积短路**：k1 的 im2col 恰为激活转置，直接
  `mul_mat(w[Cin,Cout], x[Cin,T])` 得 L2 输出——省掉 im2col 拷贝、尾部
  permute+cont，以及每步的 Q4_0→F32 激活上转。
- **dw k7 深度卷积**：保留 CPU 自定义核（`MAP_CUSTOM3`），索引换为 L2
  （时间步进 nb[1]、通道步进 nb[0]），因果 padding 语义不变（`src_t = t + k − 2p`）。

### 2.2 convT（转置卷积）lowering（Stage 2c，HTP 加速主体，占总算力 ~64%）
`ggml_conv_transpose_1d` 在 HTP 上不可用，改写为等价的 **双 GEMM lowering**：

```
文件折叠权重 W_orig [K*Cout, Cin]，真实行语义：j = cout*K + k（cout 主序、k 最快）
  W_A(ci, i = k*Cout + cout)   = W_orig(j = cout*K + k, ci),  k ∈ [0, s)
  W_B(ci, i = (k-s)*Cout + cout) = W_orig(j = cout*K + k, ci), k ∈ [s, 2s)
F_A = mul_mat(W_A, XT)   → [s·Cout, T]     ← HTP
F_B = mul_mat(W_B, XT)   → [s·Cout, T]     ← HTP
FA3 = reshape_3d(F_A, Cout, s, T);  FB3 = 同
D   = view(FA3, t∈[1,T)) + view(FB3, t∈[0,T-1))   ← 时间错位相加
D2  = reshape_2d(D, Cout, (T-1)·s)
Y   = concat(view(FA3, [Cout,s]), D2, dim=1)       ← 首 s 个 + 主体
Y  += bias[Cout]                                    （尾部 crop 语义内建）
```

要点：
- 真实文件的量化块沿 ne[0]，**行数 = Cin、行宽 = K*Cout**；块 scale 被同块
  32 个 j 共享而各 j 拆往不同目标行 → **无法字节级无损拆分**，统一走
  **反量化 → F32 重排 → F16**（`split_convt_weight_rows`）。
- F16 的 W_A/W_B 精度最优（SNR 38.7 dB）；Q8_0 33.6 dB；Q4_0 重量化 20.3 dB
  （双重量化，不达标）。

### 2.3 snake 激活 13 节点 → 5 节点
原实现：`REPEAT×2 + ARANGE×2 + ADD1 + MUL + SIN + SQR + DIV + MUL + ADD`。
alpha 的 F32 化与 `inv = 1/(alpha+eps)` 在派生权重中**预计算**后缩为
`MUL → SIN → SQR → MUL → ADD`（5 节点）。**SIN 留 CPU**（vendored HTP 后端
无 SIN kernel），其余可落 HTP。

### 2.4 HTP 拆段计划（Stage 2d，绕开跨设备拷贝失效）
`ggml_backend_sched` 的 split 间自动拷贝（CPU→HTP 方向）在 dspqueue 后端上
失效——DSP 读到陈旧/全零数据（T9 split_copy 冒烟实锤，HOSTBUF/OP_OFFLOAD
均无效）。绕过方式：**decoder 拆成 9 张交替的单后端图**，段间激活用
`tensor_get/tensor_set` 显式搬运（走 dspqueue 的 DMA 正确路径）：

```
seg0(CPU): latent → dw0 → pw1 → block2 头链(snake)      → XT2
seg1(HTP): XT2 → convT2 双 GEMM（F_A / F_B）             → (F_A, F_B)
seg2(CPU): F_A/F_B 组装 convT 语义 → block2 res×3 → XT3
seg3(HTP): XT3 → convT3                                  → (F_A, F_B)
seg4(CPU): → block3 res×3 → block4 头链                  → XT4
seg5(HTP): XT4 → convT4                                  → (F_A, F_B)
seg6(CPU): → block4 res×3 → block5 头链                  → XT5
seg7(HTP): XT5 → convT5                                  → (F_A, F_B)
seg8(CPU): → block5 res×3 → final snake → final conv k7 → tanh → audio
```

- 图**一次构建、多步复用**：uid 稳定 → HTP opbatch 只编译一次（避免每步重编译）。
- 段间搬运 ~16 MB/步（DMA 正确路径），开销约 5 ms。
- convT 段在 HTP **只做双 mul_mat**；view/concat/bias 组装放在相邻 CPU 段
  （CONCAT 的直接 graph_compute 在该后端上有异常）。

### 2.5 aarch64 fp16 SIMD 编译开关（纯构建项）
vendored ggml-cpu 自带的手动开关，默认关闭：

```
-DGGML_INTERNAL_FP16_VECTOR_ARITHMETIC=ON -DGGML_INTERNAL_DOTPROD=ON -DGGML_INTERNAL_I8MM=ON
```

生效后编译选项为 `-mcpu=native+dotprod+i8mm+nosve+nosme`，CPU 的 F16 GEMM
获得 ~5× 加速（CPU 回退路径从 206 s 恢复到 19.4 s 的关键）。

### 2.6 其它
- `VOXCPM_BACKEND=htp/cpu`：模式选择；HTP 初始化失败自动回退 CPU。
- 权重分桶：`WeightGroupFn` 谓词（仅 pw 的 Q4_0 上 HTP 的开关保留为
  `VOXCPM_HTP_WEIGHTS`，当前拆段模式下默认关闭）。
- 派生权重双桶：W_A/W_B（GEMM 大头，仅被 MUL_MAT 消费）落 HTP buffer
  （USAGE_WEIGHTS）；bias/alpha/inv 恒留 CPU（与 snake/dw 同侧消费，避免
  密集跨设备拷贝）。

## 3. 当前算子分布

### HTP（seg1/3/5/7，每块一段）
| 算子 | 说明 | 类型 |
|---|---|---|
| MUL_MAT (W_A, XT) | convT 前半核 GEMM，F16 权重 × F32 激活 | fp16 计算 |
| MUL_MAT (W_B, XT) | convT 后半核 GEMM | fp16 计算 |

（convT 的 bias add / 错位视图加 / concat 移至相邻 CPU 段执行。）

### CPU（seg0/2/4/6/8）
| 算子 | 说明 |
|---|---|
| MAP_CUSTOM3 深度卷积 k7 | L2 自定义核（model.0 与各 res unit 的 conv1，F16 权重） |
| MUL_MAT pw k1 | 各 res unit 的 conv2 与 model.1（Q4_0 权重，CPU 侧消费） |
| SIN | snake 的 sin（HTP 后端无 SIN kernel） |
| MUL/SQR/ADD | snake 其余节点（alpha/inv 的 F32 派生权重在 CPU，与 CPU 段同侧） |
| VIEW/ADD/CONCAT | convT 语义组装（F_A/F_B 错位相加 + 首块拼接） |
| IM2COL/PAD/PERMUTE/CONT | final conv k7 的预处理（F16 [672,1] 权重，mul_mat 后 permute+cont） |
| TANH | 输出级（HTP 有 TANH，但在 CPU 段尾随 final conv 执行） |
| GET_ROWS | sr_cond 条件嵌入（当前模型无 sr_cond 权重，恒透传） |

### 权重存放
| 权重 | 原始类型 | 存放 |
|---|---|---|
| convT 折叠权重（4 个，[K·Cout, Cin]） | Q4_0 | 原件留 CPU（供反量化）；**派生 W_A/W_B 为 F16，落 HTP buffer** |
| pw k1 卷积（13 个，[Cin, Cout]） | Q4_0 | CPU buffer（CPU 段消费） |
| dw k7 卷积（14 个，[7,1,C]） | F16 | CPU buffer（自定义核直读） |
| final conv [7,96] | F16 | CPU buffer |
| bias / snake alpha | F16 | 原件留 CPU；**F32 派生 + inv 派生留 CPU** |
| sr_cond embed | — | 当前模型无（透传） |

## 4. 数值验证（htp_smoke，T1–T9）

| 测试 | 内容 | 门槛 | 结果（HTP） |
|---|---|---|---|
| T1 | Q4_0 MUL_MAT ×5 形状 | rel_rms ≤ 1e-2 且 >0 | PASS（0.0038） |
| T2 | 广播 ADD/MUL | ulp 级 | PASS（Δ≈1e-7） |
| T3 | 跨视图偏移 ADD（lowering 的 D） | ulp 级 | PASS |
| T4 | CONCAT dim=1 | 逐位 0 | PASS（Δ=0） |
| T5 | snake 链 MUL→SQR→MUL→ADD | ≤1e-5 | PASS |
| T6 | GET_ROWS（I32 索引） | 逐位 0 | PASS（Δ=0） |
| T7 | TANH | ≤1e-5 | PASS |
| T8 | convT lowering 端到端 vs 原生 op（真实文件行语义闭环） | rel_rms ≤ 1e-2 | PASS（0.00048） |
| T9 | F16 / Q8_0 权重 MUL_MAT | rel_rms ≤ 1e-2 | PASS（0.00034/0.0038） |

## 5. 构建 / 运行

```bash
# 构建（HTP 后端；纯 CPU 调试用 build 目录）
cmake -B build-htp -DGGML_HEXAGON=ON -DGGML_HEXAGON_USE_MEMPOOL=OFF \
      -DHEXAGON_SDK_ROOT=$HOME/opt/hexagon-sdk -DCMAKE_BUILD_TYPE=Release \
      -DGGML_INTERNAL_FP16_VECTOR_ARITHMETIC=ON \
      -DGGML_INTERNAL_DOTPROD=ON -DGGML_INTERNAL_I8MM=ON
cmake --build build-htp -j

# HTP 异构全量解码（10.3 s）
VOXCPM_BACKEND=htp scripts/run_htp.sh voxcpm-vae-decode-mock \
    --model-path models/voxcpm-0.5b-audio-vae-q4_0.gguf \
    --mock-dir assets/mock \
    --compare-wav assets/out/cpu_baseline.wav

# CPU 回退（19.4 s，保底）
VOXCPM_BACKEND=cpu scripts/run_htp.sh voxcpm-vae-decode-mock ... # 同上
```

调试开关：`VOXCPM_PROFILE_OPS=1`（逐算子画像）、`VOXCPM_DUMP_OPS=1`（逐节点
数值）、`--max-steps N`（部分步）、`VOXCPM_LOG_SCHED=1`（splits）、
`VOXCPM_HTP_WEIGHTS/VOXCPM_HTP_DERIVED`（A/B 对拍）、`GGML_HEXAGON_VERBOSE=1`。

## 6. 已知限制与后续方向

1. **CPU 回退路径较慢（19.4 s）**：W_A/W_B 为 F16，CPU F16 GEMM 虽经 fp16
   SIMD 加速仍非最优；后续可考虑 CPU 侧换 Q8_0 版本派生（SNR 33.6 dB）或
   逐块混合精度。
2. **段间搬运 ~16 MB/步**：可进一步精简（如 convT 段直接输出拼接后布局，
   减少一次中间转置视角）。
3. **Stage 3 调参矩阵未系统扫**：`GGML_HEXAGON_OPBATCH / MM_SELECT / NHVX /
   NHMX`、CPU 线程绑核等。
4. **DSP SIN kernel**（vendored HTP 后端）缺失是 snake 无法全上 HTP 的根因；
   如后续允许改 DSP 侧，可消除每块 2 次跨段切分。
5. **sched 跨设备拷贝失效**（CPU→HTP 方向，dspqueue 缓存语义）属 vendored
   交互 bug，本次以拆段+显式搬运绕开；如上游修复可回归 sched 自动切分。
