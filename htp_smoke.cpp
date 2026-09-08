// =============================================================================
// htp_smoke.cpp — HTP 算子冒烟：{HTP0 + CPU} sched vs 纯 CPU 数值对拍
//
// 覆盖 Stage 2 decoder 图重写依赖的全部 HTP op 形态：
//   T1  Q4_0 MUL_MAT（K%32==0 的 pw conv / convT 半权重形状）
//   T2  广播 ADD（bias [M,1]）、广播 MUL（alpha [M,1]）
//   T3  跨视图偏移 ADD（convT lowering 的 D = view(FA3)+view(FB3)）
//   T4  CONCAT dim=1（convT lowering 的 Y 拼接）
//   T5  snake 链 MUL→SQR→MUL→ADD（去 sin 后的全部 HTP 节点）
//   T6  GET_ROWS（sr_cond embedding，I32 索引）
//   T7  TANH（final conv 尾部）
//   T8  convT lowering 端到端（W_A/W_B 行重排 Q4_0 → 双 GEMM → reshape/view/
//       concat/bias），对照 ggml_conv_transpose_1d 原生 op 语义（截尾 crop 内建）
//
// 门槛：GEMM 类 rel_rms ≤ 1e-2（HTP fp16 计算、fp32 累加）；逐位类（二元/拼接/
//       gather）max|Δ| == 0。Q4_0 MUL_MAT 的 rel_rms 必须严格大于 0——
//       若为 0 说明实际落回了 CPU，冒烟失去意义。
//
// 运行前需 GGML_HEXAGON=ON 构建且 ADSP_LIBRARY_PATH 指向 skel 目录
// （见 scripts/run_htp.sh）。
// =============================================================================

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "voxcpm/audio-vae.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

std::mt19937 g_rng(42);
float rnd(float lo, float hi) {
  std::uniform_real_distribution<float> d(lo, hi);
  return d(g_rng);
}

// -----------------------------------------------------------------------------
// 上下文 / 图 / 权重 buffer 小工具
// -----------------------------------------------------------------------------

struct CtxGuard {
  ggml_context* ctx = nullptr;
  void reset(size_t n_tensors, size_t n_nodes) {
    if (ctx) {
      ggml_free(ctx);
    }
    const size_t mem = ggml_tensor_overhead() * n_tensors +
                       ggml_graph_overhead_custom(n_nodes, false) + 4096;
    ggml_init_params p{/*mem_size=*/mem, /*mem_buffer=*/nullptr, /*no_alloc=*/true};
    ctx = ggml_init(p);
  }
  ~CtxGuard() {
    if (ctx) {
      ggml_free(ctx);
    }
  }
};

struct Graph {
  CtxGuard ctx;
  ggml_cgraph* g = nullptr;
  void init(size_t n_tensors, size_t n_nodes) {
    ctx.reset(n_tensors, n_nodes);
    g = ggml_new_graph_custom(ctx.ctx, n_nodes, false);
  }
};

// 权重 buffer：HTP buft 分配 + USAGE_WEIGHTS（set_tensor 触发 tiled repack 的前提）
ggml_backend_buffer_t alloc_weights(ggml_context* ctx, ggml_backend_buffer_type_t buft) {
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
  if (buf) {
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  }
  return buf;
}

void set_bytes(ggml_tensor* t, const void* data, size_t size) {
  ggml_backend_tensor_set(t, data, 0, size);
}

void set_f32(ggml_tensor* t, const std::vector<float>& v) {
  set_bytes(t, v.data(), v.size() * sizeof(float));
}

void get_f32(const ggml_tensor* t, std::vector<float>& v) {
  v.resize(ggml_nelements(t));
  ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
}

void fill_rnd(std::vector<float>& v, float lo = -1.0f, float hi = 1.0f) {
  for (auto& x : v) {
    x = rnd(lo, hi);
  }
}

// 在给定 backends 上执行图（最后一个必须是 CPU）。
// v0.22 sched 语义：图张量（含输入 leaf）在首次 sched_graph_compute 时才分配，
// 因此流程为 reserve → 首跑(仅分配) → fill() 灌输入 → 二跑 → read() 读输出。
// 输出张量驻留在 sched 的 compute buffer 里，必须在 read() 回调中取走
// （run_graph 返回后 buffer 随 sched 一起释放）。返回 n_splits，失败返回 -1。
template <typename F, typename Rd>
int run_graph(const std::vector<ggml_backend_t>& backends, ggml_cgraph* graph, F&& fill,
              Rd&& read) {
  std::vector<ggml_backend_t> bes(backends);
  bes.erase(std::unique(bes.begin(), bes.end()), bes.end());  // CPU 自检模式 htp==cpu 去重
  ggml_backend_sched_t sched = ggml_backend_sched_new(
      bes.data(), nullptr, static_cast<int>(bes.size()),
      /*graph_size=*/256, /*parallel=*/false, /*op_offload=*/true);
  if (!sched) {
    return -1;
  }
  if (!ggml_backend_sched_reserve(sched, graph)) {
    std::printf("  [run_graph] reserve failed\n");
    std::fflush(stdout);
    ggml_backend_sched_free(sched);
    return -1;
  }
  const ggml_status st1 = ggml_backend_sched_graph_compute(sched, graph);
  if (st1 != GGML_STATUS_SUCCESS) {
    std::printf("  [run_graph] first compute st=%d\n", static_cast<int>(st1));
    std::fflush(stdout);
    ggml_backend_sched_free(sched);
    return -1;
  }
  fill();
  const ggml_status st = ggml_backend_sched_graph_compute(sched, graph);
  if (st != GGML_STATUS_SUCCESS) {
    std::printf("  [run_graph] second compute st=%d\n", static_cast<int>(st));
    std::fflush(stdout);
  }
  if (st == GGML_STATUS_SUCCESS) {
    read(graph);
  }
  const int splits = ggml_backend_sched_get_n_splits(sched);
  ggml_backend_sched_free(sched);
  return st == GGML_STATUS_SUCCESS ? splits : -1;
}

ggml_tensor* last_node(ggml_cgraph* graph) {
  return ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1);
}

// -----------------------------------------------------------------------------
// Q4_0 生成 / 反量化 / 行重排（块布局：fp16 d + 16 字节 nibble，32 元素/块）
// -----------------------------------------------------------------------------

void gen_q4_0(std::vector<uint8_t>& out, int rows, int row_elems) {
  const int blocks_per_row = row_elems / 32;
  out.resize(static_cast<size_t>(rows) * blocks_per_row * 18);
  for (size_t r = 0; r < static_cast<size_t>(rows); ++r) {
    for (int b = 0; b < blocks_per_row; ++b) {
      uint8_t* blk = out.data() + (r * blocks_per_row + b) * 18;
      const ggml_fp16_t d = ggml_fp32_to_fp16(rnd(0.01f, 1.0f));
      memcpy(blk, &d, 2);
      for (int i = 0; i < 16; ++i) {
        blk[2 + i] = static_cast<uint8_t>(g_rng() & 0xFF);
      }
    }
  }
}

void dequant_q4_0_row(const uint8_t* row, int row_elems, float* dst) {
  const int blocks = row_elems / 32;
  for (int b = 0; b < blocks; ++b) {
    const uint8_t* blk = row + static_cast<size_t>(b) * 18;
    ggml_fp16_t d;
    memcpy(&d, blk, 2);
    const float df = ggml_fp16_to_fp32(d);
    for (int i = 0; i < 16; ++i) {
      const int q0 = (blk[2 + i] & 0x0F) - 8;
      const int q1 = (blk[2 + i] >> 4) - 8;
      dst[32 * b + i] = q0 * df;
      dst[32 * b + 16 + i] = q1 * df;
    }
  }
}

// -----------------------------------------------------------------------------
// 数值对比
// -----------------------------------------------------------------------------

struct Compare {
  double max_abs = 0.0;
  double rms_err = 0.0;
  double rms_ref = 0.0;
  double rel_rms() const { return rms_ref > 0.0 ? rms_err / rms_ref : 0.0; }
};

Compare compare(const std::vector<float>& ref, const std::vector<float>& got) {
  Compare c;
  double se = 0.0, sr = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    c.max_abs = std::max(c.max_abs, std::abs(d));
    se += d * d;
    sr += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  c.rms_err = std::sqrt(se / static_cast<double>(ref.size()));
  c.rms_ref = std::sqrt(sr / static_cast<double>(ref.size()));
  return c;
}

int g_failures = 0;

void report(const char* name, bool pass, const Compare& c, int splits) {
  std::printf("%-42s | max|Δ|=%10.3g rel_rms=%8.2g splits=%d | %s\n", name, c.max_abs,
              c.rel_rms(), splits, pass ? "PASS" : "FAIL");
  std::fflush(stdout);
  if (!pass) {
    ++g_failures;
  }
}

void report_fail(const char* name, const char* why) {
  std::printf("%-42s | %s | FAIL\n", name, why);
  std::fflush(stdout);
  ++g_failures;
}

// -----------------------------------------------------------------------------
// 全局 backend
// -----------------------------------------------------------------------------

struct R {
  ggml_backend_t cpu = nullptr;
  ggml_backend_t htp = nullptr;
  ggml_backend_buffer_type_t htp_buft = nullptr;
  ggml_backend_buffer_type_t cpu_buft = nullptr;
};
R g;
bool g_htp_present = false;  // false = 无 HTP 设备的纯 CPU 自检模式（仅验证图语义）

// =============================================================================
// T1：Q4_0 MUL_MAT（权重钉 HTP buffer + USAGE_WEIGHTS → 设备端反量化 + repack）
// =============================================================================

bool test_mul_mat(int K, int M, int N) {
  char name[64];
  std::snprintf(name, sizeof(name), "T1 MUL_MAT Q4_0 K=%-5d M=%-5d N=%-5d", K, M, N);

  std::vector<uint8_t> wbytes;
  gen_q4_0(wbytes, M, K);
  std::vector<float> x(K * N);
  fill_rnd(x);

  // HTP 侧：w 钉 HTP buffer；x 为 sched 输入
  CtxGuard wctx;
  wctx.reset(2, 0);
  ggml_tensor* w = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_Q4_0, K, M);
  ggml_backend_buffer_t wbuf = alloc_weights(wctx.ctx, g.htp_buft);
  if (!wbuf) {
    report_fail(name, "HTP weights alloc failed");
    return false;
  }
  set_bytes(w, wbytes.data(), wbytes.size());

  Graph G;
  G.init(8, 8);
  // 注意：不要 ggml_set_input(xt)——sched 会为 input 张量做设备副本（HTP0#leaf_N#0），
  // 而 fill 写的是本体，DSP 读到的是未初始化副本 → 输出全零。
  // 输入一律走"首跑分配 + fill 灌本体"或预分配持久 buffer。
  ggml_tensor* xt = ggml_new_tensor_2d(G.ctx.ctx, GGML_TYPE_F32, K, N);
  ggml_build_forward_expand(G.g, ggml_mul_mat(G.ctx.ctx, w, xt));

  // CPU 参考（同一份 Q4_0 字节，CPU 端精确反量化点积）
  Graph Rw;
  Rw.init(4, 4);
  ggml_tensor* wc = ggml_new_tensor_2d(Rw.ctx.ctx, GGML_TYPE_Q4_0, K, M);
  ggml_tensor* xr = ggml_new_tensor_2d(Rw.ctx.ctx, GGML_TYPE_F32, K, N);
  ggml_build_forward_expand(Rw.g, ggml_mul_mat(Rw.ctx.ctx, wc, xr));
  ggml_backend_alloc_ctx_tensors_from_buft(Rw.ctx.ctx, g.cpu_buft);
  set_bytes(wc, wbytes.data(), wbytes.size());

  std::vector<float> got, ref;
  std::vector<float> xt_rb;  // 输入读回校验（compute buffer 存活期内读）
  const int splits = run_graph({g.htp, g.cpu}, G.g, [&] { set_f32(xt, x); },
                               [&](ggml_cgraph*) {
                                 get_f32(last_node(G.g), got);
                                 xt_rb.resize(x.size());
                                 ggml_backend_tensor_get(xt, xt_rb.data(), 0,
                                                         xt_rb.size() * sizeof(float));
                               });
  const int rsplits =
      run_graph({g.cpu}, Rw.g, [&] { set_f32(xr, x); },
                [&](ggml_cgraph*) { get_f32(last_node(Rw.g), ref); });
  if (splits < 0 || rsplits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  const Compare c = compare(ref, got);
  if (!g_htp_present || c.rel_rms() > 1e-2) {
    // 读回 HTP 权重（独立 buffer，sched 释放后仍存活），区分"set 丢数据" vs "计算错"
    std::vector<uint8_t> rb(wbytes.size());
    ggml_backend_tensor_get(w, rb.data(), 0, wbytes.size());
    size_t diff = 0;
    for (size_t i = 0; i < wbytes.size(); ++i) {
      if (rb[i] != wbytes[i]) ++diff;
    }
    size_t xdiff = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      if (xt_rb[i] != x[i]) ++xdiff;
    }
    std::printf("  [dbg1] 权重diff=%zu/%zu 输入diff=%zu/%zu got前3: %g %g %g (ref: %g %g %g)\n",
                diff, wbytes.size(), xdiff, x.size(), got[0], got[1], got[2], ref[0], ref[1],
                ref[2]);
  }
  // rel_rms 须落在 (0, 1e-2]：>0 证明确实走了 HTP fp16（CPU 回退会逐位相等）
  const bool pass = c.rel_rms() <= 1e-2 && (!g_htp_present || c.rel_rms() > 0.0);
  report(name, pass, c, splits);
  return pass;
}

// =============================================================================
// T2：广播 ADD / MUL（src0 为 HTP 权重 → 节点归属 HTP；src1 [M,1] can_repeat）
// =============================================================================

bool test_broadcast(const char* tag, bool use_add, int M, int N) {
  char name[64];
  std::snprintf(name, sizeof(name), "T2 %s bcast[%d,1]→[%d,%d]", tag, M, M, N);

  std::vector<float> y0(M * N);
  fill_rnd(y0);
  std::vector<float> b(M);
  fill_rnd(b, -0.5f, 0.5f);

  CtxGuard wctx;
  wctx.reset(2, 0);
  ggml_tensor* y0t = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_F32, M, N);
  if (!alloc_weights(wctx.ctx, g.htp_buft)) {
    report_fail(name, "HTP weights alloc failed");
    return false;
  }
  set_f32(y0t, y0);

  Graph G;
  G.init(8, 8);
  ggml_tensor* bt = ggml_new_tensor_2d(G.ctx.ctx, GGML_TYPE_F32, M, 1);
  ggml_tensor* out = use_add ? ggml_add(G.ctx.ctx, y0t, bt) : ggml_mul(G.ctx.ctx, y0t, bt);
  ggml_build_forward_expand(G.g, out);

  // CPU 参考：同构小图
  Graph Rw;
  Rw.init(4, 4);
  ggml_tensor* y0r = ggml_new_tensor_2d(Rw.ctx.ctx, GGML_TYPE_F32, M, N);
  ggml_tensor* br = ggml_new_tensor_2d(Rw.ctx.ctx, GGML_TYPE_F32, M, 1);
  ggml_tensor* outr = use_add ? ggml_add(Rw.ctx.ctx, y0r, br) : ggml_mul(Rw.ctx.ctx, y0r, br);
  ggml_build_forward_expand(Rw.g, outr);

  std::vector<float> got, ref;
  const int splits = run_graph({g.htp, g.cpu}, G.g, [&] { set_f32(bt, b); },
                               [&](ggml_cgraph*) { get_f32(last_node(G.g), got); });
  const int rsplits =
      run_graph({g.cpu}, Rw.g, [&] { set_f32(y0r, y0); set_f32(br, b); },
                [&](ggml_cgraph*) { get_f32(last_node(Rw.g), ref); });
  if (splits < 0 || rsplits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  const Compare c = compare(ref, got);
  // F32 二元语义在 HTP 上与 CPU 仅差向量化舍入（实测 ~1e-7，1 ulp 级）；
  // ulp 级门槛仍能甄别错位/未执行（那会是 rel_rms≈1）
  const bool pass = c.rel_rms() <= 1e-6 && c.max_abs <= 1e-5;
  report(name, pass, c, splits);
  return pass;
}

// =============================================================================
// T3：跨视图偏移 ADD（convT lowering：D = view(FA3, off=sCout) + view(FB3, 0)）
// =============================================================================

bool test_view_add(int Cout, int s, int T) {
  char name[64];
  std::snprintf(name, sizeof(name), "T3 viewADD [%d,%d,%d] off=%d", Cout, s, T - 1,
                s * Cout);

  const size_t n = static_cast<size_t>(Cout) * s * T;
  std::vector<float> fa(n), fb(n);
  fill_rnd(fa);
  fill_rnd(fb);

  CtxGuard wctx;
  wctx.reset(4, 0);
  ggml_tensor* fat = ggml_new_tensor_3d(wctx.ctx, GGML_TYPE_F32, Cout, s, T);
  ggml_tensor* fbt = ggml_new_tensor_3d(wctx.ctx, GGML_TYPE_F32, Cout, s, T);
  if (!alloc_weights(wctx.ctx, g.htp_buft)) {
    report_fail(name, "HTP weights alloc failed");
    return false;
  }
  set_f32(fat, fa);
  set_f32(fbt, fb);

  Graph G;
  G.init(8, 8);
  const size_t nb1 = sizeof(float) * Cout;
  const size_t nb2 = nb1 * s;
  // FA3 去掉 t=0 块（offset = s*Cout 元素），FB3 全量 → 时间索引对齐
  ggml_tensor* va =
      ggml_view_3d(G.ctx.ctx, fat, Cout, s, T - 1, nb1, nb2, sizeof(float) * Cout * s);
  ggml_tensor* vb = ggml_view_3d(G.ctx.ctx, fbt, Cout, s, T - 1, nb1, nb2, 0);
  ggml_build_forward_expand(G.g, ggml_add(G.ctx.ctx, va, vb));

  // host 循环参考：ref[t,r,c] = fa[t+1,r,c] + fb[t,r,c]
  std::vector<float> ref(static_cast<size_t>(Cout) * s * (T - 1));
  for (int t = 0; t < T - 1; ++t) {
    for (int r = 0; r < s; ++r) {
      for (int c = 0; c < Cout; ++c) {
        const size_t ia = (static_cast<size_t>(t + 1) * s + r) * Cout + c;
        const size_t ib = (static_cast<size_t>(t) * s + r) * Cout + c;
        ref[(static_cast<size_t>(t) * s + r) * Cout + c] = fa[ia] + fb[ib];
      }
    }
  }

  std::vector<float> got;
  const int splits = run_graph({g.htp, g.cpu}, G.g, [] {},
                               [&](ggml_cgraph*) { get_f32(last_node(G.g), got); });
  if (splits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  const Compare c = compare(ref, got);
  // 同 T2：HTP 向量化加法与 host 循环存在 1 ulp 级舍入差（实测 ~1e-7）
  const bool pass = c.rel_rms() <= 1e-6 && c.max_abs <= 1e-5;
  report(name, pass, c, splits);
  return pass;
}

// =============================================================================
// T4：CONCAT dim=1（convT lowering：head [Cout,s] + D2 [Cout,(T-1)s]）
// =============================================================================

bool test_concat(int Cout, int s, int T) {
  char name[64];
  std::snprintf(name, sizeof(name), "T4 CONCAT [%d,%d]+[%d,%d] d=1", Cout, s, Cout,
                (T - 1) * s);

  const size_t n = static_cast<size_t>(Cout) * s * T;
  const size_t nd = static_cast<size_t>(Cout) * s * (T - 1);
  std::vector<float> fa(n), fd(nd);
  fill_rnd(fa);
  fill_rnd(fd);

  CtxGuard wctx;
  wctx.reset(4, 0);
  ggml_tensor* fat = ggml_new_tensor_3d(wctx.ctx, GGML_TYPE_F32, Cout, s, T);
  ggml_tensor* dt = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_F32, Cout, (T - 1) * s);
  if (!alloc_weights(wctx.ctx, g.htp_buft)) {
    report_fail(name, "HTP weights alloc failed");
    return false;
  }
  set_f32(fat, fa);
  set_f32(dt, fd);

  Graph G;
  G.init(8, 8);
  const size_t nb1 = sizeof(float) * Cout;
  ggml_tensor* head = ggml_view_2d(G.ctx.ctx, fat, Cout, s, nb1, 0);
  ggml_build_forward_expand(G.g, ggml_concat(G.ctx.ctx, head, dt, 1));

  // host 参考：Y[c,q] = q<s ? fa[q*Cout+c] : fd[(q-s)*Cout+c]（ne[0]=通道最快维）
  std::vector<float> ref(static_cast<size_t>(Cout) * s * T);
  for (int c = 0; c < Cout; ++c) {
    for (int q = 0; q < T * s; ++q) {
      ref[static_cast<size_t>(q) * Cout + c] =
          q < s ? fa[static_cast<size_t>(q) * Cout + c]
                : fd[static_cast<size_t>(q - s) * Cout + c];
    }
  }

  std::vector<float> got;
  const int splits = run_graph({g.htp, g.cpu}, G.g, [] {},
                               [&](ggml_cgraph*) { get_f32(last_node(G.g), got); });
  if (splits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  const Compare c = compare(ref, got);
  const bool pass = c.max_abs == 0.0;
  report(name, pass, c, splits);
  return pass;
}

// =============================================================================
// T5：snake 链 MUL→SQR→MUL→ADD（sin 留 CPU，其余全部 HTP 形态）
// =============================================================================

bool test_snake_chain(int C, int T) {
  char name[64];
  std::snprintf(name, sizeof(name), "T5 snake MUL-SQR-MUL-ADD [%d,%d]", C, T);

  std::vector<float> x(C * T), a(C), iv(C);
  fill_rnd(x, -2.0f, 2.0f);
  fill_rnd(a, 0.5f, 2.0f);
  for (int i = 0; i < C; ++i) {
    iv[i] = 1.0f / (a[i] + 1e-9f);  // 与 Stage 2c 折叠一致
  }

  CtxGuard wctx;
  wctx.reset(4, 0);
  ggml_tensor* xt = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_F32, C, T);
  ggml_tensor* at = ggml_new_tensor_3d(wctx.ctx, GGML_TYPE_F32, C, 1, 1);
  ggml_tensor* it = ggml_new_tensor_3d(wctx.ctx, GGML_TYPE_F32, C, 1, 1);
  if (!alloc_weights(wctx.ctx, g.htp_buft)) {
    report_fail(name, "HTP weights alloc failed");
    return false;
  }
  set_f32(xt, x);
  set_f32(at, a);
  set_f32(it, iv);

  Graph G;
  G.init(8, 8);
  ggml_tensor* ax = ggml_mul(G.ctx.ctx, xt, at);
  ggml_tensor* s2 = ggml_sqr(G.ctx.ctx, ax);
  ggml_tensor* t = ggml_mul(G.ctx.ctx, s2, it);
  ggml_build_forward_expand(G.g, ggml_add(G.ctx.ctx, xt, t));

  std::vector<float> ref(static_cast<size_t>(C) * T);
  for (int i = 0; i < C * T; ++i) {
    const float av = x[i] * a[i % C];
    ref[i] = x[i] + av * av * iv[i % C];
  }

  std::vector<float> got;
  const int splits = run_graph({g.htp, g.cpu}, G.g, [] {},
                               [&](ggml_cgraph*) { get_f32(last_node(G.g), got); });
  if (splits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  const Compare c = compare(ref, got);
  // 逐位不可达：host 侧 -O3 默认 -ffp-contract=fast 会把 x + av*av*iv 融合成 FMA，
  // 与图中分离的 mul/add 相差 1 ulp 级；1e-5 仍足以甄别图语义错误
  const bool pass = c.rel_rms() <= 1e-5 && c.max_abs <= 1e-4;
  report(name, pass, c, splits);
  return pass;
}

// =============================================================================
// T14：融合 snake 自定义算子（B1）：y = x + inv·sin²(alpha·x)
//      多项式 sin vs std::sin 参考；纯 CPU 执行（MAP_CUSTOM1 不上 HTP）；
//      覆盖常态 |αx|<π 范围与 |αx|>1e4 的 libm 回退分支
// =============================================================================

bool test_snake_fused(int C, int T) {
  char name[64];
  std::snprintf(name, sizeof(name), "T14 snake-fused [%d,%d]", C, T);

  std::vector<float> x(static_cast<size_t>(C) * T), a(C), iv(C);
  fill_rnd(x, -8.0f, 8.0f);
  fill_rnd(a, 0.5f, 2.0f);
  for (int i = 0; i < C; ++i) {
    iv[i] = 1.0f / (a[i] + 1e-9f);
  }
  x[0] = 2e4f;                        // |αx| 最高 4e4：触发 libm 回退分支
  x[1] = -2e4f;
  if (static_cast<size_t>(C) * T > 16) {
    x[16] = 8e3f;                     // 边界附近（α=2 时 1.6e4 恰好回退）
  }

  voxcpm::AudioVAESnakeOpData op{a.data(), iv.data()};

  CtxGuard wctx;
  wctx.reset(4, 0);
  ggml_tensor* xt = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_F32, C, T);
  if (!alloc_weights(wctx.ctx, g.cpu_buft)) {
    report_fail(name, "CPU weights alloc failed");
    return false;
  }
  set_f32(xt, x);

  Graph G;
  G.init(8, 8);
  ggml_build_forward_expand(
      G.g, ggml_map_custom1(G.ctx.ctx, xt, &voxcpm::snake_fused_l2_custom,
                            GGML_N_TASKS_MAX, &op));

  std::vector<float> ref(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    const size_t c = i % static_cast<size_t>(C);
    const float s = std::sin(a[c] * x[i]);
    ref[i] = x[i] + s * s * iv[c];
  }

  std::vector<float> got;
  const int splits = run_graph({g.cpu}, G.g, [] {},
                               [&](ggml_cgraph*) { get_f32(last_node(G.g), got); });
  if (splits < 0) {
    report_fail(name, "compute failed");
    return false;
  }
  const Compare c = compare(ref, got);
  const bool pass = c.max_abs <= 1e-5 && c.rel_rms() <= 1e-6;  // 多项式截断 <1e-11
  report(name, pass, c, splits);
  return pass;
}

// =============================================================================
// T6：GET_ROWS（sr_cond embedding：src F32 权重 + I32 索引）
// =============================================================================

bool test_get_rows(int C, int L, int N) {
  char name[64];
  std::snprintf(name, sizeof(name), "T6 GET_ROWS [%d,%d] idx[%d]", C, L, N);

  std::vector<float> src(C * L);
  fill_rnd(src);
  std::vector<int32_t> idx(N);
  for (auto& i : idx) {
    i = static_cast<int32_t>(g_rng() % L);
  }

  CtxGuard wctx;
  wctx.reset(2, 0);
  ggml_tensor* srct = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_F32, C, L);
  if (!alloc_weights(wctx.ctx, g.htp_buft)) {
    report_fail(name, "HTP weights alloc failed");
    return false;
  }
  set_f32(srct, src);

  Graph G;
  G.init(8, 8);
  ggml_tensor* idxt = ggml_new_tensor_1d(G.ctx.ctx, GGML_TYPE_I32, N);
  ggml_build_forward_expand(G.g, ggml_get_rows(G.ctx.ctx, srct, idxt));
  // 索引类输入必须在首跑（分配跑）之前预分配并灌好——垃圾索引会直接越界崩溃；
  // 且与消费 op 同侧（HTP buft），避免依赖 sched 的跨设备输入复制
  ggml_backend_alloc_ctx_tensors_from_buft(G.ctx.ctx, g.htp_buft);
  set_bytes(idxt, idx.data(), idx.size() * sizeof(int32_t));

  std::vector<float> got;
  const int splits = run_graph({g.htp, g.cpu}, G.g, [] {},
                               [&](ggml_cgraph*) { get_f32(last_node(G.g), got); });
  if (splits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  std::vector<float> ref(static_cast<size_t>(C) * N);
  for (int i = 0; i < N; ++i) {
    memcpy(ref.data() + static_cast<size_t>(i) * C,
           src.data() + static_cast<size_t>(idx[i]) * C, C * sizeof(float));
  }

  const Compare c = compare(ref, got);
  const bool pass = c.max_abs == 0.0;
  report(name, pass, c, splits);
  return pass;
}

// =============================================================================
// T7：TANH
// =============================================================================

bool test_tanh(int C, int T) {
  char name[64];
  std::snprintf(name, sizeof(name), "T7 TANH [%d,%d]", C, T);

  std::vector<float> x(C * T);
  fill_rnd(x, -4.0f, 4.0f);

  CtxGuard wctx;
  wctx.reset(2, 0);
  ggml_tensor* xt = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_F32, C, T);
  if (!alloc_weights(wctx.ctx, g.htp_buft)) {
    report_fail(name, "HTP weights alloc failed");
    return false;
  }
  set_f32(xt, x);

  Graph G;
  G.init(4, 4);
  ggml_build_forward_expand(G.g, ggml_tanh(G.ctx.ctx, xt));

  std::vector<float> got;
  const int splits = run_graph({g.htp, g.cpu}, G.g, [] {},
                               [&](ggml_cgraph*) { get_f32(last_node(G.g), got); });
  if (splits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  std::vector<float> ref(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    ref[i] = std::tanh(x[i]);
  }

  const Compare c = compare(ref, got);
  const bool pass = c.rel_rms() <= 1e-5;
  report(name, pass, c, splits);
  return pass;
}

// =============================================================================
// T11：SIN（snake 激活核心；QHL qhmath_hvx_sin_af 实现）+ snake 全链
// =============================================================================

bool test_sin(int C, int T) {
  char name[72];
  std::snprintf(name, sizeof(name), "T11 SIN [%d,%d]", C, T);

  std::vector<float> x(C * T);
  // snake 实际工作区（|αx| 小）与大值边界混合
  for (auto& v : x) {
    v = rnd(-64.0f, 64.0f);
  }
  for (size_t i = 0; i < x.size(); i += 17) {
    x[i] = rnd(-1.0f, 1.0f);
  }

  // 输入由 sched 分配（T9 同款）：全 weights-buffer 图会触发 DSP 端
  // EUNSUPPORTED（T5/T7/T11 共有的既有问题，与 SIN kernel 无关）
  Graph G;
  G.init(4, 4);
  ggml_tensor* xt = ggml_new_tensor_2d(G.ctx.ctx, GGML_TYPE_F32, C, T);
  ggml_build_forward_expand(G.g, ggml_sin(G.ctx.ctx, xt));

  std::vector<float> got;
  const int splits = run_graph({g.htp, g.cpu}, G.g, [] {},
                               [&](ggml_cgraph*) { get_f32(last_node(G.g), got); });
  if (splits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  std::vector<float> ref(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    ref[i] = std::sin(x[i]);
  }

  const Compare c = compare(ref, got);
  // QHL fp32 多项式：绝对误差为主（相对误差在过零点爆炸，用绝对门槛）
  const bool pass = c.max_abs <= 1e-5;
  report(name, pass, c, splits);
  return pass;
}

// snake 全链含 SIN：y = x + inv·sin(αx)²（α/inv [C,1] 广播）
bool test_snake_full(int C, int T) {
  char name[72];
  std::snprintf(name, sizeof(name), "T11b snake MUL-SIN-SQR-MUL-ADD [%d,%d]", C, T);

  std::vector<float> x(C * T), a(C), iv(C);
  fill_rnd(x, -4.0f, 4.0f);
  fill_rnd(a, 0.5f, 2.0f);
  for (int i = 0; i < C; ++i) {
    iv[i] = 1.0f / (a[i] + 1e-9f);
  }

  CtxGuard wctx;
  wctx.reset(4, 0);
  ggml_tensor* xt = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_F32, C, T);
  ggml_tensor* at = ggml_new_tensor_3d(wctx.ctx, GGML_TYPE_F32, C, 1, 1);
  ggml_tensor* it = ggml_new_tensor_3d(wctx.ctx, GGML_TYPE_F32, C, 1, 1);
  if (!alloc_weights(wctx.ctx, g.htp_buft)) {
    report_fail(name, "HTP weights alloc failed");
    return false;
  }
  set_f32(xt, x);
  set_f32(at, a);
  set_f32(it, iv);

  Graph G;
  G.init(8, 8);
  ggml_tensor* ax = ggml_mul(G.ctx.ctx, xt, at);
  ggml_tensor* s = ggml_sin(G.ctx.ctx, ax);
  ggml_tensor* s2 = ggml_sqr(G.ctx.ctx, s);
  ggml_tensor* t = ggml_mul(G.ctx.ctx, s2, it);
  ggml_build_forward_expand(G.g, ggml_add(G.ctx.ctx, xt, t));

  std::vector<float> got;
  const int splits = run_graph({g.htp, g.cpu}, G.g, [] {},
                               [&](ggml_cgraph*) { get_f32(last_node(G.g), got); });
  if (splits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  std::vector<float> ref(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    const float av = x[i] * a[i % C];
    const float sv = std::sin(av);
    ref[i] = x[i] + sv * sv * iv[i % C];
  }

  const Compare c = compare(ref, got);
  const bool pass = c.max_abs <= 1e-4;
  report(name, pass, c, splits);
  return pass;
}

// T9：F16 / Q8_0 权重 MUL_MAT（W_A/W_B 派生权重的候选 HTP 形态）
// =============================================================================

bool test_mul_mat_wt(int K, int M, int N, bool f16, bool split_copy = false) {
  char name[72];
  std::snprintf(name, sizeof(name), "T9 MUL_MAT %s K=%-5d M=%-5d N=%-5d",
                f16 ? "F16" : "Q8_0", K, M, N);

  std::vector<float> wf(static_cast<size_t>(M) * K);
  for (auto &v : wf) v = rnd(-1.0f, 1.0f);
  const size_t w_row = f16 ? K * 2 : (K / 32) * 34;
  std::vector<uint8_t> wq(w_row * M);
  if (f16) {
    auto *h = reinterpret_cast<ggml_fp16_t *>(wq.data());
    for (size_t i = 0; i < wf.size(); ++i) h[i] = ggml_fp32_to_fp16(wf[i]);
  } else {
    for (size_t m = 0; m < (size_t)M; ++m) {
      const float *row = wf.data() + m * K;
      for (size_t b = 0; b < (size_t)(K / 32); ++b) {
        const float *blk = row + b * 32;
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::abs(blk[i]));
        const float d = amax / 127.0f;
        uint8_t *out = wq.data() + m * w_row + b * 34;
        ggml_fp16_t d16 = ggml_fp32_to_fp16(d);
        memcpy(out, &d16, 2);
        for (int i = 0; i < 32; ++i) {
          out[2 + i] = (uint8_t)(int)lrintf(blk[i] / d);
        }
      }
    }
  }
  std::vector<float> x(K * N);
  fill_rnd(x);

  CtxGuard wctx;
  wctx.reset(2, 0);
  ggml_tensor *w = ggml_new_tensor_2d(wctx.ctx, f16 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0, K, M);
  if (!alloc_weights(wctx.ctx, split_copy ? g.cpu_buft : g.htp_buft)) {
    report_fail(name, "alloc failed");
    return false;
  }
  set_bytes(w, wq.data(), w_row * M);

  Graph G;
  G.init(4, 4);
  // split_copy=true：x 预分配到 CPU buffer（w 留 HTP），迫使 sched 插入
  // CPU→HTP 拷贝；split_copy 时 w 改留 CPU、xt 留 sched（HTP）不再构造该向，
  // 见下方 wq 为 CPU buffer 时的输入侧（此实验聚焦 CPU→HTP 方向）
  ggml_tensor *xt = ggml_new_tensor_2d(G.ctx.ctx, GGML_TYPE_F32, K, N);
  if (split_copy) {
    ggml_backend_alloc_ctx_tensors_from_buft(G.ctx.ctx, g.cpu_buft);
  }
  ggml_build_forward_expand(G.g, ggml_mul_mat(G.ctx.ctx, w, xt));

  std::vector<float> got;
  const int splits = run_graph({g.htp, g.cpu}, G.g, [&] { set_f32(xt, x); },
                               [&](ggml_cgraph *) { get_f32(last_node(G.g), got); });
  if (splits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  std::vector<float> ref(static_cast<size_t>(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += wf[(size_t)m * K + k] * x[(size_t)k + (size_t)n * K];
      ref[(size_t)m + (size_t)n * M] = s;
    }

  const Compare c = compare(ref, got);
  const bool pass = c.rel_rms() <= 1e-2;
  report(name, pass, c, splits);
  return pass;
}


// T9b：HTP→CPU 方向拷贝（w 预分配 HTP、输出自然落 CPU 图段）
bool test_mul_mat_wt_rev(int K, int M, int N, bool f16) {
  char name[72];
  std::snprintf(name, sizeof(name), "T9b MUL_MAT %s K=%-5d M=%-5d N=%-5d (w@HTP)",
                f16 ? "F16" : "Q8_0", K, M, N);
  std::vector<float> wf(static_cast<size_t>(M) * K);
  for (auto &v : wf) v = rnd(-1.0f, 1.0f);
  const size_t w_row = f16 ? K * 2 : (K / 32) * 34;
  std::vector<uint8_t> wq(w_row * M);
  if (f16) {
    auto *h = reinterpret_cast<ggml_fp16_t *>(wq.data());
    for (size_t i = 0; i < wf.size(); ++i) h[i] = ggml_fp32_to_fp16(wf[i]);
  } else {
    for (size_t m = 0; m < (size_t)M; ++m) {
      const float *row = wf.data() + m * K;
      for (size_t b = 0; b < (size_t)(K / 32); ++b) {
        const float *blk = row + b * 32;
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::abs(blk[i]));
        const float d = amax / 127.0f;
        uint8_t *out = wq.data() + m * w_row + b * 34;
        ggml_fp16_t d16 = ggml_fp32_to_fp16(d);
        memcpy(out, &d16, 2);
        for (int i = 0; i < 32; ++i) out[2 + i] = (uint8_t)(int)lrintf(blk[i] / d);
      }
    }
  }
  std::vector<float> x(K * N);
  fill_rnd(x);

  CtxGuard wctx;
  wctx.reset(2, 0);
  ggml_tensor *w = ggml_new_tensor_2d(wctx.ctx, f16 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0, K, M);
  if (!alloc_weights(wctx.ctx, g.htp_buft)) {
    report_fail(name, "alloc failed");
    return false;
  }
  set_bytes(w, wq.data(), w_row * M);

  Graph G;
  G.init(4, 4);
  // 输入与权重占位：让 MUL_MAT 归 CPU（把 w 视图到 CPU buffer 的副本），
  // 迫使 sched 把 HTP 权重拷到 CPU —— 测 HTP→CPU 方向
  ggml_tensor *xt = ggml_new_tensor_2d(G.ctx.ctx, GGML_TYPE_F32, K, N);
  ggml_build_forward_expand(G.g, ggml_mul_mat(G.ctx.ctx, w, xt));

  std::vector<float> got;
  const int splits = run_graph({g.htp, g.cpu}, G.g, [&] { set_f32(xt, x); },
                               [&](ggml_cgraph *) { get_f32(last_node(G.g), got); });
  if (splits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }
  std::vector<float> ref(static_cast<size_t>(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += wf[(size_t)m * K + k] * x[(size_t)k + (size_t)n * K];
      ref[(size_t)m + (size_t)n * M] = s;
    }
  const Compare c = compare(ref, got);
  const bool pass = c.rel_rms() <= 1e-2;
  report(name, pass, c, splits);
  return pass;
}

// =============================================================================
// T8：convT lowering 端到端 vs ggml_conv_transpose_1d 原生语义
//
//   ggml 内核（ops.cpp conv_transpose_1d_f32）：
//     Y[cout, q] = Σ_k Σ_{t·s+k=q} Σ_cin w3[k][cout][cin]·x[t][cin]
//   真实折叠文件约定（resolve_transpose_conv1d_spec）：
//     W_orig [K*Cout, Cin] 行 r = cout*K + k（cout 主序，k 最快）。
//   拆半行重排（audio-vae split_convt_weight_rows 同款映射）：
//     W_A 行 i = k*Cout + cout   ← W_orig 行 cout*K + k    (k∈[0,s))
//     W_B 行 i = (k-s)*Cout + cout ← W_orig 行 cout*K + k  (k∈[s,2s))
//   F_A/F_B = mul_mat(·, XT[Cin,T]) → [s·Cout, T]
//   FA3/FB3 = reshape_3d(·, Cout, s, T)
//   D  = add(view_3d(FA3, [Cout,s,T-1], off=s·Cout), view_3d(FB3, [Cout,s,T-1], 0))
//   D2 = reshape_2d(D, Cout, (T-1)s)
//   Y  = concat(view_2d(FA3, [Cout,s]), D2, dim=1)   // [Cout, T·s]，截尾内建
//   Yb = add(Y, bias [Cout,1])
// =============================================================================

bool test_conv_t(int Cin, int Cout, int s, int T) {
  char name[72];
  std::snprintf(name, sizeof(name), "T8 convT Cin=%d Cout=%d s=%d T=%d", Cin, Cout, s, T);
  const int K = 2 * s;
  const size_t row_bytes = static_cast<size_t>(Cin) / 32 * 18;

  // 原始权重字节（Q4_0）与激活 / bias
  std::vector<uint8_t> worig;
  gen_q4_0(worig, K * Cout, Cin);
  std::vector<float> xb(Cin * T);
  fill_rnd(xb);
  std::vector<float> biasv(Cout);
  fill_rnd(biasv, -0.1f, 0.1f);

  // 文件行（cout 主序）行重排拆半：与 audio-vae split_convt_weight_rows 同款
  std::vector<uint8_t> wa(static_cast<size_t>(s) * Cout * row_bytes, 0),
      wb(static_cast<size_t>(s) * Cout * row_bytes, 0);
  for (int c = 0; c < Cout; ++c) {
    for (int k = 0; k < K; ++k) {
      const uint8_t *src = worig.data() + static_cast<size_t>(c * K + k) * row_bytes;
      if (k < s) {
        memcpy(wa.data() + static_cast<size_t>(k * Cout + c) * row_bytes, src, row_bytes);
      } else {
        memcpy(wb.data() + static_cast<size_t>((k - s) * Cout + c) * row_bytes, src,
               row_bytes);
      }
    }
  }

  // ---- HTP 侧 lowering 图 ----
  CtxGuard wctx;
  wctx.reset(6, 0);
  ggml_tensor* wat = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_Q4_0, Cin, s * Cout);
  ggml_tensor* wbt = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_Q4_0, Cin, s * Cout);
  if (!alloc_weights(wctx.ctx, g.htp_buft)) {
    report_fail(name, "HTP weights alloc failed");
    return false;
  }
  set_bytes(wat, wa.data(), wa.size());
  set_bytes(wbt, wb.data(), wb.size());

  Graph G;
  G.init(24, 24);
  ggml_tensor* xt = ggml_new_tensor_2d(G.ctx.ctx, GGML_TYPE_F32, Cin, T);
  ggml_tensor* biast = ggml_new_tensor_2d(G.ctx.ctx, GGML_TYPE_F32, Cout, 1);
  ggml_tensor* fa = ggml_mul_mat(G.ctx.ctx, wat, xt);
  ggml_tensor* fb = ggml_mul_mat(G.ctx.ctx, wbt, xt);
  ggml_tensor* fa3 = ggml_reshape_3d(G.ctx.ctx, fa, Cout, s, T);
  ggml_tensor* fb3 = ggml_reshape_3d(G.ctx.ctx, fb, Cout, s, T);
  const size_t nb1 = sizeof(float) * Cout;
  const size_t nb2 = nb1 * s;
  ggml_tensor* d = ggml_add(G.ctx.ctx,
                            ggml_view_3d(G.ctx.ctx, fa3, Cout, s, T - 1, nb1, nb2,
                                         sizeof(float) * Cout * s),
                            ggml_view_3d(G.ctx.ctx, fb3, Cout, s, T - 1, nb1, nb2, 0));
  ggml_tensor* d2 = ggml_reshape_2d(G.ctx.ctx, d, Cout, (T - 1) * s);
  ggml_tensor* head = ggml_view_2d(G.ctx.ctx, fa3, Cout, s, nb1, 0);
  ggml_tensor* y = ggml_concat(G.ctx.ctx, head, d2, 1);
  ggml_build_forward_expand(G.g, ggml_add(G.ctx.ctx, y, biast));

  // ---- CPU 原生 conv_transpose_1d 参考 ----
  // wf：反量化到 host 行主序 [r = cout*K+k][ci]（真实文件行语义，ci 最快）；
  // oracle 解读：w3(k,c,ci) = wf[(c*K + k)*Cin + ci]（unfold 的纯 reshape 数学）。
  // w3r 的 ggml 内存序是 ne0=K 最快（flat = k + c*K + ci*K*Cout），须显式重排灌入。
  // xb 本身按 x[t][ci] 行主序生成（灌入 [Cin,T] 张量即 L2 布局），直接作 L1 输入
  std::vector<float> wf(static_cast<size_t>(K) * Cout * Cin);
  for (int r = 0; r < K * Cout; ++r) {
    dequant_q4_0_row(worig.data() + static_cast<size_t>(r) * row_bytes, Cin,
                     wf.data() + static_cast<size_t>(r) * Cin);
  }
  std::vector<float> w3f(static_cast<size_t>(K) * Cout * Cin);
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < Cout; ++c) {
      for (int ci = 0; ci < Cin; ++ci) {
        w3f[static_cast<size_t>(ci) * (K * Cout) + static_cast<size_t>(c) * K + k] =
            wf[(static_cast<size_t>(c) * K + k) * Cin + ci];
      }
    }
  }
  // convT 输入 b [T, Cin] 的内存约定：ne[1]=Cin 是行数，行内是时间 → x[ci][t]
  std::vector<float> xl1(static_cast<size_t>(Cin) * T);
  for (int t = 0; t < T; ++t) {
    for (int ci = 0; ci < Cin; ++ci) {
      xl1[static_cast<size_t>(ci) * T + t] = xb[static_cast<size_t>(t) * Cin + ci];
    }
  }

  Graph Rw;
  Rw.init(8, 8);
  // ggml_conv_transpose_1d 权重约定 [K, Cout, Cin]（ne[2] == 输入通道）
  ggml_tensor* w3r = ggml_new_tensor_3d(Rw.ctx.ctx, GGML_TYPE_F32, K, Cout, Cin);
  ggml_tensor* x3r = ggml_new_tensor_3d(Rw.ctx.ctx, GGML_TYPE_F32, T, Cin, 1);
  ggml_tensor* biasr = ggml_new_tensor_2d(Rw.ctx.ctx, GGML_TYPE_F32, Cout, 1);
  // 输出 [T', Cout]，T' = (T-1)s + K；截尾去 s → [T*s, Cout]
  ggml_tensor* convr = ggml_conv_transpose_1d(Rw.ctx.ctx, w3r, x3r, s, 0, 1);
  ggml_tensor* ycrop = ggml_view_2d(Rw.ctx.ctx, convr, T * s, Cout, convr->nb[1], 0);
  // ycrop [T*s, Cout]：时间在 ne0，bias 须 [1, Cout] 才能逐通道广播
  ggml_build_forward_expand(Rw.g, ggml_add(Rw.ctx.ctx, ycrop,
                                           ggml_reshape_2d(Rw.ctx.ctx, biasr, 1, Cout)));
  ggml_backend_alloc_ctx_tensors_from_buft(Rw.ctx.ctx, g.cpu_buft);
  set_f32(w3r, w3f);

  std::vector<float> got, ref;
  const int splits = run_graph({g.htp, g.cpu}, G.g,
                               [&] {
                                 set_f32(xt, xb);
                                 set_f32(biast, biasv);
                               },
                               [&](ggml_cgraph*) { get_f32(last_node(G.g), got); });
  // 参考图张量全部预分配：绕开 sched，直接灌入 + graph_compute（对照探针路径）
  set_f32(x3r, xl1);
  set_f32(biasr, biasv);
  if (ggml_backend_graph_compute(g.cpu, Rw.g) != GGML_STATUS_SUCCESS) {
    report_fail(name, "ref compute failed");
    return false;
  }
  get_f32(last_node(Rw.g), ref);
  if (splits < 0) {
    report_fail(name, "sched compute failed");
    return false;
  }

  // got [Cout, T*s]（L2，ne[0]=通道最快维）：flat = c + q*Cout；
  // ref [T*s, Cout]（L1，ne[0]=时间最快维）：flat = q + c*(T*s)
  Compare c;
  for (int q = 0; q < T * s; ++q) {
    for (int cc = 0; cc < Cout; ++cc) {
      const double rv = ref[static_cast<size_t>(q) + static_cast<size_t>(cc) * (T * s)];
      const double dd = static_cast<double>(got[static_cast<size_t>(q) * Cout + cc]) - rv;
      c.max_abs = std::max(c.max_abs, std::abs(dd));
      c.rms_err += dd * dd;
      c.rms_ref += rv * rv;
    }
  }
  c.rms_err = std::sqrt(c.rms_err / static_cast<double>(got.size()));
  c.rms_ref = std::sqrt(c.rms_ref / static_cast<double>(got.size()));
  // GEMM 类门槛：rel_rms（Q4_0 反量化点积的 f16/f32 累加差天然给出 max|Δ| 长尾，
  // 不适用逐位门槛）
  const bool pass = c.rel_rms() <= 1e-2;
  report(name, pass, c, splits);
  return pass;
}

// =============================================================================
// T10：L2Plan seg1 复现器 — direct graph_compute（无 sched），Q8_0 GEMM。
// 复现真机 decode-mock L2 拆段 EUNSUPPORTED（seg1: 2×MUL_MAT 共享 x，
// w Q8_0 [1536,6144]@HTP buft(USAGE_WEIGHTS)，x[1536,6] 由段 gallocr 分配）。
// 环境变量二分变量：
//   T10_OPS=1|2      MUL_MAT 个数（默认 2，复刻共享输入）
//   T10_XBUF=alloc|cpu|htp  x 分配方式：gallocr(HTP)/预分配 CPU/预分配 HTP
//   T10_K=<n>        权重 K 维（默认 1536）
// =============================================================================

bool test_seg1_replica() {
  const char* e_ops = std::getenv("T10_OPS");
  const int n_ops = e_ops ? std::atoi(e_ops) : 2;
  const char* e_xbuf = std::getenv("T10_XBUF");
  const std::string xbuf = e_xbuf ? e_xbuf : "alloc";
  const char* e_k = std::getenv("T10_K");
  const int K = e_k ? std::atoi(e_k) : 1536;
  const int M = 6144, N = 6;

  char name[96];
  std::snprintf(name, sizeof(name), "T10 replica ops=%d K=%-5d xbuf=%-5s", n_ops, K,
                xbuf.c_str());

  // 权重字节：Q8_0 [K, M]（确定性内容，scale 0.01）
  const size_t row_bytes = static_cast<size_t>(K) / 32 * 34;
  std::vector<uint8_t> wbytes(row_bytes * M);
  for (size_t r = 0; r < static_cast<size_t>(M); ++r) {
    for (size_t b = 0; b < static_cast<size_t>(K) / 32; ++b) {
      uint8_t* blk = wbytes.data() + r * row_bytes + b * 34;
      const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f);
      memcpy(blk, &d, 2);
      for (int i = 0; i < 32; ++i) {
        blk[2 + i] = static_cast<uint8_t>((r + b + i) % 15);
      }
    }
  }
  std::vector<float> x(K * N);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = rnd(-1.0f, 1.0f);
  }

  // ---- 权重上下文（HTP buft + USAGE_WEIGHTS，与 derived 桶同构）----
  CtxGuard wctx;
  wctx.reset(2 * static_cast<size_t>(n_ops), 0);
  ggml_tensor* ws[2] = {nullptr, nullptr};
  for (int i = 0; i < n_ops; ++i) {
    ws[i] = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_Q8_0, K, M);
  }
  ggml_backend_buffer_t wbuf = alloc_weights(wctx.ctx, g.htp_buft);
  if (!wbuf) {
    report_fail(name, "HTP weights alloc failed");
    return false;
  }
  for (int i = 0; i < n_ops; ++i) {
    set_bytes(ws[i], wbytes.data(), wbytes.size());
  }

  // ---- 图：n_ops × MUL_MAT 共享 xt ----
  Graph G;
  G.init(8, 8);
  ggml_tensor* xt = ggml_new_tensor_2d(G.ctx.ctx, GGML_TYPE_F32, K, N);
  ggml_tensor* outs[2] = {nullptr, nullptr};
  outs[0] = ggml_mul_mat(G.ctx.ctx, ws[0], xt);
  ggml_build_forward_expand(G.g, outs[0]);
  if (n_ops > 1) {
    outs[1] = ggml_mul_mat(G.ctx.ctx, ws[1], xt);
    ggml_build_forward_expand(G.g, outs[1]);
  }

  // ---- 执行路径对比（T10_MODE）：direct / sched1（alloc+set+单跑）/
  // sched2（首跑分配 + set + 二跑）。L2Plan 现为 direct（真机 FAIL）；
  // htp-smoke T1-T9 全部为 sched2（真机 PASS）。
  const char* e_mode = std::getenv("T10_MODE");
  const std::string mode = e_mode ? e_mode : "direct";
  std::snprintf(name + strlen(name), sizeof(name) - strlen(name), " mode=%s",
                mode.c_str());

  ggml_status st = GGML_STATUS_FAILED;
  ggml_backend_sched_t sched = nullptr;
  ggml_gallocr_t gallocr = nullptr;
  if (mode == "direct") {
    // 复刻 L2Plan：gallocr reserve+alloc 后 set，direct graph_compute
    gallocr = ggml_gallocr_new(g.htp_buft);
    if (!gallocr || !ggml_gallocr_reserve(gallocr, G.g) ||
        !ggml_gallocr_alloc_graph(gallocr, G.g)) {
      std::printf("%-42s | gallocr failed | FAIL\n", name);
      std::fflush(stdout);
      if (gallocr) ggml_gallocr_free(gallocr);
      ++g_failures;
      return false;
    }
    if (xbuf == "cpu") {
      ggml_backend_buffer_t xbuf_cpu =
          ggml_backend_buft_alloc_buffer(g.cpu_buft, ggml_nbytes(xt));
      xt->buffer = xbuf_cpu;
      xt->data = xbuf_cpu ? ggml_backend_buffer_get_base(xbuf_cpu) : nullptr;
      if (!xt->data) {
        report_fail(name, "cpu xbuf alloc failed");
        return false;
      }
    }
    set_f32(xt, x);
    st = ggml_backend_graph_compute(g.htp, G.g);
  } else {
    // sched 路径（htp-smoke run_graph 成功模式）
    ggml_backend_t bes[2] = {g.htp, g.cpu};
    sched = ggml_backend_sched_new(bes, nullptr, 2, 256, false,
                                   /*op_offload=*/false);
    if (!sched || !ggml_backend_sched_reserve(sched, G.g) ||
        !ggml_backend_sched_alloc_graph(sched, G.g)) {
      std::printf("%-42s | sched init failed | FAIL\n", name);
      std::fflush(stdout);
      if (sched) ggml_backend_sched_free(sched);
      ++g_failures;
      return false;
    }
    if (mode == "sched2") {
      // 首跑：输入为 sched 副本垃圾（预热 + 分配完成）
      st = ggml_backend_sched_graph_compute(sched, G.g);
    }
    set_f32(xt, x);
    const ggml_status st2 = ggml_backend_sched_graph_compute(sched, G.g);
    st = (st == GGML_STATUS_SUCCESS && st2 == GGML_STATUS_SUCCESS) ? GGML_STATUS_SUCCESS
                                                                  : st2;
  }
  if (st != GGML_STATUS_SUCCESS) {
    std::printf("%-42s | exec st=%d | FAIL\n", name, static_cast<int>(st));
    std::fflush(stdout);
    if (sched) ggml_backend_sched_free(sched);
    if (gallocr) ggml_gallocr_free(gallocr);
    ++g_failures;
    return false;
  }
  if (sched) ggml_backend_sched_free(sched);
  if (gallocr) ggml_gallocr_free(gallocr);

  // ---- 读回 + CPU 参考（抽查 dst[0]：权重行 0 与 x 列 0 的 Q8_0 点积）----
  bool pass = true;
  for (int i = 0; i < n_ops; ++i) {
    std::vector<float> got;
    get_f32(outs[i], got);
    double ref0 = 0.0;
    for (int k = 0; k < K; ++k) {
      const size_t blk = static_cast<size_t>(k) / 32;
      const uint8_t* wblk = wbytes.data() + blk * 34;
      ggml_fp16_t d;
      memcpy(&d, wblk, 2);
      const float df = ggml_fp16_to_fp32(d);
      const int q = static_cast<int8_t>(wblk[2 + (k % 32)]);
      ref0 += static_cast<double>(q) * df * x[k];
    }
    const double d0 = std::abs(static_cast<double>(got[0]) - ref0);
    const double scale = std::abs(ref0) > 0 ? d0 / std::abs(ref0) : d0;
    std::printf("%-42s | out%d got[0]=%-12.6g ref[0]=%-12.6g rel=%.3g | %s\n", name, i,
                got[0], ref0, scale, scale <= 1e-2 ? "PASS" : "FAIL");
    std::fflush(stdout);
    pass = pass && scale <= 1e-2;
  }
  if (!pass) {
    ++g_failures;
  }
  return pass;
}

}  // namespace

int main() {
  ggml_backend_dev_t dev = ggml_backend_dev_by_name("HTP0");
  if (!dev) {
    dev = ggml_backend_dev_by_name("HTP");
  }
  g.cpu = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(g.cpu, 4);
  g.cpu_buft = ggml_backend_cpu_buffer_type();

  if (dev) {
    g.htp = ggml_backend_dev_init(dev, nullptr);
    if (g.htp) {
      g_htp_present = true;
      g.htp_buft = ggml_backend_dev_buffer_type(dev);
      std::printf("devices: %s(%s) + %s\n\n", ggml_backend_dev_name(dev),
                  ggml_backend_dev_description(dev), ggml_backend_name(g.cpu));
    }
  }
  if (!g_htp_present) {
    // 纯 CPU 自检：两侧同走 CPU，验证图构建 / 视图布局 / 截尾语义（逐位一致）
    g.htp = g.cpu;
    g.htp_buft = g.cpu_buft;
    std::printf("HTP device 不可用（非 GGML_HEXAGON=ON 构建或 DSP 初始化失败），"
                "退化为 CPU 自检模式——仅验证图语义，不代表 HTP 数值\n\n");
  }

  bool ok = true;
  // T1：pw conv 与 convT 半权重代表形状（K 均 %32==0）
  ok &= test_mul_mat(64, 1536, 6);
  ok &= test_mul_mat(1536, 768, 48);
  ok &= test_mul_mat(768, 384, 384);
  ok &= test_mul_mat(384, 192, 1920);
  ok &= test_mul_mat(64, 6144, 48);
  ok &= test_mul_mat_wt(64, 1536, 6, true);
  ok &= test_mul_mat_wt(1536, 6144, 6, true);
  ok &= test_mul_mat_wt(64, 1536, 6, false);
  ok &= test_mul_mat_wt(1536, 6144, 6, false);
  ok &= test_mul_mat_wt_rev(64, 1536, 6, true);
  ok &= test_mul_mat_wt(64, 1536, 6, false, /*split_copy=*/true);
  ok &= test_mul_mat_wt(1536, 6144, 6, true, /*split_copy=*/true);
  // T2：bias / alpha 广播
  ok &= test_broadcast("ADD", true, 1536, 48);
  ok &= test_broadcast("MUL", false, 192, 1920);
  // T3/T4：convT lowering 的视图与拼接
  ok &= test_view_add(768, 8, 6);
  ok &= test_concat(768, 8, 6);
  // T5：snake 链
  ok &= test_snake_chain(384, 384);
  // T14：融合 snake（B1，纯 CPU 多项式 sin vs std::sin）
  ok &= test_snake_fused(96, 3840);
  ok &= test_snake_fused(384, 48);
  // T6/T7
  ok &= test_get_rows(384, 100, 12);
  ok &= test_tanh(96, 3840);
  // T11：SIN（snake 上 HTP 的前置）
  ok &= test_sin(96, 3840);
  ok &= test_snake_full(96, 3840);
  ok &= test_snake_full(384, 48);
  // T8：convT lowering 端到端（block0 / block3 形状）
  ok &= test_conv_t(64, 768, 8, 6);
  ok &= test_conv_t(384, 96, 2, 48);
  // T10：L2Plan seg1 复现器（T10_OPS/T10_XBUF/T10_K 环境变量二分）
  {
    const char* e_ops = std::getenv("T10_OPS");
    ok &= test_seg1_replica();
    (void)e_ops;
  }

  std::printf("\n%s (%d failures)\n", ok ? "SMOKE OK" : "SMOKE FAILED", g_failures);
  return ok ? 0 : 1;
}
