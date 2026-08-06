# bpftime：用户态 eBPF 运行时 — GPU Kernel 可观测性增强

基于 bpftime 构建的 GPU 算子→Kernel→Warp 三层可观测性系统，实现从 CPU 端算子名到 GPU 内部每个 warp 执行时间的完整追踪链。

```
PyTorch 算子              →  CUDA Kernel           →  Warp 耗时分布
at::native::matmul_out →  vectorAdd               →  p50=2.2ms  p90=2.3ms  max=2.4ms
at::native::relu_out   →  multiplyAdd             →  p50=2.9ms  p90=3.6ms  max=3.8ms
```

## 原理

### 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Process                     │
│                                                              │
│  PyTorch/CUDA App ──▶ bpftime Agent ──▶ GPU Kernel          │
│       │                    │                   │             │
│       │  cudaLaunchKernel  │  PTX 注入        │ eBPF 探针   │
│       ▼                    ▼                   ▼             │
│   调用栈回溯           共享内存             Ring Buffer      │
│   解析算子名           传输数据            记录时间戳        │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                   bpftime Server Process                     │
│                                                              │
│   Launch Trace ──▶   算子→Kernel 映射                        │
│   Ring Buffer   ──▶   Warp 耗时统计                          │
│   kerneltiming  ──▶   per-second 输出 (percentile + 直方图)  │
└──────────────────────────────────────────────────────────────┘
```

### 三层映射链

**第一层：CPU 端算子名解析**

bpftime 通过 Frida 动态插桩 Hook `libcudart.so` 的 `cudaLaunchKernel`。每次 kernel 启动时，调用栈回溯（`backtrace` + `dladdr` + `__cxa_demangle`）解析出是哪个 C++ 函数触发了此次调用。匹配规则：demangled 符号名中包含 `at::native::` 或 `at::cuda::` 的帧即为算子名。

```
torch.mm(a, b)
  → at::native::matmul_out()     ← 在这里被抓到
    → cudaLaunchKernel(...)      ← bpftime 拦截点
```

**第二层：GPU Kernel 注入 eBPF 探针**

bpftime 拦截 CUDA fatbin 加载，将 eBPF 程序编译为 PTX 汇编码，通过 PTX Pass 注入到目标 kernel 的入口和出口。注入后的 kernel 在 GPU 上执行时会先运行 eBPF 探针，记录时间戳。

**第三层：逐 Warp 耗时采集**

eBPF 探针在 GPU 每个线程上记录纳秒级时间戳，通过 GPU Ring Buffer（共享内存零拷贝映射）传回 Host。Host 端每秒排空 ring buffer，配对 enter/exit 事件，计算每个 warp（32 线程一组）的执行时长。

### 关键修改

| 文件 | 修改 | 效果 |
|------|------|------|
| `nv_attach_impl_frida_setup.cpp` | `resolve_pytorch_caller_from_backtrace` 增加 `__cxa_demangle` | 修复算子名解析 bug（原版用 mangled name 搜索永远不会匹配） |
| `nv_gpu_ringbuf_map.cpp` | `drain_data` 中 `dirty` 检查从 `return 0` 改为 `continue` + `while` 排空 | 修复脏页锁死 bug（一个线程写入时阻塞全部数据读取），吞吐从 17% → 100% |
| `kerneltiming.bpf.c` | value_type 从 1024B 改为 event struct (72B) | 每线程 ringbuf 容量提升 14 倍 |
| `kerneltiming.c` | 复合 key（算子名,kernel名）、p50/p90/p99、未排序直方图 | 支持多算子共享 kernel 的独立追踪，输出维度更丰富 |

### 对比

| 指标 | bpftime 原版 | 修改后 |
|------|-------------|--------|
| Ringbuf 事件捕获率 | ~17% | ~100% |
| 算子名解析 | 不工作（bug） | 正常 |
| 多算子共享 kernel | 不支持 | 支持 |
| 输出维度 | avg/min/max | p50/p90/p99 + 直方图 |
| 支持的 kernel 数 | 2 | 3+（可扩展） |

## 快速开始

### 编译

```bash
git clone --recursive https://github.com/LeBron-jc/bpftime.git
cd bpftime

# 启用 CUDA 支持
cmake -Bbuild -DBPFTIME_ENABLE_CUDA_ATTACH=ON -DBPFTIME_CUDA_ROOT=/usr/local/cuda
cmake --build build -j$(nproc)

# 编译 kerneltiming 示例
make -C example/gpu/kerneltiming
```

### 运行

**终端 1（eBPF 服务端）：**
```bash
BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/kerneltiming/kerneltiming
```

**终端 2（GPU 应用 — 算子→kernel 演示）：**
```bash
BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/kerneltiming/torch_ops
```

### 输出示例

```
==== at::native::matmul_out(float const*, float const*, float*) -> vectorAdd <<<(4,1,1),(256,1,1)>>> ====
  warps=32  p50=2.2ms  p90=2.3ms  p99=2.4ms  max=2.4ms
  ####%+*#+*+%+.=+****-+**#@#%###%  1.5..2.4ms
==== at::native::matmul_out(float const*, float const*, float*) -> multiplyAdd <<<(4,1,1),(256,1,1)>>> ====
  warps=32  p50=2.9ms  p90=3.6ms  p99=3.7ms  max=3.8ms
  %=%-%=#=%=#+#=%.=+=+*+*+-=#+.+.+  1.8..3.8ms
==== at::native::relu_out(float*) -> relu <<<(4,1,1),(256,1,1)>>> ====
  warps=32  p50=2.6ms  p90=3.9ms  p99=4.0ms  max=4.0ms
  +:+-=.=-=-+-+:+-%-%-%-%-%-%:+-#-  1.5..4.0ms
-- total events: 5120 --
```

- **标题**：`算子名 → kernel名 <<<(grid),(block)>>>`
- **warps=32**：全部 32 个 warp（1024 线程）无损捕获
- **p50/p90/p99**：warp 耗时百分位数
- **直方图**：32 个字符，每个对应一个 warp（未排序，按 warp ID 原始顺序），`.` 最快 `#%@` 最慢
- **total events**：每秒 enter+exit 配对事件总数

## 文件说明

| 文件 | 用途 |
|------|------|
| `example/gpu/kerneltiming/kerneltiming.bpf.c` | eBPF 探针程序：注入到 GPU kernel 入口/出口 |
| `example/gpu/kerneltiming/kerneltiming.c` | Host 端：轮询 ringbuf、配对 enter/exit、输出统计 |
| `example/gpu/kerneltiming/torch_ops.cu` | 算子→kernel 演示程序（3 算子 × 3 kernel） |
| `example/gpu/kerneltiming/vec_add.cu` | 简易 CUDA 测试（无算子层） |
| `attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp` | 算子名解析、cudaLaunchKernel Hook |
| `runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.cpp` | GPU Ring Buffer 实现（Producer/Consumer） |

## 疑难解答

- **段错误 / 共享内存报错**：`rm -f /dev/shm/bpftime_*`
- **CUDA 初始化 error 999**（休眠后常见）：`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`
- **算子名不显示**：确保 client 程序有用 `at::native::` C++ namespace 包裹的调用，或使用真实 PyTorch
