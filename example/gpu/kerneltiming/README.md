# kerneltiming：算子 → Kernel → 逐 Warp 耗时

从 CPU 端算子名一路追踪到 GPU 内部每个 warp 的执行时间，完整的三层映射链。

```
算子                          →  CUDA Kernel   →  耗时分布
at::native::matmul_out        →  vectorAdd     →  p50=2.2ms  p90=3.0ms  max=3.2ms
at::native::relu_out          →  multiplyAdd   →  p50=2.4ms  p90=2.7ms  max=2.7ms
```

## 原理

1. **CPU 端** — bpftime 拦截 `cudaLaunchKernel`，通过调用栈回溯解析出是哪个算子
   （如 `at::native::matmul_out`）触发了这次 kernel 启动
2. **GPU 端** — eBPF kprobe/kretprobe 注入到 kernel 的入口和出口，记录每个线程的
   纳秒级时间戳
3. **Ring Buffer** — GPU 上的事件通过共享内存 ring buffer 传到 host
4. **轮询** — host 每 100ms 排空 ring buffer，配对 enter/exit 事件，每秒输出统计

## 输出格式

```
==== at::native::matmul_out -> vectorAdd <<<(4,1,1),(256,1,1)>>> ====
  warps=32  p50=2.2ms  p90=3.0ms  p99=3.2ms  max=3.2ms
  .:.-:...=-*==-+=:-----::##%%###@  1.8..3.2ms
-- total events: 2048 --
```

- **标题** — `算子 → kernel <<<(grid大小),(block大小)>>>`
- **warps** — 捕获的 warp 数量（32 = 全部 1024 个线程无损）
- **p50 / p90 / p99** — warp 耗时的分位数，比平均值更有参考价值
- **直方图** — 32 个字符，每个代表一个 warp（未排序，按 warp ID 原始顺序）：
  `.` 最快 → `@` 最慢，一眼看出各 warp 之间的负载分布
- **total events** — 每秒 enter+exit 配对事件总数

## 编译

```bash
# 确保 bpftime 已启用 CUDA 支持
cmake -Bbuild -DBPFTIME_ENABLE_CUDA_ATTACH=ON -DBPFTIME_CUDA_ROOT=/usr/local/cuda
cmake --build build -j$(nproc)

# 编译本例
make -C example/gpu/kerneltiming
```

## 运行

需要两个终端。

**终端 1 — eBPF 服务端：**
```bash
cd bpftime
BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/kerneltiming/kerneltiming
```

**终端 2 — GPU 应用（三选一）：**

*算子→kernel 演示（无需 PyTorch）：*
```bash
BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/kerneltiming/torch_ops
```

*纯 kernel 演示（无算子名）：*
```bash
BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/kerneltiming/vec_add
```

*对接真实 PyTorch（需从源码编译 PyTorch 并开启 PTX）：*
```bash
BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  python3 example/gpu/kerneltiming/torch_test.py
```

## 添加 PyTorch Kernel Hook

在 `kerneltiming.bpf.c` 中加入目标 kernel 的 mangled 名：

```c
// torch.mm() 在 Ampere 上调用 cuBLAS gemm kernel：
SEC("kprobe/_ZN10cutlass6gemm...")
int torch_mm_enter() { ... }

// torch.relu()：
SEC("kprobe/_ZN2at6native...")
int torch_relu_enter() { ... }
```

先用 bpftime agent 跑一次 torch 测试，从 launch trace 日志中获取 kernel 名，
再填入 BPF 文件即可。

## 文件说明

| 文件 | 用途 |
|------|------|
| `kerneltiming.bpf.c` | eBPF 探针：GPU kernel 的 kprobe/kretprobe |
| `kerneltiming.c` | Host 端加载器：排空 ringbuf，输出统计 |
| `vec_add.cu` | 简易 CUDA 测试（vectorAdd + multiplyAdd） |
| `torch_ops.cu` | 算子→kernel 演示，用 `at::native::` 命名空间模拟 |
| `torch_test.py` | 真实 PyTorch 测试（`torch.mm` + `torch.relu`） |
| `Makefile` | 编译所有目标 |

## 常见问题

- **段错误 / 共享内存错误** — 清理残留共享内存：
  ```bash
  rm -f /dev/shm/bpftime_*
  ```
- **CUDA 初始化报错 999** — nvidia-uvm 模块卡死（常见于笔记本休眠唤醒后）：
  ```bash
  sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
  ```
- **只捕获到 ~11 个 warp** — ringbuf 溢出。增大 `kerneltiming.bpf.c` 中的
  `max_entries`，或减小 `kerneltiming.c` 中的轮询间隔。
- **没有算子名** — 调用栈中必须包含 demangled 名带 `at::native::` 的函数。
  用 C++ namespace 包裹你的测试函数，或用真实 PyTorch 运行。
