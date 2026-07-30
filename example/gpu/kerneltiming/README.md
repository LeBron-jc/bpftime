# kerneltiming: Operator → Kernel → Per-Warp Timing

Full-chain GPU kernel observability: from CPU-side operator name down to
per-warp execution time inside the GPU.

```
Operator                     →  CUDA Kernel    →  Timing
at::native::matmul_out       →  vectorAdd      →  p50=2.2ms  p90=3.0ms  max=3.2ms
at::native::relu_out         →  multiplyAdd    →  p50=2.4ms  p90=2.7ms  max=2.7ms
```

## How It Works

1. **CPU side** — bpftime intercepts `cudaLaunchKernel` and walks the call
   stack to resolve which operator (e.g. `at::native::matmul_out`) triggered
   the kernel launch
2. **GPU side** — eBPF kprobe/kretprobe attached to the kernel entry/exit
   captures nanosecond timestamps per thread
3. **Ring buffer** — GPU events streamed to host via shared-memory ring buffer
4. **Polling** — host drains the ring buffer every 100ms, matches enter/exit
   pairs, and prints per-second summary

## Output Format

```
==== at::native::matmul_out -> vectorAdd <<<(4,1,1),(256,1,1)>>> ====
  warps=32  p50=2.2ms  p90=3.0ms  p99=3.2ms  max=3.2ms
  .:.-:...=-*==-+=:-----::##%%###@  1.8..3.2ms
-- total events: 2048 --
```

- **Header** — `operator → kernel <<<(grid),(block)>>>`
- **warps** — number of warps captured (32 = all 1024 threads)
- **p50 / p90 / p99** — warp duration percentiles
- **Histogram bar** — 32 characters, one per warp (unsorted, by warp ID):
  `.` = fastest, `@` = slowest; shows spatial distribution across warps
- **total events** — total enter+exit pairs per second

## Building

```bash
# Ensure bpftime is built with CUDA support
cmake -Bbuild -DBPFTIME_ENABLE_CUDA_ATTACH=ON -DBPFTIME_CUDA_ROOT=/usr/local/cuda
cmake --build build -j$(nproc)

# Build this example
make -C example/gpu/kerneltiming
```

## Running

Two terminals required:

**Terminal 1 — eBPF server:**
```bash
cd bpftime
BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/kerneltiming/kerneltiming
```

**Terminal 2 — GPU application (choose one):**

*Operator→kernel demo (no PyTorch required):*
```bash
BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/kerneltiming/torch_ops
```

*Raw kernel demo (no operator names):*
```bash
BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/kerneltiming/vec_add
```

*With real PyTorch (requires torch built from source with PTX):*
```bash
BPFTIME_LOG_OUTPUT=console \
  LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  python3 example/gpu/kerneltiming/torch_test.py
```

## Adding PyTorch Kernel Hooks

To trace real PyTorch operators, add the kernel's mangled name to
`kerneltiming.bpf.c`:

```c
// torch.mm() on Ampere launches a cuBLAS gemm kernel:
SEC("kprobe/_ZN10cutlass6gemm...")
int torch_mm_enter() { ... }

// torch.relu() launches an elementwise kernel:
SEC("kprobe/_ZN2at6native...")
int torch_relu_enter() { ... }
```

Run the torch test with the bpftime agent once, then check the launch trace
in the server output to discover kernel names.

## Files

| File | Purpose |
|------|---------|
| `kerneltiming.bpf.c` | eBPF programs: kprobe/kretprobe on CUDA kernels |
| `kerneltiming.c` | Host loader: drains ringbuf, prints stats |
| `vec_add.cu` | Simple CUDA test (vectorAdd + multiplyAdd) |
| `torch_ops.cu` | Operator→kernel demo using `at::native::` wrappers |
| `torch_test.py` | Real PyTorch test (`torch.mm` + `torch.relu`) |
| `Makefile` | Build all targets |

## Troubleshooting

- **Segfault / shared memory error** — clean stale shm:
  ```bash
  rm -f /dev/shm/bpftime_*
  ```
- **CUDA init error 999** — nvidia-uvm module is stuck (common after suspend):
  ```bash
  sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
  ```
- **Only ~11 warps captured** — ringbuf overflow. Increase `max_entries` in
  `kerneltiming.bpf.c` or reduce poll interval in `kerneltiming.c`.
- **No operator names shown** — the call stack must contain functions with
  `at::native::` in their demangled name. Use proper C++ namespaces in your
  test code, or run with real PyTorch.
