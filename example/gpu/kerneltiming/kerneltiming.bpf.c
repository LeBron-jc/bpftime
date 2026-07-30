#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_RINGBUF_MAP 1527
#define KERN_NAME_LEN 64

struct event {
	u64 type;
	u64 bid_x, bid_y, bid_z;
	u64 tid_x, tid_y, tid_z;
	u64 timestamp;
	char kern_name[KERN_NAME_LEN];
};

struct {
	__uint(type, BPF_MAP_TYPE_GPU_RINGBUF_MAP);
	__uint(max_entries, 32);
	__type(key, u32);
	__type(value, struct event);
} rb SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

static __always_inline void
push_event(u64 type, const char *name, u64 bx, u64 by, u64 bz,
	   u64 tx, u64 ty, u64 tz, u64 ts)
{
	struct event e = {0};
	e.type = type;
	e.bid_x = bx; e.bid_y = by; e.bid_z = bz;
	e.tid_x = tx; e.tid_y = ty; e.tid_z = tz;
	e.timestamp = ts;
	for (int i = 0; i < KERN_NAME_LEN - 1 && name[i]; i++)
		e.kern_name[i] = name[i];
	bpf_perf_event_output(NULL, &rb, 0, &e, sizeof(e));
}

// ── DEMO hooks (vec_add kernels) ──
SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__vec_add_enter()
{
	u64 bx, by, bz, tx, ty, tz;
	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_thread_idx(&tx, &ty, &tz);
	push_event(0, "vectorAdd", bx, by, bz, tx, ty, tz, bpf_get_globaltimer());
	return 0;
}

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__vec_add_exit()
{
	u64 bx, by, bz, tx, ty, tz;
	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_thread_idx(&tx, &ty, &tz);
	push_event(1, "vectorAdd", bx, by, bz, tx, ty, tz, bpf_get_globaltimer());
	return 0;
}

SEC("kprobe/_Z11multiplyAddPKfS0_Pf")
int cuda__mul_add_enter()
{
	u64 bx, by, bz, tx, ty, tz;
	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_thread_idx(&tx, &ty, &tz);
	push_event(0, "multiplyAdd", bx, by, bz, tx, ty, tz, bpf_get_globaltimer());
	return 0;
}

SEC("kretprobe/_Z11multiplyAddPKfS0_Pf")
int cuda__mul_add_exit()
{
	u64 bx, by, bz, tx, ty, tz;
	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_thread_idx(&tx, &ty, &tz);
	push_event(1, "multiplyAdd", bx, by, bz, tx, ty, tz, bpf_get_globaltimer());
	return 0;
}

// ── relu kernel ──
SEC("kprobe/_Z4reluPf")
int cuda__relu_enter()
{
	u64 bx, by, bz, tx, ty, tz;
	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_thread_idx(&tx, &ty, &tz);
	push_event(0, "relu", bx, by, bz, tx, ty, tz, bpf_get_globaltimer());
	return 0;
}

SEC("kretprobe/_Z4reluPf")
int cuda__relu_exit()
{
	u64 bx, by, bz, tx, ty, tz;
	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_thread_idx(&tx, &ty, &tz);
	push_event(1, "relu", bx, by, bz, tx, ty, tz, bpf_get_globaltimer());
	return 0;
}

// ── EXAMPLE: PyTorch kernel hooks (uncomment when torch is ready) ──
// To find kernel names: run torch test with bpftime agent,
// then check launch trace for kernel_name entries.

// torch.mm() on Ampere → cuBLAS sgemm kernel:
// SEC("kprobe/_ZN10cutlass_gemm...")
// int torch_mm_enter() { ... }

// torch.relu() kernel:
// SEC("kprobe/_ZN2at6native...")
// int torch_relu_enter() { ... }

char LICENSE[] SEC("license") = "GPL";
