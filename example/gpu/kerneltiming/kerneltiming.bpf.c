#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_RINGBUF_MAP 1527

struct big_struct {
	char s[1024];
};

struct {
	__uint(type, BPF_MAP_TYPE_GPU_RINGBUF_MAP);
	__uint(max_entries, 16);
	__type(key, u32);
	__type(value, struct big_struct);
} rb SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

// Type=0 means ENTRY, Type=1 means EXIT
struct event {
	u64 type;          // 0=enter, 1=exit
	u64 bid_x, bid_y, bid_z;
	u64 tid_x, tid_y, tid_z;
	u64 timestamp;
};

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe()
{
	u64 bx, by, bz, tx, ty, tz;
	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_thread_idx(&tx, &ty, &tz);

	struct event e = {
		.type = 0,  // enter
		.bid_x = bx, .bid_y = by, .bid_z = bz,
		.tid_x = tx, .tid_y = ty, .tid_z = tz,
		.timestamp = bpf_get_globaltimer(),
	};
	bpf_perf_event_output(NULL, &rb, 0, &e, sizeof(e));

	return 0;
}

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe()
{
	u64 bx, by, bz, tx, ty, tz;
	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_thread_idx(&tx, &ty, &tz);

	struct event e = {
		.type = 1,  // exit
		.bid_x = bx, .bid_y = by, .bid_z = bz,
		.tid_x = tx, .tid_y = ty, .tid_z = tz,
		.timestamp = bpf_get_globaltimer(),
	};
	bpf_perf_event_output(NULL, &rb, 0, &e, sizeof(e));

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
