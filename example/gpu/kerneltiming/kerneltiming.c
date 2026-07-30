// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <dlfcn.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "./.output/kerneltiming.skel.h"
#include <inttypes.h>
#define warn(...) fprintf(stderr, __VA_ARGS__)

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

#define KERN_NAME_LEN 64

struct event {
	uint64_t type;       // 0=enter, 1=exit
	uint64_t bid_x, bid_y, bid_z;
	uint64_t tid_x, tid_y, tid_z;
	uint64_t timestamp;
	char kern_name[KERN_NAME_LEN];
};

struct launch_trace {
	uint64_t timestamp_ns;
	uint32_t grid_x, grid_y, grid_z;
	uint32_t block_x, block_y, block_z;
	uint64_t stream;
	char op_name[128];
	char kernel_name[128];
};

static uint64_t launch_read_pos = 0;
static struct launch_trace launch_records[512];
static int launch_count = 0;

static void demangle(const char *mangled, char *out, int outsz)
{
	if (mangled[0] == '_' && mangled[1] == 'Z') {
		int len = 0;
		const char *p = mangled + 2;
		while (*p >= '0' && *p <= '9') {
			len = len * 10 + (*p - '0');
			p++;
		}
		int n = len < outsz - 1 ? len : outsz - 1;
		memcpy(out, p, n);
		out[n] = '\0';
	} else {
		strncpy(out, mangled, outsz - 1);
	}
}

// Per-kernel aggregate stats
#define MAX_KERN 32
struct kern_stats {
	char name[128];
	uint64_t total_duration;
	uint64_t min_duration;
	uint64_t max_duration;
	int count;
	int have_launch;
	uint32_t grid_x, grid_y, grid_z;
	uint32_t block_x, block_y, block_z;
};

static struct kern_stats kstats[MAX_KERN];
static int nkern = 0;

static struct kern_stats *get_kern(const char *name)
{
	for (int i = 0; i < nkern; i++)
		if (strcmp(kstats[i].name, name) == 0)
			return &kstats[i];
	if (nkern >= MAX_KERN)
		return NULL;
	struct kern_stats *k = &kstats[nkern++];
	memset(k, 0, sizeof(*k));
	strncpy(k->name, name, sizeof(k->name) - 1);
	k->min_duration = UINT64_MAX;
	return k;
}

static void reset_kstats(void)
{
	for (int i = 0; i < nkern; i++) {
		kstats[i].total_duration = 0;
		kstats[i].min_duration = UINT64_MAX;
		kstats[i].max_duration = 0;
		kstats[i].count = 0;
		kstats[i].have_launch = 0;
	}
	nkern = 0;
}

static void poll_launch_traces(void)
{
	int (*read_fn)(struct launch_trace *, int, uint64_t *) =
		(int (*)(struct launch_trace *, int, uint64_t *))dlsym(
			RTLD_DEFAULT,
			"bpftime_syscall_server__get_gpu_launch_records");
	if (!read_fn)
		return;

	int n = read_fn(launch_records, 512, &launch_read_pos);
	for (int i = 0; i < n; i++) {
		char dname[128];
		demangle(launch_records[i].kernel_name, dname, sizeof(dname));
		struct kern_stats *k = get_kern(dname);
		if (k && !k->have_launch &&
		    (launch_records[i].grid_x > 0 ||
		     launch_records[i].block_x > 0)) {
			k->grid_x = launch_records[i].grid_x;
			k->grid_y = launch_records[i].grid_y;
			k->grid_z = launch_records[i].grid_z;
			k->block_x = launch_records[i].block_x;
			k->block_y = launch_records[i].block_y;
			k->block_z = launch_records[i].block_z;
			k->have_launch = 1;
		}
	}
	launch_count += n;
}

struct entry {
	uint64_t bid_x, tid_x;
	char kern_name[KERN_NAME_LEN];
	uint64_t ts;
	int used;
};
#define HASH_SIZE 65536
static struct entry enter_map[HASH_SIZE];

static uint32_t hash_key(uint64_t bid, uint64_t tid)
{
	return (uint32_t)((bid * 31 + tid) % HASH_SIZE);
}

static void poll_callback(const void *data, uint64_t size, void *ctx)
{
	uint64_t *total = (uint64_t *)ctx;
	const struct event *e = data;

	if (e->type == 0) {
		uint32_t h = hash_key(e->bid_x, e->tid_x);
		enter_map[h].bid_x = e->bid_x;
		enter_map[h].tid_x = e->tid_x;
		enter_map[h].ts = e->timestamp;
		enter_map[h].used = 1;
		strncpy(enter_map[h].kern_name, e->kern_name, KERN_NAME_LEN - 1);
	} else {
		uint32_t h = hash_key(e->bid_x, e->tid_x);
		if (enter_map[h].used &&
		    enter_map[h].bid_x == e->bid_x &&
		    enter_map[h].tid_x == e->tid_x) {
			uint64_t duration = e->timestamp - enter_map[h].ts;
			enter_map[h].used = 0;
			(*total)++;
			if (!(e->tid_x % 32 == 0))
				return;
			struct kern_stats *k = get_kern(enter_map[h].kern_name);
			if (k) {
				k->count++;
				k->total_duration += duration;
				if (duration < k->min_duration)
					k->min_duration = duration;
				if (duration > k->max_duration)
					k->max_duration = duration;
			}
		}
	}
}

static void print_frame(uint64_t total)
{
	if (nkern == 0 && total == 0)
		return;

	for (int i = 0; i < nkern; i++) {
		struct kern_stats *k = &kstats[i];
		if (k->count == 0)
			continue;
		printf("==== %s <<<(%u,%u,%u),(%u,%u,%u)>>> ====\n",
		       k->name,
		       k->grid_x, k->grid_y, k->grid_z,
		       k->block_x, k->block_y, k->block_z);
		printf("  count=%d  avg=%.0f ns  min=%lu ns  max=%lu ns\n",
		       k->count,
		       k->count ? (double)k->total_duration / k->count : 0,
		       (unsigned long)k->min_duration,
		       (unsigned long)k->max_duration);
	}
	if (total > 0)
		printf("-- total events: %lu --\n\n", total);
}

int main(int argc, char **argv)
{
	struct kerneltiming_bpf *skel;
	int err;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	memset(enter_map, 0, sizeof(enter_map));
	memset(kstats, 0, sizeof(kstats));

	skel = kerneltiming_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = kerneltiming_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton\n");
		goto cleanup;
	}
	err = kerneltiming_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

	int (*poll_fn)(int, void *, void (*)(const void *, uint64_t, void *)) =
		dlsym(RTLD_DEFAULT,
		      "bpftime_syscall_server__poll_gpu_ringbuf_map");
	if (poll_fn == NULL) {
		puts("This example must run under bpftime!");
		goto cleanup;
	}

	int mapfd = bpf_map__fd(skel->maps.rb);
	uint64_t total = 0, frame_total = 0;
	int tick = 0;
	while (!exiting) {
		usleep(100000);
		tick++;
		poll_launch_traces();
		err = poll_fn(mapfd, &total, poll_callback);
		if (err < 0) {
			printf("Poll error: %d\n", err);
			goto cleanup;
		}
		frame_total += total;
		total = 0;
		if (tick % 10 == 0) {
			print_frame(frame_total);
			frame_total = 0;
			reset_kstats();
		}
	}

cleanup:
	kerneltiming_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
