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

// Poll and print GPU launch association records (op_name -> kernel_name)
static void poll_launch_traces()
{
	int (*read_fn)(struct launch_trace *, int, uint64_t *) =
		(int (*)(struct launch_trace *, int, uint64_t *))dlsym(
			RTLD_DEFAULT,
			"bpftime_syscall_server__get_gpu_launch_records");
	if (!read_fn)
		return;

	int n = read_fn(launch_records, 512, &launch_read_pos);
	for (int i = 0; i < n; i++) {
		if (launch_records[i].op_name[0] != '\0') {
			printf("[%s]\n", launch_records[i].op_name);
		}
		printf("  %s<<<(%u,%u,%u),(%u,%u,%u)>>>\n",
		       launch_records[i].kernel_name,
		       launch_records[i].grid_x,
		       launch_records[i].grid_y,
		       launch_records[i].grid_z,
		       launch_records[i].block_x,
		       launch_records[i].block_y,
		       launch_records[i].block_z);
	}
	launch_count += n;
}

// Extension of struct event for enter/exit matching
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
			// only print per-warp to avoid noise
			if (e->tid_x % 32 == 0) {
				printf("    %-16s block(%lu,%lu,%lu) thread(%lu,%lu,%lu) duration: %lu ns\n",
				       enter_map[h].kern_name,
				       (unsigned long)e->bid_x,
				       (unsigned long)e->bid_y,
				       (unsigned long)e->bid_z,
				       (unsigned long)e->tid_x,
				       (unsigned long)e->tid_y,
				       (unsigned long)e->tid_z,
				       (unsigned long)duration);
			}
			enter_map[h].used = 0;
			(*total)++;
		}
	}
}

int main(int argc, char **argv)
{
	struct kerneltiming_bpf *skel;
	int err;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	memset(enter_map, 0, sizeof(enter_map));

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
	uint64_t total = 0;
	while (!exiting) {
		sleep(1);
		poll_launch_traces();
		err = poll_fn(mapfd, &total, poll_callback);
		if (err < 0) {
			printf("Poll error: %d\n", err);
			goto cleanup;
		}
		if (total > 0) {
			printf("  Total events: %lu\n", total);
			total = 0;
		}
	}

cleanup:
	kerneltiming_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
