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

struct event {
	uint64_t type;       // 0=enter, 1=exit
	uint64_t bid_x, bid_y, bid_z;
	uint64_t tid_x, tid_y, tid_z;
	uint64_t timestamp;
};

// Simple hash table for matching enter/exit: key = (bid_x, tid_x)
#define HASH_SIZE 65536
struct entry {
	uint64_t bid_x, tid_x;
	uint64_t ts;
	int used;
};
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
		// Enter: store timestamp
		uint32_t h = hash_key(e->bid_x, e->tid_x);
		enter_map[h].bid_x = e->bid_x;
		enter_map[h].tid_x = e->tid_x;
		enter_map[h].ts = e->timestamp;
		enter_map[h].used = 1;
	} else {
		// Exit: lookup and compute duration
		uint32_t h = hash_key(e->bid_x, e->tid_x);
		if (enter_map[h].used &&
		    enter_map[h].bid_x == e->bid_x &&
		    enter_map[h].tid_x == e->tid_x) {
			uint64_t duration = e->timestamp - enter_map[h].ts;
			printf("block(%lu,%lu,%lu) thread(%lu,%lu,%lu) duration: %lu ns\n",
			       e->bid_x, e->bid_y, e->bid_z,
			       e->tid_x, e->tid_y, e->tid_z,
			       duration);
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
		err = poll_fn(mapfd, &total, poll_callback);
		if (err < 0) {
			printf("Poll error: %d\n", err);
			goto cleanup;
		}
		if (total > 0) {
			printf("Total events: %lu\n", total);
			total = 0;
		}
	}

cleanup:
	kerneltiming_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
