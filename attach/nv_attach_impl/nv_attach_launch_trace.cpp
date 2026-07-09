#include "nv_attach_launch_trace.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>

static int g_trace_fd = -1;

static struct gpu_launch_trace_header *g_trace_header = nullptr;
static struct gpu_launch_trace_record *g_trace_records = nullptr;

static void init_trace_if_needed()
{
	if (g_trace_header)
		return;

	g_trace_fd = shm_open(GPU_LAUNCH_TRACE_SHM_NAME,
			       O_CREAT | O_RDWR, 0666);
	if (g_trace_fd < 0) {
		perror("gpu_launch_trace: shm_open failed");
		return;
	}

	size_t total_size =
		sizeof(struct gpu_launch_trace_header) +
		sizeof(struct gpu_launch_trace_record) * GPU_LAUNCH_TRACE_MAX_RECORDS;

	if (ftruncate(g_trace_fd, total_size) < 0) {
		perror("gpu_launch_trace: ftruncate failed");
		close(g_trace_fd);
		g_trace_fd = -1;
		return;
	}

	void *ptr = mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
			 MAP_SHARED, g_trace_fd, 0);
	if (ptr == MAP_FAILED) {
		perror("gpu_launch_trace: mmap failed");
		close(g_trace_fd);
		g_trace_fd = -1;
		return;
	}

	g_trace_header = (struct gpu_launch_trace_header *)ptr;
	g_trace_records = (struct gpu_launch_trace_record *)(g_trace_header + 1);

	pthread_mutexattr_t attr;
	pthread_mutexattr_init(&attr);
	pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
	pthread_mutex_init(&g_trace_header->mutex, &attr);
	pthread_mutexattr_destroy(&attr);

	g_trace_header->write_pos = 0;
	g_trace_header->records_written = 0;
}

int gpu_launch_trace_write(const struct gpu_launch_trace_record *rec)
{
	init_trace_if_needed();
	if (!g_trace_header || !g_trace_records)
		return -1;

	pthread_mutex_lock(&g_trace_header->mutex);
	uint64_t pos = g_trace_header->write_pos % GPU_LAUNCH_TRACE_MAX_RECORDS;
	memcpy(&g_trace_records[pos], rec, sizeof(*rec));
	g_trace_header->write_pos++;
	pthread_mutex_unlock(&g_trace_header->mutex);
	return 0;
}

int gpu_launch_trace_read(struct gpu_launch_trace_record *out_records,
			  int max_count, uint64_t *last_read_pos)
{
	init_trace_if_needed();
	if (!g_trace_header || !g_trace_records)
		return -1;

	pthread_mutex_lock(&g_trace_header->mutex);
	uint64_t total = g_trace_header->write_pos;
	uint64_t start = (*last_read_pos > total) ? total : *last_read_pos;
	uint64_t available = total - start;
	int count = (int)(available < (uint64_t)max_count ? available
						   : (uint64_t)max_count);

	for (int i = 0; i < count; i++) {
		uint64_t pos = (start + i) % GPU_LAUNCH_TRACE_MAX_RECORDS;
		memcpy(&out_records[i], &g_trace_records[pos],
		       sizeof(*out_records));
	}

	*last_read_pos = start + count;
	pthread_mutex_unlock(&g_trace_header->mutex);
	return count;
}
