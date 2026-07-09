#ifndef _NV_ATTACH_LAUNCH_TRACE_HPP
#define _NV_ATTACH_LAUNCH_TRACE_HPP

#include <cstdint>
#include <cstring>
#include <pthread.h>

#define GPU_LAUNCH_TRACE_SHM_NAME "/bpftime_gpu_launch_trace"
#define GPU_LAUNCH_TRACE_MAX_RECORDS 8192

#ifdef __cplusplus
extern "C" {
#endif

struct gpu_launch_trace_record {
	uint64_t timestamp_ns;
	uint32_t grid_x, grid_y, grid_z;
	uint32_t block_x, block_y, block_z;
	uint64_t stream;
	char op_name[128];
	char kernel_name[128];
};

struct gpu_launch_trace_header {
	pthread_mutex_t mutex;
	uint64_t write_pos;
	uint64_t records_written;
};

// Write a launch record to the shared memory trace buffer.
// Called from nv_attach_impl's cudaLaunchKernel replacement.
int gpu_launch_trace_write(const struct gpu_launch_trace_record *rec);

// Read and consume all records since last read.
// out_records: pre-allocated buffer, at least max_count entries.
// last_read_pos: pointer to the position last read (set to 0 initially).
// Returns number of records read.
int gpu_launch_trace_read(struct gpu_launch_trace_record *out_records,
			  int max_count, uint64_t *last_read_pos);

#ifdef __cplusplus
}
#endif

#endif // _NV_ATTACH_LAUNCH_TRACE_HPP
