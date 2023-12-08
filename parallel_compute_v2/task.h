#ifndef _TASK_H_
#define _TASK_H_

struct binary_semaphore; // FWD

#define MAX_PARAMS_SIZE_BYTES 0X40

typedef struct split_task_params {
    u32 offset;
    u32 block_size;
} split_task_params;

typedef void(task_func)(void *params, split_task_params st_params); // function that does the actual computation

typedef struct task {
    void* params;
    task_func *func_wrapper;
    binary_semaphore *completion_sem;
    split_task_params st_params;

    u32 num_chunks;
    volatile u32 *chunk_counter; // shared between worker threads
} task;

#endif // _TASK_H_
