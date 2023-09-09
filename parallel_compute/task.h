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
    binary_semaphore *sem;
    s32 work_size;
} task;

typedef struct worker_params { // that's what being passed to the worker thread function
    void* params;
    task_func *func_wrapper;
    split_task_params st_params;
} worker_params;

#endif // _TASK_H_
