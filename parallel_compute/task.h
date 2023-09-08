#ifndef _TASK_H_
#define _TASK_H_

#define MAX_PARAMS_SIZE_BYTES 0X40

typedef struct split_task_params {
    u32 offset;
    u32 block_size;
} split_task_params ;

typedef void(task_func)(void *params); // function that does the actual computation
typedef void(divide_work_func)(void *params, split_task_params st_params); // divide work for mulatiple workers

typedef struct task {
    void* params;
    task_func *func;
    divide_work_func *divide_work;
    binary_semapore *sem;
    s32 work_size;
    s32 params_size_bytes;
} task;

typdef struct worker_params { // that's what being passed to the worker thread function
    void* params;
    task_func *func;
    divide_work_func *divide_work;
    split_task_params st_params;
    s32 params_size_bytes;
} worker_params;

#endif // _TASK_H_
