#ifndef _SCHEDULER_H
#define _SCHEDULER_H

#include "queue.h"

#ifndef NUM_WORKERS
#define NUM_WORKERS 0X8
#endif // NUM_WORKERS

volatile s32 g_stop_scheduler; // = 0
volatile s32 g_stop_workers; // = 0
binary_semapore g_scheduler_sem;
barrier g_scheduler_barrier;
pthread_t g_workers[NUM_WORKERS];
counting_semaphore g_workers_sem;

task_queue g_task_queue;

void init_scheduler() {
    semaphore_init(&g_scheduler_sem);
    barrier_init(&g_scheduler_barrier);
}

void deinit_scheduler() {
    semaphore_deinit(&g_scheduler_sem);
    barrier_deinit(&g_scheduler_barrier);
}

void init_workers(worker_params *params) {
    int result_code = 0;
    for (int i = 0; i < NUM_WORKERS; i++) {
        result_code |= pthread_create(&g_workers[i], NULL, execute_task, (void*)(params+i));
    }
    assert(!result_code);
}

void deinit_workers() {
    int result_code = 0;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        result_code = pthread_join(g_workers[i], NULL);
        assert(!result_code);
    }
}

void* worker_func(void* params) { // (worker_params* params)
    u8 params_copy[MAX_PARAMS_SIZE_BYTES];
    worker_params *p = (worker_params*)params;
    while (TRUE) {
        counting_semaphore_wait(&g_workers_sem);
        if (g_stop_workers)
            break;

        if (!p->st_params.block_size)
            continue;   // go back to sleep

        // make a copy of the func params
        assert(p->params_size_bytes <= MAX_PARAMS_SIZE_BYTES);
        memcpy((void*)params_copy, &p->params, p->params_size_bytes);
        // get this worker's chunk of work
        p->divide_work((void*)params_copy, p->st_params);
        // do work
        p->func((void*)params_copy);
        // signal scheduler
        barrier_signal(&g_scheduler_barrier);
    }

    return NULL;
}

void* scheduler_func(void* params) {
    worker_params params[NUM_WORKERS];
    task new_task;
    counting_semaphore_init(&g_workers_sem, NUM_WORKERS);
    init_workers(params);
    while (!g_stop_scheduler) {
        // wait for the next task(s) to be posted
        while (is_empty_task_queue(&g_task_queue)) {
            g_scheduler_sem.binary_semaphore_wait();
        }
        assert(try_pop_task_queue(&g_task_queue, &new_task)); 
               
        const int work_size = new_task.work_size;
		const int max_threads = _min(THREADCOUNT, work_size);
		const int block_size = work_size / max_threads;
		int tail = work_size - block_size * max_threads;
		int offset_accum = 0;
		for (int i = 0; i < max_threads; i++) {
            worker_params[i].params = new_task.params;
            worker_params[i].task_func = &new_task.func;
            worker_params[i].divide_work = &new_task.divide_work;
			worker_params[i].st_params.block_size = block_size + ((tail-- > 0) ? 1 : 0);
			worker_params[i].st_params.offset = offset_accum;
			offset_accum += worker_params[i].st_params.block_size;
            worker_params[i].params_size_bytes = new_task.params_size_bytes;
		}
		for (int i = max_threads; i < THREADCOUNT; i++)
			worker_params[i].st_params.block_size = 0;
        // signal workers to start computing
        counting_semaphore_signal(&g_workers_sem, max_threads);
        
        // wait until all the workers have finished
        barrier_try_wait(&g_scheduler_barrier);
        
        // signal client that the work is done        
        new_task->sem->binary_semaphore_signal();
    }
    // stop workers
    g_stop_workers = 0;
    deinit_workers();
    counting_semaphore_deinit(&g_workers_sem);

    return NULL;
}

#endif // _SCHEDULER_H
