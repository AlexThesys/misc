#ifndef _SCHEDULER_H
#define _SCHEDULER_H

#include "queue.h"
#include "barrier.h"

#ifndef NUM_WORKERS
#define NUM_WORKERS 0X4
#endif // NUM_WORKERS

volatile s32 g_stop_scheduler; // = 0
volatile s32 g_stop_workers; // = 0
binary_semaphore g_scheduler_sem;
barrier g_scheduler_barrier;
pthread_t g_scheduler_thread;
pthread_t g_workers[NUM_WORKERS];
counting_semaphore g_workers_sem;
barrier g_workers_barrier;

task_queue g_task_queue;

void* scheduler_func(void* params);
void* worker_func(void* params);

void init_scheduler() {
    int result_code;
    g_stop_scheduler = FALSE;
    binary_semaphore_init(&g_scheduler_sem);
    barrier_init(&g_scheduler_barrier);
    result_code = pthread_create(&g_scheduler_thread, NULL, scheduler_func, NULL);
    assert(!result_code);
}

void deinit_scheduler() {
    int result_code;
    g_stop_scheduler = TRUE;
    binary_semaphore_signal(&g_scheduler_sem);
    result_code = pthread_join(g_scheduler_thread, NULL);
    assert(!result_code);
    binary_semaphore_deinit(&g_scheduler_sem);
    barrier_deinit(&g_scheduler_barrier);
}

void init_workers(worker_params *params) {
    int result_code = 0;
    counting_semaphore_init(&g_workers_sem, NUM_WORKERS);
    barrier_init(&g_workers_barrier);
    g_stop_workers = FALSE;
    for (int i = 0; i < NUM_WORKERS; i++) {
        result_code |= pthread_create(&g_workers[i], NULL, worker_func, (void*)(params+i));
    }
    assert(!result_code);
}

void deinit_workers() {
    int result_code = 0;
    g_stop_workers = TRUE;
    counting_semaphore_signal_all(&g_workers_sem);
    for (int i = 0; i < NUM_WORKERS; ++i) {
        result_code |= pthread_join(g_workers[i], NULL);
        assert(!result_code);
    }
    barrier_deinit(&g_workers_barrier);
    counting_semaphore_deinit(&g_workers_sem);
}

void* worker_func(void* params) { // (worker_params* params)
    worker_params *p = (worker_params*)params;
    while (TRUE) {
        counting_semaphore_try_wait(&g_workers_sem);
        if (g_stop_workers)
            break;

        // do work
        p->func_wrapper((void*)p->params, p->st_params);
        // wait for the others
        barrier_wait(&g_workers_barrier);
        // signal scheduler
        barrier_signal(&g_scheduler_barrier);
    }

    return NULL;
}

void* scheduler_func(void*) {
    worker_params params[NUM_WORKERS];
    task new_task;
    init_workers(params);
    while (TRUE) {
        // wait for the next task(s) to be posted
        while (is_empty_task_queue(&g_task_queue)) {
            binary_semaphore_try_wait(&g_scheduler_sem);
            if (g_stop_scheduler)
                goto exit;
            g_scheduler_sem.do_wait = TRUE;
        }
        assert(try_pop_task_queue(&g_task_queue, &new_task)); 
               
        const int work_size = new_task.work_size;
		const int max_threads = _min(NUM_WORKERS, work_size);
		const int block_size = work_size / max_threads;
		int tail = work_size - block_size * max_threads;
		int offset_accum = 0;
		for (int i = 0; i < max_threads; i++) {
            params[i].params = new_task.params;
            params[i].func_wrapper = new_task.func_wrapper;
			params[i].st_params.block_size = block_size + ((tail-- > 0) ? 1 : 0);
			params[i].st_params.offset = offset_accum;
			offset_accum += params[i].st_params.block_size;
		}
        // signal workers to start computing
        barrier_reset(&g_workers_barrier, max_threads);
        counting_semaphore_signal(&g_workers_sem, max_threads);
        
        // wait until all the workers have finished
        barrier_try_wait(&g_scheduler_barrier, max_threads);
        
        // signal client that the work is done        
        binary_semaphore_signal(new_task.sem);
    }
exit:
    // stop workers
    deinit_workers();

    return NULL;
}

#endif // _SCHEDULER_H
