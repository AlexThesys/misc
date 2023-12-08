#ifndef _WORKERS_H_
#define _WORKERS_H_

#include "queue.h"

#ifndef NUM_WORKERS
#define NUM_WORKERS 0X4
#endif // NUM_WORKERS

#define atomic_incr(x) __sync_fetch_and_add(x, 1)

volatile s32 g_stop_workers; // = 0
pthread_t g_workers[NUM_WORKERS];
binary_semaphore g_workers_sem[NUM_WORKERS];

task_queue g_task_queue;

void* worker_func(void* params);

typedef struct worker_params {
    task_queue *task_queue;
    binary_semaphore *workers_sem;
    volatile const s32 *stop;
} worker_params;

worker_params g_worker_params[NUM_WORKERS];

void init_workers(worker_params params[NUM_WORKERS]) {
    int result_code = 0;
    g_stop_workers = FALSE;
    for (int i = 0; i < NUM_WORKERS; i++) {
        binary_semaphore_init(g_workers_sem+i);
        params[i].task_queue = &g_task_queue;
        params[i].workers_sem = g_workers_sem + i;
        params[i].stop = &g_stop_workers;
        result_code |= pthread_create(&g_workers[i], NULL, worker_func, (void*)(params+i));
    }
    assert(!result_code);
}

void deinit_workers() {
    int result_code = 0;
    g_stop_workers = TRUE;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        binary_semaphore_signal(g_workers_sem+i);
    }
    for (int i = 0; i < NUM_WORKERS; ++i) {
        result_code |= pthread_join(g_workers[i], NULL);
        assert(!result_code);
        binary_semaphore_deinit(g_workers_sem+i);
    }
}

void* worker_func(void* params) {
    worker_params *wp = (worker_params*)params;
    task_queue *tq = wp->task_queue;
    binary_semaphore *w_sem = wp->workers_sem;
    volatile const s32 *stop = wp->stop;
    printf("%x\n", *stop);
    while (TRUE) {
        task t;
        printf("%x\n", *stop);
        while (!(*stop) && !try_pop_task_queue(tq, &t)) {
            binary_semaphore_try_wait(w_sem);
            w_sem->do_wait = TRUE;
        }
        if (*stop) {
            puts("Stopping workers");
            if (try_pop_task_queue(tq, &t)) {
                binary_semaphore_signal(t.completion_sem);
            }
            break;
        }
        // do work
        t.func_wrapper((void*)t.params, t.st_params);
        // signal requester
        if ((t.num_chunks - 1) == atomic_incr(t.chunk_counter)) { // <=
            binary_semaphore_signal(t.completion_sem);
        }
    }
    return NULL;
}

void assign_workload(task params[NUM_WORKERS], int work_size, volatile u32 *chunk_counter, void *args, task_func *func, binary_semaphore *sem) {
        assert(work_size > 0);
        *chunk_counter = 0;

		const int max_threads = _min(NUM_WORKERS, work_size);
		const int block_size = work_size / max_threads;
		int tail = work_size - block_size * max_threads;
		int offset_accum = 0;
		for (int i = 0; i < max_threads; i++) {
			params[i].st_params.block_size = block_size + ((tail-- > 0) ? 1 : 0);
			params[i].st_params.offset = offset_accum;
			offset_accum += params[i].st_params.block_size;

            params[i].num_chunks = max_threads;
            params[i].chunk_counter = chunk_counter;
            params[i].params = args;
            params[i].func_wrapper = func;
            params[i].completion_sem = sem;
        }
}

void post_tasks(task tasks[NUM_WORKERS]) {
    const int num_threads = (int)tasks[0].num_chunks;
    for (int i = 0; i < num_threads; i++) {
         while (!try_push_task_queue(&g_task_queue, (tasks+i))) {
             puts("Not enough space in the queue!");
            sched_yield(); // wait for the queue to have room
         }
    }
    // wake up all the workers - we don't know which ones a currently sleeping
    for (int i = 0; i < NUM_WORKERS; i++) {
        binary_semaphore_signal(g_workers_sem + i);
    }
}

#endif // _WORKERS_H_
