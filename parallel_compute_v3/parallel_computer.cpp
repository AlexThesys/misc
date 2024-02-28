#include "parallel_computer.h"

void parallel_computer::initialize() {
	const s32 num_workers		= _min((s32)platform.features._cpucount - 3, MAX_WORKERS); // temp
	_workers_sem_feedback.resize(num_workers, nullptr);

	for (s32 t = 0; t < num_workers; t++) {
		threading::spawn		(-1, worker_func, "parallel_computer", 0, (void*)this);
	}
	while (_thread_counter != num_workers) {
		_master_sem.wait		();
	}
}

void parallel_computer::deinitialize() {
	_stop_workers				= true;
	for (auto& sem : _workers_sem_feedback) {
		sem->signal				();
	}
	parallel_task task			{};
	while (_task_queue.try_pop_task_queue(task)) {
		task.completion_sem->signal();
	}
	while (0 != _thread_counter) {
		_master_sem.wait		();
	}
}

void parallel_computer::worker_func(void *args) {
	threading::semaphore worker_sem{ 1 };

	parallel_computer *worker_params = (parallel_computer*)args;
	parallel_task_queue &task_queue = worker_params->_task_queue;
	const s32 t_id				= threading::inc(&worker_params->_thread_counter) - 1;
	worker_params->_workers_sem_feedback[t_id] = &worker_sem;
	volatile s32 &stop			= worker_params->_stop_workers;
	worker_params->_master_sem.signal();
	while (true) {
		parallel_task task{};
        while (!(stop) && !task_queue.try_pop_task_queue(task)) {
			worker_sem.wait		();
        }
        if (stop) {
			threading::dec(&worker_params->_thread_counter);
			worker_params->_master_sem.signal();
            break;
        }
        // do work
        task.func_wrapper		((void*)task.params, task.st_params);
        // signal requester
        if ((task.num_chunks) == threading::inc(task.chunk_counter)) {
			task.completion_sem->signal();
        }
    }
}

void parallel_computer::launch_task_and_wait(task_func *func, void *args, u32 work_size) {
	u_vector<parallel_task> tasks;
	volatile s32 chunk_counter	= 0;
	threading::semaphore completion_sem;

	assign_workload				(tasks, work_size, &chunk_counter, args, func, &completion_sem);
	post_tasks					(tasks);

	completion_sem.wait			();
}

void parallel_computer::assign_workload(u_vector<parallel_task> &tasks, u32 work_size, volatile s32 *chunk_counter, void *args, task_func *func, threading::semaphore *sem) {
		const u32 workers_size	= _workers_sem_feedback.size();
		const u32 max_threads	= _min(workers_size, work_size);
		const u32 block_size	= (work_size + max_threads - 1) / max_threads;
		s32 tail				= work_size - block_size * max_threads;
		u32 offset_accum		= 0;
		tasks.resize			(max_threads);
		for (u32 i = 0; i < max_threads; i++) {
			tasks[i].st_params.block_size = block_size + ((tail-- > 0) ? 1 : 0);
			tasks[i].st_params.offset = offset_accum;
			offset_accum		+= tasks[i].st_params.block_size;

			tasks[i].num_chunks = max_threads;
			tasks[i].chunk_counter = chunk_counter;
			tasks[i].params		= args;
			tasks[i].func_wrapper = func;
			tasks[i].completion_sem = sem;
        }
}

void parallel_computer::post_tasks(u_vector<parallel_task> &tasks) {
    const s32 num_threads		= (s32)tasks[0].num_chunks;
    for (s32 i = 0; i < num_threads; i++) {
         while (!_task_queue.try_push_task_queue(tasks[i])) {
             rlog				(LOG_WARNING "[ANIM] Not enough space in the parallel tasks queue!");
			 threading::yield	(); // disable this on PS5 ?
         }
    }
    // wake up all the workers - we don't know which ones a currently sleeping
    for (auto &sem : _workers_sem_feedback) { // do we need a lock here?
		sem->signal				();
    }
}

