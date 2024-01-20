#include "parallel_computer.h"

parallel_computer::parallel_computer() : _stop_workers(false), _thread_counter(0) {
	const int32_t num_workers = _min((int32_t)platform.features._cpucount - 3, MAX_WORKERS); // temp
	_workers_sem_feedback.resize(num_workers, nullptr);

	for (int32_t t = 0; t < num_workers; t++) {
		spawn_thread(worker_func, (void*)this);
	}
}

parallel_computer::~parallel_computer() {
	_stop_workers = true;
	for (auto &sem : _workers_sem_feedback) {
		sem->signal();
	}
}

void parallel_computer::worker_func(void *args) {
	semaphore worker_sem{1};

	parallel_computer *worker_params = (parallel_computer*)args;
	parallel_task_queue &task_queue = worker_params->_task_queue;
	const int32_t t_id = atomic_inc(&worker_params->_thread_counter) - 1;
	worker_params->_workers_sem_feedback[t_id] = &worker_sem;
	volatile int32_t &stop = worker_params->_stop_workers;

	while (true) {
		parallel_task task{};
        while (!(stop) && !task_queue.try_pop_task_queue(task)) {
			worker_sem.wait();
        }
        if (stop) {
            while (task_queue.try_pop_task_queue(task)) {
				task.completion_sem->signal();
            }
            break;
        }
        // do work
        task.func_wrapper((void*)task.params, task.st_params);
        // signal requester
        if ((task.num_chunks) == atomic_inc(task.chunk_counter)) {
			task.completion_sem->signal();
        }
    }
}

void parallel_computer::launch_task_and_wait(task_func *func, void *args, u32 work_size) {
	vector<parallel_task> tasks;
	volatile int32_t chunk_counter = 0;
	threading::semaphore completion_sem;

	assign_workload(tasks, work_size, &chunk_counter, args, func, &completion_sem);
	post_tasks(tasks);

	completion_sem.wait();
}

void parallel_computer::assign_workload(vector<parallel_task> &tasks, uint32_t work_size, volatile int32_t *chunk_counter, void *args, task_func *func, semaphore *sem) {
		const uint32_t workers_size	= _workers_sem_feedback.size();
		const uint32_t max_threads = _min(workers_size, work_size);
		const uint32_t block_size = (work_size + max_threads - 1) / max_threads;
		int32_t tail = work_size - block_size * max_threads;
		uint32_t offset_accum = 0;
		tasks.resize			(max_threads);
		for (uint32_t i = 0; i < max_threads; i++) {
			tasks[i].st_params.block_size = block_size + ((tail-- > 0) ? 1 : 0);
			tasks[i].st_params.offset = offset_accum;
			offset_accum += tasks[i].st_params.block_size;

			tasks[i].num_chunks = max_threads;
			tasks[i].chunk_counter = chunk_counter;
			tasks[i].params	= args;
			tasks[i].func_wrapper = func;
			tasks[i].completion_sem = sem;
        }
}

void parallel_computer::post_tasks(vector<parallel_task> &tasks) {
    const int32_t num_threads = (int32_t)tasks[0].num_chunks;
    for (int32_t i = 0; i < num_threads; i++) {
         while (!_task_queue.try_push_task_queue(tasks[i])) {
             puts("Parallel tasks queue is full! Waiting...");
			 threading::yield(); // disable this on PS5 ?
         }
    }
    // wake up all the workers - we don't know which ones a currently sleeping
    for (auto &sem : _workers_sem_feedback) { // do we need a lock here?
		sem->signal();
    }
}