#pragma once

struct split_task_params {
	uint32_t	offset;
	uint32_t	block_size;
};

typedef void(task_func)(void *args, split_task_params st_args); // function that does the actual computation

struct parallel_task {
	void *params;
	task_func *func_wrapper;
	semaphore *completion_sem; // binary semaphore
	volatile int32_t *chunk_counter; // shared between worker threads
	split_task_params st_params;
	int32_t	num_chunks;

public:
	parallel_task() = default;
};

template <size_t size_log2> // 32 elements
class parallel_task_queue {
	circular_buffer<parallel_task, size_log2> _buffer;
	spin_lock _lock;
public:
	bool try_push_task_queue(const parallel_task &task);
	bool try_pop_task_queue(parallel_task &task);
};

inline bool	parallel_task_queue::try_push_task_queue(const parallel_task &task) {
	scoped_spin_lock _(_lock);
	return _buffer.write_tail(task);
}

inline bool	parallel_task_queue::try_pop_task_queue(parallel_task &task) {
	scoped_spin_lock _(_lock);
	return _buffer.read(task);
}

class parallel_computer {
	static constexpr int32_t MAX_WORKERS = 4; // 5

	svector<semaphore*, MAX_WORKERS> _workers_sem_feedback;
	parallel_task_queue<5>	_task_queue; // 32 elements
	volatile bool _stop_workers;
	volatile int32_t _thread_counter;
public:
	parallel_comuter();
	~parallel_compter();
	void launch_task_and_wait(task_func *func, void *args, uint32_t work_size);
private:
	void assign_workload(vector<parallel_task> &tasks, uint32_t work_size, volatile s32 *chunk_counter, void *args, task_func *func, semaphore *sem);
	void post_tasks(vector<parallel_task> &tasks);

	static void worker_func(void *args);
};
