#pragma once

struct split_task_params {
	u32					offset;
	u32					block_size;
};

typedef void(task_func)(void *args, split_task_params st_args); // function that does the actual computation

struct parallel_task {
	void				*params;
	task_func			*func_wrapper;
	threading::semaphore *completion_sem;
	volatile s32		*chunk_counter; // shared between worker threads
	split_task_params	st_params;
	s32					num_chunks;

public:
						parallel_task		() = default;
};

class parallel_task_queue {
	circular_buffer<parallel_task, 5> _buffer;
	threading::spin_lock _lock;
public:
	inline	BOOL		try_push_task_queue	(const parallel_task &task);
	inline	BOOL		try_pop_task_queue	(parallel_task &task);
};

inline BOOL	parallel_task_queue::try_push_task_queue(const parallel_task &task) {
	thread::scoped_spin_lock _(_lock);
	return				_buffer.write_tail(task);
}

inline BOOL	parallel_task_queue::try_pop_task_queue(parallel_task &task) {
	thread::scoped_spin_lock _(_lock);
	return				_buffer.read(task);
}

class parallel_computer {
	static constexpr s32 MAX_WORKERS = 4; // 5

	u_svector<thread::semaphore*, MAX_WORKERS> _workers_sem_feedback;
	parallel_task_queue	_task_queue;
	volatile BOOL		_stop_workers;
	volatile s32		_thread_counter;
	thread::semaphore _master_sem{ 1 }; // binary semaphore
public:
						parallel_computer	() : _stop_workers(false), _thread_counter(0) {}

			void		initialize			();
			void		deinitialize		();
			void		launch_task_and_wait(task_func *func, void *args, u32 work_size);
private:
			void		assign_workload		(vector<parallel_task> &tasks, u32 work_size, volatile s32 *chunk_counter, void *args, task_func *func, thread::semaphore *sem);
			void		post_tasks			(vector<parallel_task> &tasks);

	static	void		worker_func			(void *args);
};
