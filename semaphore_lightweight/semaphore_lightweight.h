class semaphore_lw {
	volatile int32_t _counter;
	int32_t _max_count;
public:
	semaphore_lw();
	semaphore_lw(int32_t max_count);
	~semaphore_lw() = default;
	bool signal()	;
	bool signal(int32_t cnt);
	bool wait();
	bool trywait();
};
	
semaphore_lw::semaphore_lw()	{
	_max_count = 0x7FFFFFFF;
	atomic_store(_counter, 0);
}

semaphore_lw::semaphore_lw(int32_t max_count)	{
	assert(max_count > 0);
	_max_count = max_count;
	atomic_store(&_counter, 0);
}

bool semaphore_lw::signal()	{
	int32_t old_counter, new_counter;
	new_counter	= atomic_load(&_counter);
	do {
		old_counter	= new_counter;
		if (old_counter < _max_count) {
			new_counter	= old_counter + 1;
		}
		new_counter	= atomic_cmpxchg(&_counter, new_counter, old_counter); // val, new, cmp
	} while (new_counter != old_counter);
	return true;
}

bool semaphore_lw::signal(int32_t cnt) {
	int32_t old_counter, new_counter;
	new_counter					= atomic_load(&_counter);
	do {
		old_counter = new_counter;
		new_counter += cnt;
		if (new_counter > _max_count) {
			new_counter = _max_count;
		}
		new_counter = atomic_cmpxchg(&_counter, new_counter, old_counter); // val, new, cmp
	} while (new_counter != old_counter);
	return true;
}

bool semaphore_lw::wait() {
	uint_t8 wait = 1;
	while (1) {
		if (trywait()) {
			return true;
		}
		wait += wait;
		if (0 == wait) {
			yield_thread(0);
			wait = 1;
		} else {
			for (uint8_t i = 0; i < wait; ++i) {
				pause();
			}
		}
	}
	return false;
}

bool semaphore_lw::trywait() {
	bool result;
	int32_t old_counter, new_counter;
	new_counter = atomic_load(&_counter);
	do {
		result = false;
		old_counter = new_counter;
		if (old_counter > 0) {
			new_counter = old_counter - 1;
			result = true;
		} else {
			break; // design choice to have fewer cmpxchg instructions
		}
		new_counter = atomic_cmpxchg(&_counter, new_counter, old_counter); // val, new, cmp
	} while	(new_counter != old_counter);
	return result;
}