// consumer thread can wait for data, but producer can't be blocked
class producer_consumer {
    public:
        static constexpr u64 guard = u64(-1);
        typedef vector<data> data_buffer;
    private:
        data_buffer _front_buf;
        data_buffer _back_buf;
        data_buffer *_front_buf_ptr;
        data_buffer *_back_buf_ptr;
        volatile data_buffer *_shared_buf_ptr;
        data_buffer *_save_buf_ptr;
        volatile u32		_new_data_ready;
        volatile u32		_stop_consumption;    // consumer is going to run a while loop, and this is the exit condition
        semaphore _sem{/*max_count=*/ 1 };	// binary semaphore
    public:
        producer_consumer ()	:	_front_buf_ptr(&_front_buf), _back_buf_ptr(&_back_buf), _shared_buf_ptr(_back_buf_ptr), _save_buf_ptr(nullptr),
                                        _new_data_ready(0), _stop_consumption(0) {
            create_thread(consume_data_mt, (pvoid)(this));
        }

        ~producer_consumer () {
            atomic_store((volatile u32&)_stop_consumption, 1u);
            atomic_store((volatile u32&)_new_data_ready, 1u);
            _sem.signal();
        }
        // called from the producer thread
        data_buffer &begin_producing()		{
            _save_buf_ptr	= (data_buffer*)(atomic_xchg(((volatile u64*)&_shared_buf_ptr), guard));
            return			*_save_buf_ptr;
        }
        void end_producing() {
            atomic_store((volatile u64&)_shared_buf_ptr, (u64)_save_buf_ptr);
            atomic_store((volatile u32&)_new_data_ready, 1u);
            _sem.signal();
        }
    private:
        // called from the consumer thread
            data_buffer &begin_consuming() {
            wait_for_data	();
            swap			(_front_buf_ptr, _back_buf_ptr);	// doesn't need to be atomic
            return			*_front_buf_ptr;
        }
        void end_consuming() {
            _front_buf_ptr->clear();
        }
        void wait_for_data() {    // atomic_cmpxchg(dst*, xchg, cmp)->old
            while (!atomic_xchg((volatile u32*)&_new_data_ready, 0u) || (guard == atomic_cmpxchg((volatile s64*)&_shared_buf_ptr, (s64)_front_buf_ptr, (s64)_back_buf_ptr))) {
                _sem.wait();	// doesn't wait, if the object is already signaled
            }
        }
    public:
        static void	consume_data_mt(pvoid args);
}						_motion_folder_validator;

    void producer_consumer::consume_data_mt(pvoid_args) {
        producer_consumer &consumer = *(producer_consumer*)args;
        while (!consumer._stop_consumption) {
            data_buffer &data = consumer.begin_consuming();
            // do something with data
            consumer.end_consuming();
        }
    }

    void produce(producer_consumer& producer) {
        data_buffer &data = producer.begin_producing();
        // do something with data
        producer.end_producing();
    }
