#include "scheduler.h"

#define EXAMPLE_PARAMS float* a, float* b, float *c, int size

#define EXAMPLE_WORK_SIZE 0X1000

typedef struct example_params {
    float *a;
    float *b;
    float *c;
} example_params;

// the function one wants to compute in parallel
void example_func(EXAMPLE_PARAMS);
// the wrapper for the function to be computed in parallel - the task_func
void example_wrapper_func(void* params, split_task_params st_params);

void generate_example_data(EXAMPLE_PARAMS);

int main() {
    task new_task;
    binary_semapore sem;
    example_params params;
    // test data
    float a[EXAMPLE_WORK_SIZE];
    float b[EXAMPLE_WORK_SIZE];
    float c[EXAMPLE_WORK_SIZE];

    // initialization
    init_task_queue(&g_task_queue);
    init_scheduler();
    binary_semaphore_init(&sem);

    // test run
    generate_example_data(a, b, c, size);
    example_params.a = a;
    example_params.b = b;
    example_params.c = c;
    new_task.params = (void*)&example_params;
    new_task.task_wrapper = &example_wrapper_func;
    new_task.sem = &sem;
    new_task.work_size = EXAMPLE_WORK_SIZE; 
    if (try_push_next_task_queue(&g_task_queue, &new_task)) {
        binary_semaphore_signal(&g_scheduler_sem);
        binary_semaphore_try_wait(&sem);
        sem.do_wait = true;     

        // print results
        for (int i = 0; i < EXAMPLE_WORK_SIZE; i++)
            printf("%.2f\t%.2f\t%.2f\n", a[i], b[i], c[i]);
    }

    // cleanup
    deinit_scheduler();
    binary_semaphore_deinit(&sem);

    return EXIT_SUCCESS;
}

void example_task(EXAMPLE_PARAMS) {
    for (int i = 0; i < size; i++)
        c[i] = a[i] + b[i];
}

void example_wrapper_func(void* params, split_task_params st_params) {
    example_params *p = (example_params*)params;
    //  adjust the work chunk using the offset and block_size
    float *a = p->a + st_params.offset;
    float *b = p->b + st_params.offset;
    float *c = p->c + st_params.offset;

    example_task(a, b, c, st_params.block_size);
}

void generate_example_data(EXAMPLE_PARAMS) {
    for (int i = 0; i < size; i++) {
        a[i] = (float)rand();
        b[i] = (float)rand();
    }
    memset(c, 0, sizeof(float) * size);
}
