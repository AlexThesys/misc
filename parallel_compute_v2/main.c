#include <xmmintrin.h>
#include "workers.h"

#define EXAMPLE_PARAMS float* a, float* b, float *c, int size

#define EXAMPLE_WORK_SIZE 0X90

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

#define NUM_TESTS 2

#include <unistd.h>
int main() {
    task new_tasks[NUM_WORKERS];
    example_params params;
    volatile u32 chunk_counter = 0;
    // test data
    float a[EXAMPLE_WORK_SIZE];
    float b[EXAMPLE_WORK_SIZE];
    float c[EXAMPLE_WORK_SIZE];

    // initialization
    init_task_queue(&g_task_queue);
    init_workers(&g_worker_params);
    //sleep(1);

    // test run
    generate_example_data(a, b, c, EXAMPLE_WORK_SIZE);
    params.a = a;
    params.b = b;
    params.c = c;

    for (int j = 0; j < NUM_TESTS; j++) {
        binary_semaphore sem;
        binary_semaphore_init(&sem);

        // assign work to workers
        assign_workload(new_tasks, EXAMPLE_WORK_SIZE, &chunk_counter, &params, &example_wrapper_func, &sem);
        // post tasks to workers
        post_tasks(new_tasks);

        // wait for completion
        binary_semaphore_try_wait(&sem);

        // print results
        for (int i = 0; i < EXAMPLE_WORK_SIZE; i++) {
            printf("%d: %.2f\t%.2f\t%.2f\n", i, a[i], b[i], c[i]);
        }
        puts("\n==================================\n");

        binary_semaphore_deinit(&sem);
    }
    // cleanup
    deinit_workers();

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
