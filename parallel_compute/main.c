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

int main() {
    task new_task;
   new_task.work_size = EXAMPLE_WORK_SIZE; 

    return EXIT_SUCCESS;
}

void example_task(EXAMPLE_PARAMS) {

}

void example_wrapper_func(void* params, split_task_params st_params) {
    example_params *p = (example_params*)params;
    //  adjust the work chunk using the offset and block_size
    float *a = p->a + st_params.offset;
    float *b = p->b + st_params.offset;
    float *c = p->c + st_params.offset;

    example_task(a, b, c, st_params.block_size);
}
