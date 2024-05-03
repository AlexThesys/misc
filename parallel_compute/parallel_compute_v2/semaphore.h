#ifndef _SEMAPHORE_H
#define _SEMAPHORE_H_

#include <pthread.h>

#include "common.h"

typedef struct binary_semaphore {
    volatile BOOL do_wait;
    pthread_cond_t cond;
    pthread_mutex_t wait_mtx;
} binary_semaphore;

void binary_semaphore_init(binary_semaphore* sem) {
    sem->do_wait = TRUE;
    pthread_cond_init(&sem->cond, NULL);
    pthread_mutex_init(&sem->wait_mtx, NULL);
}

void binary_semaphore_deinit(binary_semaphore* sem) {
    pthread_cond_destroy(&sem->cond);
    pthread_mutex_destroy(&sem->wait_mtx);
}

void binary_semaphore_try_wait(binary_semaphore* sem) {
    pthread_mutex_lock(&sem->wait_mtx);
    while (sem->do_wait) {
        pthread_cond_wait(&sem->cond, &sem->wait_mtx);
    }
    pthread_mutex_unlock(&sem->wait_mtx);
}

void binary_semaphore_wait(binary_semaphore* sem) {
    pthread_mutex_lock(&sem->wait_mtx);
    sem->do_wait = TRUE;
    while (sem->do_wait) {
        pthread_cond_wait(&sem->cond, &sem->wait_mtx);
    }
    pthread_mutex_unlock(&sem->wait_mtx);
}

void binary_semaphore_signal(binary_semaphore* sem) {
    pthread_mutex_lock(&sem->wait_mtx);
    sem->do_wait = FALSE;
    pthread_cond_signal(&sem->cond);
    pthread_mutex_unlock(&sem->wait_mtx);

}

void binary_semaphore_signal_all(binary_semaphore* sem) {
    pthread_mutex_lock(&sem->wait_mtx);
    sem->do_wait = FALSE;
    pthread_cond_broadcast(&sem->cond);
    pthread_mutex_unlock(&sem->wait_mtx);
}

/////////////////////////////////////////////////////////////////

typedef struct counting_semaphore_v2 {
    pthread_cond_t cond;
    pthread_mutex_t wait_mtx;
    volatile s32 counter;
    s32 max_count;
} counting_semaphore_v2;

void counting_semaphore_init(counting_semaphore_v2* sem, s32 max_cnt) {
    sem->counter = 0;
    pthread_cond_init(&sem->cond, NULL);
    pthread_mutex_init(&sem->wait_mtx, NULL);
    sem->max_count = max_cnt;
}

void counting_semaphore_deinit(counting_semaphore_v2* sem) {
    sem->counter = 0;
    pthread_cond_destroy(&sem->cond);
    pthread_mutex_destroy(&sem->wait_mtx);
}

void counting_semaphore_try_wait(counting_semaphore_v2* sem) {
    pthread_mutex_lock(&sem->wait_mtx);
    while (sem->counter < sem->max_count) {
        pthread_cond_wait(&sem->cond, &sem->wait_mtx);
    }
    pthread_mutex_unlock(&sem->wait_mtx);
}

void counting_semaphore_signal_one(counting_semaphore_v2* sem) {
    pthread_mutex_lock(&sem->wait_mtx);
    sem->counter++;
    //pthread_cond_signal(&sem->cond);
    pthread_cond_broadcast(&sem->cond);
    pthread_mutex_unlock(&sem->wait_mtx);
}

void counting_semaphore_reset(counting_semaphore_v2* sem, s32 max_count) {
    pthread_mutex_lock(&sem->wait_mtx);
    sem->max_count = max_count;
    sem->counter = 0;
    pthread_mutex_unlock(&sem->wait_mtx);
}
#endif // _SEMAPHORE_H_
