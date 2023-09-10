#ifndef _BARRIER_H_
#define _BARRIER_H_

#include <pthread.h>
#include "common.h"

typedef struct barrier {
    pthread_cond_t cond;
    pthread_mutex_t wait_mtx;
    volatile s32 counter;
    volatile s32 max_count;
} barrier;

void barrier_init(barrier* sem) {
    pthread_cond_init(&sem->cond, NULL);
    pthread_mutex_init(&sem->wait_mtx, NULL);
    sem->counter = 0;
    sem->max_count = -1;
}

void barrier_deinit(barrier* sem) {
    pthread_cond_destroy(&sem->cond);
    pthread_mutex_destroy(&sem->wait_mtx);
    sem->counter = 0;
    sem->max_count = -1;
}

void barrier_wait(barrier* sem) {
    pthread_mutex_lock(&sem->wait_mtx);
    if (++sem->counter == sem->max_count) {
        pthread_cond_broadcast(&sem->cond);
    } else {
        while (sem->counter != sem->max_count) {
            pthread_cond_wait(&sem->cond, &sem->wait_mtx);
        }
    }
    pthread_mutex_unlock(&sem->wait_mtx);
}

void barrier_try_wait(barrier* sem, s32 count) {
    pthread_mutex_lock(&sem->wait_mtx);
    sem->max_count = count;
    while (sem->counter != sem->max_count) {
        pthread_cond_wait(&sem->cond, &sem->wait_mtx);
    }
    sem->counter = 0;
    pthread_mutex_unlock(&sem->wait_mtx);
}

void barrier_signal(barrier* sem) {
    pthread_mutex_lock(&sem->wait_mtx);
    sem->counter++; // % max_count ?
    pthread_cond_broadcast(&sem->cond);
    pthread_mutex_unlock(&sem->wait_mtx);

}

void barrier_reset(barrier* sem, s32 max_count) {
    pthread_mutex_lock(&sem->wait_mtx);
    sem->counter = 0;
    sem->max_count = max_count;
    pthread_mutex_unlock(&sem->wait_mtx);
}
#endif // _BARRIER_H_
