#ifndef _QUEUE_H_
#define _QUEUE_H_

#include "common.h"
#include "spinlock.h"
#include "semaphore.h"
#include "task.h"

#define MAX_QUEUE_LENGHT 0x20

typedef struct task_queue {
    task tasks[MAX_QUEUE_LENGHT];
    volatile s32 lock;
    u16 read_idx;
    u16 write_idx;
    u8 read_round;
    u8 write_round;
} task_queue;

void init_task_queue(task_queue* tasks_q) {
    tasks_q->lock = 0;
    tasks_q->write_idx = 0;
    tasks_q->read_idx = 0;
    tasks_q->write_round = 0;
    tasks_q->read_round = 0;
}

BOOL is_empty_task_queue(task_queue* tasks_q) {
    spinlock_lock(&tasks_q->lock);
    BOOL empty = (tasks_q->write_idx == tasks_q->read_idx 
        && tasks_q->write_round == tasks_q->read_round);
    spinlock_unlock(&tasks_q->lock);
    return empty;
}

BOOL try_pop_task_queue(task_queue* tasks_q, task* task) {
    spinlock_lock(&tasks_q->lock);
    if (tasks_q->write_idx == tasks_q->read_idx 
        && tasks_q->write_round == tasks_q->read_round) {
        spinlock_unlock(&tasks_q->lock);
        puts("Task queue is empty.");
        return FALSE;
    }
    memcpy(task, &tasks_q->tasks[tasks_q->read_idx], sizeof(*task));
    tasks_q->read_round ^= (tasks_q->read_idx == (MAX_QUEUE_LENGHT - 1));
    tasks_q->read_idx = (tasks_q->read_idx + 1) & (MAX_QUEUE_LENGHT - 1);
    puts("Pop task from task queue");
    spinlock_unlock(&tasks_q->lock);
    return TRUE;
}

BOOL try_push_next_task_queue(task_queue* tasks_q, task* task) {
    spinlock_lock(&tasks_q->lock);
    if (tasks_q->write_idx == tasks_q->read_idx
            && tasks_q->write_round != tasks_q->read_round) {
        spinlock_unlock(&tasks_q->lock);
        puts("Task queue is full.");
        return FALSE;
    }
    memcpy(&tasks_q->tasks[tasks_q->write_idx], task, sizeof(*task));
    tasks_q->write_round ^= (tasks_q->write_idx == (MAX_QUEUE_LENGHT - 1));
    tasks_q->write_idx = (tasks_q->write_idx + 1) & (MAX_QUEUE_LENGHT - 1);
    puts("Pushed task to task queue");
    spinlock_unlock(&tasks_q->lock);
    return TRUE;
}    
#endif // _QUEUE_H_
