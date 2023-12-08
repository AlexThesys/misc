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
    u16 size;
} task_queue;

void init_task_queue(task_queue* tasks_q) {
    tasks_q->lock = 0;
    tasks_q->write_idx = 0;
    tasks_q->read_idx = 0;
    tasks_q->size = 0;
}

inline BOOL is_empty_task_queue(task_queue* tasks_q) {
    return (tasks_q->size == 0);
}

inline BOOL is_full_task_queue(task_queue* tasks_q) {
    return (tasks_q->size == MAX_QUEUE_LENGHT);
}

BOOL try_pop_task_queue(task_queue* tasks_q, task* task) {
    spinlock_lock(&tasks_q->lock);
    if (is_empty_task_queue(tasks_q)) { // probably doesn't need to be locked
        spinlock_unlock(&tasks_q->lock);
        puts("Task queue is empty.");
        return FALSE;
    }
    memcpy(task, &tasks_q->tasks[tasks_q->read_idx], sizeof(*task));
    tasks_q->size--;
    tasks_q->read_idx = (tasks_q->read_idx + 1) & (MAX_QUEUE_LENGHT - 1);
    puts("Pop task from task queue");
    spinlock_unlock(&tasks_q->lock);
    return TRUE;
}

BOOL try_push_task_queue(task_queue* tasks_q, task* task) {
    spinlock_lock(&tasks_q->lock);
    if (is_full_task_queue(tasks_q)) { // probably doesn't need to be locked
        spinlock_unlock(&tasks_q->lock);
        puts("Task queue is full.");
        return FALSE;
    }
    memcpy(&tasks_q->tasks[tasks_q->write_idx], task, sizeof(*task));
    tasks_q->size++;
    tasks_q->write_idx = (tasks_q->write_idx + 1) & (MAX_QUEUE_LENGHT - 1);
    puts("Pushed task to task queue");
    spinlock_unlock(&tasks_q->lock);
    return TRUE;
}    
#endif // _QUEUE_H_
