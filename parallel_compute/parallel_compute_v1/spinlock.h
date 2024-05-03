#ifndef _SPINLOCK_H_
#define _SPINLOCK_H_

#include <stdatomic.h>
#include <time.h>
#include <emmintrin.h>

void spinlock_lock(volatile s32* lock) {
    static const struct timespec ns = {0, 1};
    for (int i = 0; *lock || atomic_exchange(lock, 1); i++) {
        if (i == 8) {
            i = 0;
            nanosleep(&ns, NULL);
        }
        _mm_pause();
    }
}

void spinlock_unlock(volatile s32* lock) {
    *lock = 0;
}

#endif // _SPINLOCK_H_
