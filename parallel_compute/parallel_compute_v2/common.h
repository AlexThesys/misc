#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdatomic.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>

#define TRUE 1
#define FALSE 0

#define _max(x, y) (x) > (y) ? (x) : (y)
#define _min(x, y) (x) < (y) ? (x) : (y)

typedef int64_t s64;
typedef uint64_t u64;
typedef int32_t s32;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef int BOOL;

#endif // _COMMON_H_
