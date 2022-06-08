#pragma once

#include <assert.h>
#include <stdint.h>

namespace librandom
{

class simple
{
  private:
    volatile int64_t holdrand;

  public:
    inline simple(const int64_t _seed = 1) : holdrand(_seed)
    {
    }

    // C5220
    simple(const simple &) = delete;
    simple &operator=(const simple &) = delete;
    simple(simple &&) = delete;
    simple &operator=(simple &&) = delete;

    inline void seed(const int64_t val)
    {
        holdrand = val;
    }
    inline int64_t seed() const
    {
        return holdrand;
    }
    inline int32_t max_i() const
    {
        return 32767;
    }

    inline int32_t i()
    {
        return (((holdrand = holdrand * 214013L + 2531011L) >> 16) & 0x7fff);
    }

    inline int32_t i(int32_t max)
    {
        assert(max);
        return i() % max;
    }
    inline int32_t i(int32_t min, int32_t max)
    {
        return min + i(max - min);
    }
    inline int32_t is(int32_t range)
    {
        return i(-range, range);
    }
    inline int32_t is(int32_t range, int32_t offs)
    {
        return offs + is(range);
    }

    inline float max_f()
    {
        return 32767.f;
    }
    inline float f()
    {
        return float(i()) / max_f();
    }
    inline float f(float max)
    {
        return f() * max;
    }
    inline float f(float min, float max)
    {
        return min + f(max - min);
    }
    inline float fs(float range)
    {
        return f(-range, range);
    }
    inline float fs(float range, float offs)
    {
        return offs + fs(range);
    }
};
} // namespace librandom