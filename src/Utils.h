#ifndef SOCNET_UTILS
#define SOCNET_UTILS
    #include <time.h>

    clock_t getTime();
    clock_t getTimeSince(clock_t sincetime);

    float rand01();
    int randInt(int lim);
    void randomColor(float& r, float& g, float& b);

    float min(float v1, float v2);
    float max(float v1, float v2);
    float clamp(float val, float min, float max);

#endif
