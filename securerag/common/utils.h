#ifndef UTIL_H
#define UTIL_H
#include "sgx_error.h"
#include <cstddef>
#include <vector>

void err(const char *fmt, ...);
void ret_error_support(sgx_status_t ret);

class Param {
  public:
    char *m;
    char *now;
    int msize;
    std::vector<int> offset;

    Param(char *m, int msize, std::vector<int> offset);
    Param(size_t initSize);
    ~Param();

    template <typename T> void putPtr(T *p, int len) {
        size_t siz = len * sizeof(T);
        memcpy(now, p, siz);
        now += siz;
        offset.push_back(msize);
        msize += siz;
    };

    template <typename T> void put(T p) {
        memcpy(now, &p, sizeof(T));
        now += sizeof(T);
        offset.push_back(msize);
        msize += sizeof(T);
    }

    template <typename T> T *getPtr(int idx) {
        auto p = m + offset.at(idx);
        return (T *)p;
    }

    template <typename T> T get(int idx) {
        auto p = m + offset.at(idx);
        return *((T *)p);
    }
};
#endif