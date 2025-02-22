#ifndef UTIL_H
#define UTIL_H
#include <cstddef>
#include <functional>
#include <vector>

#include "sgx_error.h"

#define ERR(fmt, ...)                       \
    printf("[%s:%d] ", __FILE__, __LINE__); \
    err(fmt, ##__VA_ARGS__);                \
    printf("\n");

void err(const char *fmt, ...);

#ifndef SGX
// this API is only for non-SGX env.
void ret_error_support(sgx_status_t ret);
#endif

class Param {
   public:
    char *m;
    char *now;
    int msize;
    std::vector<int> offset;
    std::vector<std::function<void()>> todoFn;

    Param(char *m, int msize, std::vector<int> offset);
    Param();
    ~Param();

    template <typename T>
    Param &putPtr(T *p, int len) {
        size_t siz = len * sizeof(T);
        todoFn.push_back([=]() {
            memcpy(now, p, siz);
            now += siz;
        });
        offset.push_back(msize);
        msize += siz;
        return *this;
    };

    template <typename T>
    Param &put(T p) {
        todoFn.push_back([=]() {
            memcpy(now, &p, sizeof(T));
            now += sizeof(T);
        });
        offset.push_back(msize);
        msize += sizeof(T);
        return *this;
    }

    void executeMemcpy();

    template <typename T>
    T *getPtr(int idx) {
        auto p = m + offset.at(idx);
        return (T *)p;
    }

    template <typename T>
    T get(int idx) {
        auto p = m + offset.at(idx);
        return *((T *)p);
    }
};
#endif