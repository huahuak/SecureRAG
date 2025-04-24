#include "utils.h"

#include <stdarg.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef SGX
#include "enclave.h"
#endif

void err(const char *fmt, ...) {
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    printf("[ERROR MSG]: %s\n", buf);
#ifndef SGX
    exit(-1);
#endif
}

// MARK: Param
Param::Param(char *m, int msize, std::vector<int> offset) {
    this->m = m;
    this->now = nullptr;
    this->msize = msize;
    this->offset = offset;
}

Param::Param() {
    m = nullptr;
    now = m;
    msize = 0;
    offset = {};
}

Param::~Param() {
    if (this->now != nullptr) {  // free memory in normal env.
        free(m);
    }
}

void Param::executeMemcpy() {
    m = (char *)malloc(msize);
    now = m;
    for (auto fn : todoFn) {
        fn();
    }
}