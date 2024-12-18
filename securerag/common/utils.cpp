#include <cstdio>
#include <cstdlib>
#ifdef SGX
#include "../Enclave/Enclave.h"
#endif

void err(const char *msg) {
    printf("[ERROR MSG]: %s\n", msg);
#ifndef SGX
    exit(-1);
#endif
}