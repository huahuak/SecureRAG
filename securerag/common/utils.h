#ifndef UTIL_H
#define UTIL_H
#include "sgx_error.h"

void err(const char *fmt, ...);
void ret_error_support(sgx_status_t ret);

#endif