#include "sgx_error.h"
#include <cstdio>
#include <cstdlib>
#include <stdarg.h>

#ifdef SGX
#include "../Enclave/Enclave.h"
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

#ifndef SGX
typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
} sgx_errlist_t;

static sgx_errlist_t sgx_errlist[] = {
    {SGX_ERROR_UNEXPECTED, "Unexpected error occurred."},
    {SGX_ERROR_INVALID_PARAMETER, "Invalid parameter."},
    {SGX_ERROR_OUT_OF_MEMORY, "Out of memory."},
    {SGX_ERROR_ENCLAVE_LOST, "Power transition occurred."},
    {SGX_ERROR_INVALID_ENCLAVE, "Invalid enclave image."},
    {SGX_ERROR_INVALID_ENCLAVE_ID, "Invalid enclave identification."},
    {SGX_ERROR_INVALID_SIGNATURE, "Invalid enclave signature."},
    {SGX_ERROR_OUT_OF_EPC, "Out of EPC memory."},
    {SGX_ERROR_NO_DEVICE, "Invalid SGX device."},
    {SGX_ERROR_MEMORY_MAP_CONFLICT, "Memory map conflicted."},
    {SGX_ERROR_MEMORY_MAP_FAILURE, "Failed to reserve memory for the enclave."},
    {SGX_ERROR_INVALID_METADATA, "Invalid encalve metadata."},
    {SGX_ERROR_DEVICE_BUSY, "SGX device is busy."},
    {SGX_ERROR_INVALID_VERSION, "Enclave metadata version is invalid."},
    {SGX_ERROR_ENCLAVE_FILE_ACCESS, "Can't open enclave file."},

    {SGX_ERROR_INVALID_FUNCTION, "Invalid function name."},
    {SGX_ERROR_OUT_OF_TCS, "Out of TCS."},
    {SGX_ERROR_ENCLAVE_CRASHED, "The enclave is crashed."},

    {SGX_ERROR_MAC_MISMATCH, "Report varification error occurred."},
    {SGX_ERROR_INVALID_ATTRIBUTE, "The enclave is not authorized."},
    {SGX_ERROR_INVALID_CPUSVN, "Invalid CPUSVN."},
    {SGX_ERROR_INVALID_ISVSVN, "Invalid ISVSVN."},
    {SGX_ERROR_INVALID_KEYNAME, "The requested key name is invalid."},

    {SGX_ERROR_SERVICE_UNAVAILABLE, "AESM service is not responsive."},
    {SGX_ERROR_SERVICE_TIMEOUT, "Request to AESM is time out."},
    {SGX_ERROR_SERVICE_INVALID_PRIVILEGE,
     "Error occurred while getting launch token."},
};

void ret_error_support(sgx_status_t ret) {
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist / sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if (ret == sgx_errlist[idx].err) {
            printf("ERROR: %s\n", sgx_errlist[idx].msg);
            // std::cout << "Error: " << sgx_errlist[idx].msg << std::endl;
            break;
        }
    }
    if (idx == ttl)
        printf("Error: Unexpected error occurred.");
    return;
}

#endif