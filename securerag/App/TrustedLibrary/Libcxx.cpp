/*
 * Copyright (C) 2011-2017 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

#include "../App.h"
#include "Enclave_u.h"
#include "func.h"
#include "operator.h"
#include "sgx_error.h"
#include "utils.h"
#include <thread>

void ecall_libcxx_functions(void) {
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    // Example for lambda function feature:
    ret = ecall_lambdas_demo(global_eid);
    if (ret != SGX_SUCCESS)
        abort();
}

void sgxSecureLinear(float *input, float *weight, float *bias, float *output,
                     int N, int indim, int outdim) {
    void *param;
    size_t paramsize;
    size_t outsize;

    // calculate size and malloc space
    paramsize = (N * indim * sizeof(float)) + (indim * outdim * sizeof(float)) +
                (outdim * sizeof(float)) + (3 * sizeof(int));
    outsize = (N * outdim * sizeof(float));
    param = (void *)malloc(paramsize);

    // copy data
    char *p = (char *)param;
    memcpy(p, input, N * indim * sizeof(float));
    p += N * indim * sizeof(float);
    memcpy(p, weight, outdim * indim * sizeof(float));
    p += indim * outdim * sizeof(float);
    memcpy(p, bias, outdim * sizeof(float));
    p += outdim * sizeof(float);
    int *p2 = (int *)p;
    *p2 = N;
    p2 += 1;
    *p2 = indim;
    p2 += 1;
    *p2 = outdim;

    //  call
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    ret = ecallSGXOperator(global_eid, LINEAR, (void *)param, paramsize,
                           (void *)output, outsize);
    if (ret != SGX_SUCCESS) {
        ret_error_support(ret);
        err("ecallSGXOperator FAILED, [ERR CODE]: %d\n", ret);
    }
    // linear(input, weight, bias, output, N, indim, outdim);
    free(param);
}