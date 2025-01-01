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

#include "../Enclave.h"
#include "Enclave_t.h"
#include "func.h"
#include "operator.h"
#include "utils.h"
#include <string.h>

void ecall_lambdas_demo() {}

void ecallSGXOperator(const char *func, void *param, size_t paramSize,
                      int *offset, size_t offsetLen, void *output,
                      size_t outSize) {
    Param parameter((char *)param, (int)paramSize,
                    std::vector<int>(offset, offset + offsetLen));
    if (strcmp(func, LINEAR) == 0) {
        float *input = parameter.getPtr<float>(0);
        float *weight = parameter.getPtr<float>(1);
        float *bias = parameter.getPtr<float>(2);
        int N = parameter.get<int>(3);
        int indim = parameter.get<int>(4);
        int outdim = parameter.get<int>(5);
        linear(input, weight, bias, (float *)output, N, indim, outdim);
        return;
    }
}

// void ocallGetLinearInfo(float *weight, float *bias, int N, int indim,
//                         int outdim, int *linearid) {}

