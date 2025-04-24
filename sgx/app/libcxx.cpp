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
#include <stdio.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "app.h"
#include "Enclave_u.h"
#include "sgx_error.h"
#include "core.h"

void ecall_libcxx_functions(void) {
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    // Example for lambda function feature:
    ret = ecall_lambdas_demo(global_eid);
    if (ret != SGX_SUCCESS) abort();
}

TensorRef sgxCopyTensorToSGX(Tensor &tensor) {
    ID tensorRefId;
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    auto p = tensor.dataPtr();
    ret = ecallCopyTensorToSGX(global_eid, p.get(), tensor.elementSize, int(tensor.typ),
                               const_cast<long *>(tensor.dim.data()),
                               tensor.dim.size(), &tensorRefId);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        err("ecallCopyTensorToSGX FAILED, [ERR CODE]: %d\n", ret);
    }
    return TensorRef(tensorRefId, tensor.elementSize);
}

Tensor sgxCopyTensorFromSGX(TensorRef ref) {
    void *mem = malloc(ref.elementSize);
    size_t siz;
    int typ;
    long dim[MAX_DIM];
    size_t offset;
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    ret = ecallCopyTensorFromSGX(global_eid, ref.id, mem, &siz, &typ, dim,
                                 &offset);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        err("ecallCopyTensorToSGX FAILED, [ERR CODE]: %d\n", ret);
    }
    return Tensor(mem, siz, Typ(typ), std::vector<long>(dim, dim + offset));
}

void sgxSecureLinear(float *input, float *weight, float *bias, float *output,
                     int N, int indim, int outdim) {
    // calculate size and malloc space
    size_t outsize;
    outsize = (N * outdim * sizeof(float));

    // copy data
    Param param;
    param.putPtr(input, N * indim);
    param.putPtr(weight, outdim * indim);
    param.putPtr(bias, outdim);
    param.put(N);
    param.put(indim);
    param.put(outdim);
    param.executeMemcpy();

    //  call
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    ret = ecallSGXOperator(global_eid, LINEAR, (void *)param.m, param.msize,
                           param.offset.data(), param.offset.size(),
                           (void *)output, outsize);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        err("ecallSGXOperator FAILED, [ERR CODE]: %d\n", ret);
    }
}

void sgxSecureAttention(float *q, float *k, float *out, float *qw, float *qb,
                        float *kw, float *kb, float *vw, float *vb, float *fw,
                        float *fb, int bsz, int tgtlen, int srclen,
                        int embeddim, int nh) {
    // calculate size and malloc space
    size_t outsize = tgtlen * bsz * embeddim * sizeof(float);

    // copy data
    Param param;
    param.putPtr(q, bsz * tgtlen * embeddim)
        .putPtr(k, bsz * srclen * embeddim)
        .putPtr(qw, embeddim * embeddim)
        .putPtr(qb, embeddim)
        .putPtr(kw, embeddim * embeddim)
        .putPtr(kb, embeddim)
        .putPtr(vw, embeddim * embeddim)
        .putPtr(vb, embeddim)
        .putPtr(fw, embeddim * embeddim)
        .putPtr(fb, embeddim)
        .put(bsz)
        .put(tgtlen)
        .put(srclen)
        .put(embeddim)
        .put(nh)
        .executeMemcpy();

    // call
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    ret = ecallSGXOperator(global_eid, ATTENTION, (void *)param.m, param.msize,
                           param.offset.data(), param.offset.size(),
                           (void *)out, outsize);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        err("ecallSGXOperator FAILED, [ERR CODE]: %d\n", ret);
    }
}
