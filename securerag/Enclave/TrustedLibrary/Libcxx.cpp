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

#include <string.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../Enclave.h"
#include "Enclave_t.h"
#include "constant.h"
#include "data.h"
#include "func.h"
#include "operator.h"
#include "utils.h"

void ecall_lambdas_demo() {}

void ecallSGXOperator(const char *func, void *param, size_t paramSize,
                      int *offset, size_t offsetLen, void *output,
                      size_t outSize) {
    Param p((char *)param, (int)paramSize,
            std::vector<int>(offset, offset + offsetLen));
    if (strcmp(func, LINEAR) == 0) {
        float *input = p.getPtr<float>(0);
        float *weight = p.getPtr<float>(1);
        float *bias = p.getPtr<float>(2);
        int N = p.get<int>(3);
        int indim = p.get<int>(4);
        int outdim = p.get<int>(5);
        linear(input, weight, bias, (float *)output, N, indim, outdim);
        return;
    }
    if (strcmp(func, ATTENTION) == 0) {
        attention(p.getPtr<float>(0), p.getPtr<float>(1), (float *)output,
                  p.getPtr<float>(2), p.getPtr<float>(3), p.getPtr<float>(4),
                  p.getPtr<float>(5), p.getPtr<float>(6), p.getPtr<float>(7),
                  p.getPtr<float>(8), p.getPtr<float>(9), p.get<int>(10),
                  p.get<int>(11), p.get<int>(12), p.get<int>(13),
                  p.get<int>(14));
        return;
    }
}

void ecallSGXAttention(float *q, float *k, float *out, float *qw, float *qb,
                       float *kw, float *kb, float *vw, float *vb, float *fw,
                       float *fb, int bsz, int tgtlen, int srclen, int embeddim,
                       int nh) {
    attention(q, k, out, qw, qb, kw, kb, vw, vb, fw, fb, bsz, tgtlen, srclen,
              embeddim, nh);
}

// MARK: Tensor
class TensorManager {
    int nextId() {
        std::unique_lock<std::mutex> lck(mtx);
        IncrementalID += 1;
        return IncrementalID;
    }

    std::unordered_map<unsigned long, Tensor> tensorHolder;
    std::mutex mtx;
    ID IncrementalID = 0;

   public:
    ID put(void *mem, int siz, Typ typ, std::vector<long> dim) {
        auto id = nextId();
        tensorHolder.emplace(id, Tensor(mem, siz, typ, dim));
        return id;
    }

    Tensor get(ID id) {
        auto item = tensorHolder.find(id);
        if (item == tensorHolder.end()) {
            ERR("can't found ID");
            return Tensor::EMPTY;
        }
        return item->second;
    }

    Tensor remove(ID id) {
        auto item = tensorHolder.find(id);
        if (item == tensorHolder.end()) {
            ERR("can't found ID");
            return Tensor::EMPTY;
        }
        auto ret = std::move(item->second);
        tensorHolder.erase(id);
        return ret;
    }
};

TensorManager TM;

void ecallCopyTensorToSGX(void *src, size_t siz, int typ, long *dim,
                          size_t dimSiz, ID *tensorRefId) {
    auto tmp = (uint8_t *)malloc(siz);
    memcpy(tmp, src, siz);
    std::vector<long> dimvec;
    dimvec.assign(dim, dim + dimSiz);
    *tensorRefId = TM.put(tmp, siz, Typ(typ), dimvec);
}

void ecallCopyTensorFromSGX(ID tensorRefId, void *dst, size_t *siz, int *typ,
                            long *dim, size_t *dimSiz) {
    auto t = TM.get(tensorRefId);
    memcpy(dst, t.dataPtr().get(), t.siz);
    *siz = t.siz;
    *typ = int(t.typ);
    assert(t.dim.size() < MAX_DIM);
    memcpy(dim, t.dim.data(), t.dim.size() * sizeof(long));
    *dimSiz = t.dim.size();
}
