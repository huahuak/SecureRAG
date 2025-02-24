#ifndef FUNC_H
#define FUNC_H

#include "data.h"

using T = float;
static const auto TensorTyp = Typ::FLOAT32;

void embedding();

void linear(T *input, T *weight, T *bias, T *output, int N, int indim,
            int outdim);

void attention(T *q, T *k, T *out, T *qw, T *qb, T *kw, T *kb, T *vw, T *vb,
               T *fw, T *fb, int bsz, int tgtlen, int srclen, int embeddim,
               int nh);

#if defined(EIGEN) && defined(SGX)
#include <vector>

#include "dnnl.hpp"

namespace dnnlfunc {

using namespace dnnl;
static const auto dnnlDataType = memory::data_type::f32;

struct attn_dims_t {
    memory::dim mb;
    memory::dim seq_len;
    memory::dim head_num;
    memory::dim head_size;
    memory::dim query_num;
};

void attention(const T *const query, const T *const key, const T *const &value,
               const T *const mask, T *const output, const attn_dims_t &p,
               bool hasMask = false, memory::data_type dt = dnnlDataType,
               engine::kind ekind = engine::kind::cpu);

void linear(std::vector<T> &input, std::vector<T> &weight, std::vector<T> &bias,
            std::vector<T> &output, int N, int indim, int outdim);
}  // namespace dnnlfunc
#endif

#endif