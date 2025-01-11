#include "func.h"

#ifdef EIGEN
#include <unsupported/Eigen/CXX11/Tensor>

#include "Eigen/Dense"

using TensorFloat3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using TensorFloat2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
#endif

#ifdef SGX
#include "../Enclave/Enclave.h"
#endif

#ifndef SGX
#include <cstdio>
#endif

void linear(float *input, float *weight, float *bias, float *output, int N,
            int indim, int outdim) {
#ifdef EIGEN
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        inputMatrix(input, N, indim);

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        weightMatrix(weight, outdim, indim);

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        biasMatrix(bias, 1, outdim);

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        outputMatrix(output, N, outdim);

    outputMatrix =
        inputMatrix * weightMatrix.transpose() + biasMatrix.replicate(N, 1);
#else
#ifdef DEBUG
    // printf("N: %d\n", N);
    // printf("indim: %d\n", indim);
    // printf("outdim: %d\n", outdim);
#endif
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < outdim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < indim; k++) {
                sum += input[i * indim + k] * weight[j * indim + k];
#ifdef DEBUG
// printf("weight: %f\n", weight[j * indim + k]);
// printf("input: %f\n", input[i * indim + k]);
// printf("sum: %f\n", sum);
#endif
            }
            output[i * outdim + j] = sum + bias[j];
        }
    }
#endif
}

void attention(float *q, float *k, float *out, float *qw, float *qb, float *kw,
               float *kb, float *vw, float *vb, float *fw, float *fb, int bsz,
               int tgtlen, int srclen, int embeddim, int nh) {
#ifdef EIGEN
    Eigen::TensorMap<TensorFloat3D> query(q, bsz, tgtlen, embeddim);
    Eigen::TensorMap<TensorFloat3D> key(k, bsz, srclen, embeddim);
    Eigen::TensorMap<TensorFloat3D> value(k, bsz, srclen, embeddim);

    Eigen::TensorMap<TensorFloat2D> queryW(qw, embeddim, embeddim);
    Eigen::TensorMap<TensorFloat2D> keyW(kw, embeddim, embeddim);
    Eigen::TensorMap<TensorFloat2D> valueW(vw, embeddim, embeddim);
    Eigen::TensorMap<TensorFloat2D> outW(fw, embeddim, embeddim);

    Eigen::TensorMap<TensorFloat2D> queryB(qb, embeddim, embeddim);
    Eigen::TensorMap<TensorFloat2D> keyB(kb, embeddim, embeddim);
    Eigen::TensorMap<TensorFloat2D> valueB(vb, embeddim, embeddim);
    Eigen::TensorMap<TensorFloat2D> outB(fb, embeddim, embeddim);

    Eigen::TensorMap<TensorFloat3D> output(out, bsz, tgtlen, embeddim);

    auto Q = query * queryW + queryB;
    auto K = key * keyW + keyB;
    auto V = value * valueW + valueB;

    int headdim = embeddim / nh;
    auto QT = Q.reshape(Eigen::array<int, 3>{tgtlen, bsz * nh, headdim})
                  .shuffle(Eigen::array<int, 3>({1, 0, 2}));

    auto KT = K.reshape(Eigen::array<int, 3>{srclen, bsz * nh, headdim})
                  .shuffle(Eigen::array<int, 3>({1, 2, 0}));

    auto QK = QT * KT;

    // softmax
    auto rowmax = QK.maximum(Eigen::array<int, 1>{2});
    auto stabilized = QK - rowmax.broadcast(Eigen::array<int, 3>{1, 1, srclen});
    auto exp = stabilized.exp();
    auto rowsum = exp.sum(Eigen::array<int, 1>{2});
    auto scores = exp / rowsum.broadcast(Eigen::array<int, 3>{1, 1, srclen});

    // full connection
    auto attn = scores * V;
    output = output.shuffle(Eigen::array<int, 3>{1, 0, 2})
                 .reshape(Eigen::array<int, 3>{tgtlen, bsz, embeddim});
    output = attn * outW + outB;
#endif
}