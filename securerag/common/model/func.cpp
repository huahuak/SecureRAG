#include "func.h"

// #include <assert.h>

#include <cmath>
#include <cstddef>
// #include <iomanip>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifndef SGX
#include <cstdio>
#endif

#ifdef SGX

#ifdef DEBUG
#include "../Enclave/Enclave.h"
#endif
#include "dnnl.hpp"
#include "dnnl_types.h"
#include "dnnl_utils.h"
#include "eigen_sgx.h"

using namespace dnnl;

using tag = memory::format_tag;
auto dtype = memory::data_type::f32;
using precision = float;
#endif

#ifdef EIGEN

#ifndef SGX
#include <unsupported/Eigen/CXX11/Tensor>
#endif

using TensorFloat4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;
using TensorFloat3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using TensorFloat2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
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

namespace dnnlfunc {
#if defined(EIGEN) && defined(SGX)
struct sdpa_dims_t {
    memory::dim mb;
    memory::dim seq_len;
    memory::dim head_num;
    memory::dim head_size;
    memory::dim query_num;
};

static size_t product(const memory::dims &dims) {
    return (size_t)std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
                                   std::multiplies<memory::dim>());
}

// m3 = m1 * m2;
static void BMM(memory m1, memory m2, memory m3) {
    auto Tensor = [](memory m) -> TensorFloat4D {
        auto DIM_SIZE = 4;
        auto desc = m.get_desc();
        auto dim = desc.data.dims;
        assert(desc.data.ndims == DIM_SIZE && "wrong ndims!");
        assert(desc.data.data_type == dnnl_data_type_t::dnnl_f32 &&
               "unsupported data_type!");
        Eigen::TensorMap<TensorFloat4D> tensor((float *)m.get_data_handle(),
                                               dim[0], dim[1], dim[2], dim[3]);
        return tensor;
    };

    auto t1 = Tensor(m1);
    auto t2 = Tensor(m2);
    auto t3 = Tensor(m3);

    typedef TensorFloat4D::DimensionPair DimPair;
    Eigen::array<DimPair, 1> dims{DimPair(1, 0)};
    const long d0 = t1.dimension(0);
    const long d1 = t1.dimension(1);
    for (int i = 0; i < d0; ++i) {
        TensorFloat3D t1_d0 = t1.chip<0>(i);
        TensorFloat3D t2_d0 = t2.chip<0>(i);
        TensorFloat3D t3_d0 = t3.chip<0>(i);
        for (int j = 0; j < d1; ++j) {
            t3_d0.chip<0>(j) =
                t1_d0.chip<0>(j).contract(t2_d0.chip<0>(j), dims);
        }
    }
}

void attention_network(engine::kind ekind,
                       const sdpa_dims_t &p = {32, 384, 16, 64, 384},
                       memory::data_type dt = dtype) {
    // Create execution dnnl::engine.
    dnnl::engine eng(ekind, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // network and argument
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> arg;

    // Prepare input and output shapes.
    const dnnl::memory::dims q_sz = {p.mb, p.head_num, p.query_num,
                                     p.head_size};
    const dnnl::memory::dims k_sz = {p.mb, p.head_num, p.head_size, p.seq_len};
    const dnnl::memory::dims v_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const dnnl::memory::dims score_sz = {p.mb, p.head_num, p.query_num,
                                         p.seq_len};
    const dnnl::memory::dims scale_sz = {p.mb, p.head_num, p.query_num,
                                         p.seq_len};
    // const memory::dims scale_sz = {1, 1, 1, 1};
    const dnnl::memory::dims mask_sz = {p.mb, 1, p.query_num, p.seq_len};

    // Memory description
    auto query_md = memory::desc(q_sz, dt, tag::abcd);
    auto key_md = memory::desc(
        k_sz, dt, tag::abcd);  // MARK: todo: need to reorder to tag::abdc
    auto score_md = memory::desc(score_sz, dt, tag::abcd);
    auto scale_md = memory::desc(scale_sz, dt, tag::abcd);
    auto mask_md = memory::desc(mask_sz, dt, tag::abcd);
    auto value_md = memory::desc(v_sz, dt, tag::abcd);
    auto output_md = memory::desc(q_sz, dt, tag::abcd);

    // Create memory objects.
    auto m_query = memory(query_md, eng);
    auto m_key = memory(key_md, eng);
    auto m_scale = memory(scale_md, eng);
    auto m_mask = memory(mask_md, eng);
    auto m_value = memory(value_md, eng);
    auto m_output = memory(output_md, eng);
    auto m_score = memory(score_md, eng);

    // Allocate user data.
    std::vector<float> query_data(product(q_sz));
    std::vector<float> key_data(product(k_sz));
    std::vector<float> scale_data(product(scale_sz));
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(v_sz));
    std::vector<float> output_data(product(q_sz));

    // Write data to tensor object's handle.
    write_to_dnnl_memory(query_data.data(), m_query);
    write_to_dnnl_memory(key_data.data(), m_key);
    write_to_dnnl_memory(scale_data.data(), m_scale);
    write_to_dnnl_memory(mask_data.data(), m_mask);
    write_to_dnnl_memory(value_data.data(), m_value);

    // scaled_score = score / scale
    auto scaledDesc = eltwise_forward::desc(
        prop_kind::forward_inference, algorithm::eltwise_linear, score_md,
        float(1.f / std::sqrt(p.head_size)), 0);
    auto scaledPrimDesc = eltwise_forward::primitive_desc(scaledDesc, eng);
    auto scaledPrim = eltwise_forward(scaledPrimDesc);
    net.push_back(scaledPrim);
    arg.push_back({{DNNL_ARG_SRC, m_score}, {DNNL_ARG_DST, m_scale}});

    // masked_score = scaled_score + mask
    auto maskedDesc =
        binary::desc(algorithm::binary_add, scale_md, mask_md, score_md);
    auto maskedPrimDesc = binary::primitive_desc(maskedDesc, eng);
    auto maskedPrim = binary(maskedPrimDesc);
    net.push_back((maskedPrim));
    arg.push_back({{DNNL_ARG_SRC, m_scale},
                   {DNNL_ARG_SRC_1, m_mask},
                   {DNNL_ARG_DST, m_score}});

    // attention_probs = softmax(masked_score)
    primitive_attr softmax_attr;
    softmax_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto lastDim = int(score_sz.size() - 1);
    auto softmaxDesc =
        softmax_forward::desc(prop_kind::forward_inference, score_md, lastDim);
    auto softmax_pd =
        softmax_forward::primitive_desc(softmaxDesc, softmax_attr, eng);
    auto softmax_prim = softmax_forward(softmax_pd);
    auto softmax_scratchpad = softmax_pd.scratchpad_desc().get_size();
    auto scratchpad_md =
        memory::desc({static_cast<memory::dim>(softmax_scratchpad)},
                     memory::data_type::u8, tag::a);
    auto m_scratchpad = memory(scratchpad_md, eng);
    net.push_back(softmax_prim);
    arg.push_back({{DNNL_ARG_SRC, m_score},
                   {DNNL_ARG_SRC, m_score},
                   {DNNL_ARG_SCRATCHPAD, m_scratchpad}});

    const auto loop = [&]() {
        // score = query x key.T
        BMM(m_query, m_key, m_score);

        // attention_probs = softmax(score / scale + mask)
        assert(net.size() == arg.size() && "something is missing");
        for (size_t i = 0; i < net.size(); i++) {
            net.at(i).execute(strm, arg.at(i));
        }

        // attention_output = attention_probs x value
        BMM(m_score, m_value, m_output);
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop();

    // Wait for the computation to finish.
    strm.wait();
}
#endif
}  // namespace dnnlfunc