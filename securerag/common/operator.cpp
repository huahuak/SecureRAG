#include "operator.h"

#include <memory>

#include "data.h"
#include "func.h"

// MARK: Operator
std::shared_ptr<Tensor> Operator::getTensor(std::string key) {
    assert(tensorParam.find(key) != tensorParam.end() && "key not found!");
    return tensorParam[key];
}

std::shared_ptr<Tensor> Operator::getTensor(TensorDict x, std::string key) {
    assert(x.find(key) != x.end() && "key not found!");
    return x[key];
}

// MARK: Linear
const std::string Linear::WEIGHT = "weight";
const std::string Linear::BIAS = "bias";
const std::string Linear::X = "x";
const std::string Linear::Y = "y";

Linear::Linear(std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias) {
    tensorParam[WEIGHT] = weight;
    tensorParam[BIAS] = bias;
}

TensorDict Linear::forward(TensorDict x) {
    auto output = forward(getTensor(x, Linear::X));
    return {{Y, output}};
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> x) {
    auto weight = getTensor(Linear::WEIGHT);
    auto bias = getTensor(Linear::BIAS);
    auto input = x;

    assert(input->dim.size() == 2 && "input dim should be 2!");
    assert(weight->dim.size() == 2 && "weight dim should be 2!");
    assert(bias->dim.size() == 1 && "bias dim should be 1!");
    assert(input->dim[1] == weight->dim[0] && "input dim mismatch!");
    assert(weight->dim[1] == bias->dim[0] && "weight dim mismatch!");

    auto indim = input->dim[1];
    auto outdim = weight->dim[1];
    auto N = input->dim[0];
    auto outSiz = sizeof(float) * indim * outdim;
    auto mem = (float*)malloc(outSiz);
    std::shared_ptr<Tensor> output(
        new Tensor((void*)mem, outSiz, Typ::FLOAT32, {indim, outdim}));

    linear((float*)input->dataPtr().get(), (float*)weight->dataPtr().get(),
           (float*)bias->dataPtr().get(), (float*)mem, indim, N, outdim);
};

// MARK: Attention
const std::string Attention::QUERY = "query";
const std::string Attention::KEY = "key";
const std::string Attention::VALUE = "value";
const std::string Attention::OUTPUT = "output";
const std::string Attention::BATCH_SIZE = "batch_size";
const std::string Attention::SEQ_LEN = "seq_len";
const std::string Attention::QUERY_NUM = "query_num";
const std::string Attention::HEADS_NUM = "heads_num";
const std::string Attention::HEAD_SIZE = "head_size";

Attention::Attention(std::shared_ptr<Linear> queryProj,
                     std::shared_ptr<Linear> keyProj,
                     std::shared_ptr<Linear> valueProj,
                     std::shared_ptr<Linear> outProj) {
    this->queryProj = queryProj;
    this->keyProj = keyProj;
    this->valueProj = valueProj;
    this->outProj = outProj;
}

TensorDict Attention::forward(TensorDict x) {
    auto query = queryProj->forward(x[Attention::QUERY]);
    auto key = keyProj->forward(x[Attention::KEY]);
    auto value = valueProj->forward(x[Attention::VALUE]);
    std::shared_ptr<Tensor> attnprob(
        new Tensor(malloc(query->elementSize * query->elementNum()),
                   query->elementSize, TensorTyp, query->dim));
    auto p = [](std::shared_ptr<Tensor> tensor) {
        return (T*)tensor->dataPtr().get();
    };
    dnnlfunc::attention(p(query), p(key), p(value), {}, p(attnprob),
                        {
                            .mb = x[Attention::BATCH_SIZE]->first<int>(),
                            .seq_len = x[Attention::SEQ_LEN]->first<int>(),
                            .head_num = x[Attention::HEADS_NUM]->first<int>(),
                            .head_size = x[Attention::HEAD_SIZE]->first<int>(),
                            .query_num = x[Attention::QUERY_NUM]->first<int>(),
                        });
    auto output = outProj->forward(attnprob);
    return {{OUTPUT, output}};
}