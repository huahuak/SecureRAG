#ifndef OPERATOR_H
#define OPERATOR_H

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>

#include "model/data.h"

#define LINEAR "Linear"
#define ATTENTION "Attention"

class Operator {
   protected:
    TensorDict param;

   public:
    virtual TensorDict forward(TensorDict x);
};

class Linear : Operator {
   public:
    Linear(std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias) {
        param["w"] = weight;
        param["b"] = bias;
    }

    TensorDict forward(TensorDict x);
};

#endif