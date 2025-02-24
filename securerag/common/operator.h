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
   public:
    virtual TensorDict forward(TensorDict x);

   protected:
    TensorDict tensorParam;
    std::shared_ptr<Tensor> getTensor(std::string key);
    std::shared_ptr<Tensor> getTensor(TensorDict x, std::string key);
};

class Linear : Operator {
   public:
    static const std::string WEIGHT;
    static const std::string BIAS;
    static const std::string X;
    static const std::string Y;
    Linear(std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias);
    TensorDict forward(TensorDict x);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

class Attention : Operator {
   public:
    static const std::string QUERY;
    static const std::string KEY;
    static const std::string VALUE;
    static const std::string OUTPUT;
    static const std::string BATCH_SIZE;
    static const std::string QUERY_NUM;
    static const std::string SEQ_LEN;
    static const std::string HEADS_NUM;
    static const std::string HEAD_SIZE;
    Attention(std::shared_ptr<Linear> queryProj,
              std::shared_ptr<Linear> keyProj,
              std::shared_ptr<Linear> valueProj,
              std::shared_ptr<Linear> outProj);
    TensorDict forward(TensorDict x);

   private:
    std::shared_ptr<Linear> queryProj;
    std::shared_ptr<Linear> keyProj;
    std::shared_ptr<Linear> valueProj;
    std::shared_ptr<Linear> outProj;
};

#endif