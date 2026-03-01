
#pragma once

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPURomeVector final : public Op {
public:
    CPURomeVector(Backend *bn, string opName, int batch, int head, int seq, int dim, int threadCount);
    virtual ~CPURomeVector() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        return romeTensor;
    }

private:
    int thread_count = 4;
    Tensor romeTensor;
    int batch_;
    int head_;
    int seq_;
    int dim_;
};

class CPURomeVectorCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int batch = (int)op_param["batch"];
        int head = (int)op_param["head"];
        int seq = (int)op_param["seq"];
        int dim = (int)op_param["dim"];
        return new CPURomeVector(bn, name, batch, head, seq, dim, threadCount);
    }
};

} // namespace mllm
