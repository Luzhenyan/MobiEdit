#include "Context.hpp"
#include "Module.hpp"
#include "OpDefined.hpp"
#include "QNNBackend.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include <cstdint>
#include <memory>

using namespace mllm;

auto main() -> int {
    Context::Instance().initBackend(MLLM_CPU);
    Context::Instance().initBackend(MLLM_QNN);

    auto qnnBackend = Context::Instance().globalBackends<QNNBackend>(MLLM_QNN);

    auto input = std::make_shared<Tensor>(qnnBackend);
    input->reshape(1, 16, 1680, 1680);
    input->setDtype(MLLM_TYPE_F16);
    input->setName("input");
    input->alloc();

    auto output = std::make_shared<Tensor>(qnnBackend);
    output->reshape(1, 16, 1680, 1680);
    output->setDtype(MLLM_TYPE_F16);
    output->setName("output");
    output->alloc();
    output->setTtype(GRAPH_OUTPUT);

    for (int i = 0; i < 4; i++) {
        input->hostPtr<float>()[i] = i;
    }

    // qnn build graph
    uint32_t dimensionsOutput[4] = {1, 16, 1680, 1680};

    vector<shared_ptr<Tensor>> inputs = {input}, outputs{output};
    qnnBackend->onSetUpStart(inputs, outputs, "test_graph");

    OpParam param;
    param["type"] = SOFTMAX;
    param["axis"] = 3;
    auto matmulOp = qnnBackend->opCreate(param, "matmul_op");

    matmulOp->reshape(inputs, outputs);
    matmulOp->setUp(inputs, outputs);

    qnnBackend->pushOutputBuffers(output->hostPtr<uint8_t>());
    qnnBackend->onSetUpEnd(inputs, outputs, "test_graph");
    qnnBackend->onExecuteStart(inputs, outputs);

    return 0;
}