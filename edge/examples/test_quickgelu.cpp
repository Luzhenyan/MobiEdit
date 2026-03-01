#include "Module.hpp"
#include "OpDefined.hpp"
#include "QNNBackend.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include <cstdint>
#include <memory>

using namespace mllm;

auto main() -> int {
    Module::initBackend(MLLM_CPU);
    Module::initBackend(MLLM_QNN);

    auto qnnBackend = static_cast<QNNBackend *>(Backend::global_backends[MLLM_QNN]);

    auto input = std::make_shared<Tensor>(qnnBackend);
    input->reshape(1, 1, 1, 1680);
    input->setDtype(MLLM_TYPE_F32);
    input->setName("input");
    input->alloc();

    auto output = std::make_shared<Tensor>(qnnBackend);
    output->reshape(1, 1, 1, 1680);
    output->setDtype(MLLM_TYPE_F32);
    output->setName("output");
    output->alloc();
    output->setTtype(GRAPH_OUTPUT);

    for (int i = 0; i < 1680; i++) {
        input->hostPtr<float>()[i] = i-840;
    }

    // qnn build graph
    uint32_t dimensionsOutput[4] = {1, 1, 1, 1680};

    vector<shared_ptr<Tensor>> inputs = {input}, outputs{output};
    qnnBackend->onSetUpStart(inputs, outputs, "test_graph");

    OpParam param;
    param["type"] = QUICKGLUE;
    auto matmulOp = qnnBackend->opCreate(param, "quickgelu");

    matmulOp->reshape(inputs, outputs);
    matmulOp->setUp(inputs, outputs);

    qnnBackend->pushOutputBuffers(output->hostPtr<uint8_t>());
    qnnBackend->onSetUpEnd(inputs, outputs, "test_graph");
    qnnBackend->onExecuteStart(inputs, outputs);

    output->printData<float>();

    return 0;
}