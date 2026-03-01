#include "Context.hpp"
#include "Module.hpp"
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
    input->reshape(1, 2, 2, 2);
    input->setDtype(MLLM_TYPE_F32);
    input->setName("input");
    input->alloc();
    auto input1 = std::make_shared<Tensor>(qnnBackend);
    input1->reshape(1, 2, 2, 2);
    input1->setDtype(MLLM_TYPE_F32);
    input1->setName("input1");
    input1->alloc();

    auto output = std::make_shared<Tensor>(qnnBackend);
    output->reshape(1, 2, 2, 2);
    output->setDtype(MLLM_TYPE_F32);
    output->setName("output");
    output->alloc();
    output->setTtype(GRAPH_OUTPUT);

    for(int i = 0; i < 8; i++) {
        input->hostPtr<float>()[i] = i;
        input1->hostPtr<float>()[i] = i;
    }

    input->printData<float>();
    input1->printData<float>();

    // qnn build graph
    uint32_t dimensionsOutput[4] = {1, 2, 2, 2};

    vector<shared_ptr<Tensor>> inputs = {input, input1}, outputs{output};
    qnnBackend->onSetUpStart(inputs, outputs, "test_graph");

    OpParam param;
    param["type"] = MATMUL;
    param["transpose0"] = false;
    param["transpose1"] = true;
    auto matmulOp = qnnBackend->opCreate(param, "matmul_op.qk");

    matmulOp->reshape(inputs, outputs);
    matmulOp->setUp(inputs, outputs);

    qnnBackend->pushOutputBuffers(output->hostPtr<uint8_t>());
    qnnBackend->onSetUpEnd(inputs, outputs, "test_graph");
    qnnBackend->onExecuteStart(inputs, outputs);

    output->printData<float>();

    auto cpuMatmul = Context::Instance().globalBackends(MLLM_CPU)->opCreate(param, "cpu_matmul_op");
    cpuMatmul->reshape(inputs, outputs);
    cpuMatmul->setUp(inputs, outputs);
    cpuMatmul->execute(inputs, outputs);

    output->printData<float>();

    return 0;
}