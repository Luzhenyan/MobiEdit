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
    input->reshape(1, 1, 2, 6);
    input->setDtype(MLLM_TYPE_I16);
    input->setName("input");
    input->alloc();

    auto output = std::make_shared<Tensor>(qnnBackend);
    output->reshape(1, 1, 2, 2);
    output->setDtype(MLLM_TYPE_I16);
    output->setName("output");
    output->alloc();
    output->setTtype(GRAPH_OUTPUT);
    auto output1 = std::make_shared<Tensor>(qnnBackend);
    output1->reshape(1, 1, 2, 2);
    output1->setDtype(MLLM_TYPE_I16);
    output1->setName("output1");
    output1->alloc();
    output1->setTtype(GRAPH_OUTPUT);
    auto output2 = std::make_shared<Tensor>(qnnBackend);
    output2->reshape(1, 1, 2, 2);
    output2->setDtype(MLLM_TYPE_I16);
    output2->setName("output2");
    output2->alloc();
    output2->setTtype(GRAPH_OUTPUT);

    for (int i = 0; i < 12; i++) {
        input->hostPtr<int16_t>()[i] = i;
    }

    input->printData<int16_t>();

    // qnn build graph

    vector<shared_ptr<Tensor>> inputs = {input}, outputs{output, output1, output2};
    qnnBackend->onSetUpStart(inputs, outputs, "test_graph");

    std::cout << "------after setup" << std::endl;

    OpParam param;
    param["type"] = SPLIT;
    param["split_num"] = 3;
    param["split_dim"] = DIMENSION;
    param["split_dim_size"] = 2;

    // store each dims
    vector<int> each_dims = {2, 2, 2};
    for (size_t i = 0; i < each_dims.size(); ++i) {
        param["split_dim_size_" + std::to_string(i)] = (float)each_dims[i];
    }
    auto matmulOp = qnnBackend->opCreate(param, "split_op");

    matmulOp->reshape(inputs, outputs);
    matmulOp->setUp(inputs, outputs);


    qnnBackend->pushOutputBuffers(output->hostPtr<uint8_t>());
    std::cout  << "fef" << (void*)output->hostPtr<uint8_t>() << std::endl;
    qnnBackend->pushOutputBuffers(output1->hostPtr<uint8_t>());
    qnnBackend->pushOutputBuffers(output2->hostPtr<uint8_t>());
    qnnBackend->onSetUpEnd(inputs, outputs, "test_graph");
    qnnBackend->onExecuteStart(inputs, outputs);

    output->printData<int16_t>();
    output1->printData<int16_t>();
    output2->printData<int16_t>();

    return 0;
}