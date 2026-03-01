#include "Context.hpp"
#include "Layer.hpp"
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

    auto cpuBackend = Context::Instance().globalBackends<CPUBackend>(MLLM_CPU);

    auto image_size = std::make_shared<Tensor>(cpuBackend);
    image_size->reshape(1, 1, 1, 3);
    image_size->setDtype(MLLM_TYPE_F32);
    image_size->setName("input");
    image_size->alloc();
    image_size->hostPtr<float>()[0] = float(1);
    image_size->hostPtr<float>()[1] = float(2);
    image_size->hostPtr<float>()[2] = float(4);

    auto pos_id = std::make_shared<Tensor>(cpuBackend);
    pos_id->setDtype(MLLM_TYPE_F32);
    pos_id->setName("output");
    pos_id->alloc();

    OpParam vropeparam;
    vropeparam["type"] = VISIONROPE;
    vropeparam["dim"] = 2;
    vropeparam["spatial_merge_size"] = 2;
    auto vrope = cpuBackend->opCreate(vropeparam, "calrope", 4);

    vrope->reshape({image_size}, {pos_id});
    image_size->printShape();
    pos_id->printShape();
    vrope->setUp({image_size}, {pos_id});
    vrope->execute({image_size}, {pos_id});

    pos_id->printData<float>();


    auto image_input = std::make_shared<Tensor>(cpuBackend);
    image_input->reshape(1, 1, 4, 4);
    image_input->setDtype(MLLM_TYPE_F32);
    image_input->setName("image_input");
    image_input->alloc();
    image_input->fullData(1.f);
    auto image_output = std::make_shared<Tensor>(cpuBackend);
    image_output->reshape(1, 1, 4, 4);
    image_output->setDtype(MLLM_TYPE_F32);
    image_output->setName("image_output");
    image_output->alloc();

    auto apply_rope = cpuBackend->funcCreate(mllm::FUNC_APPLY_VISIOROPE);
    apply_rope->setUp({image_output}, {image_input, pos_id}, {});
    apply_rope->execute({image_output}, {image_input,  pos_id}, {});
    image_output->printData<float>();

    auto qnnBackend = Context::Instance().globalBackends<QNNBackend>(MLLM_QNN);
    auto qnnInput = std::make_shared<Tensor>(qnnBackend);
    qnnInput->reshape(1, 1, 4, 4);
    qnnInput->setDtype(MLLM_TYPE_F32);
    qnnInput->setName("qnn_input");
    qnnInput->alloc();
    qnnInput->fullData(1.f);

    auto qnnSinInput = std::make_shared<Tensor>(qnnBackend);
    qnnSinInput->reshape(1, 1, 8, 2);
    qnnSinInput->setDtype(MLLM_TYPE_F32);
    qnnSinInput->setName("qnn_sin_input");
    qnnSinInput->alloc();
    qnnSinInput->fullData(1.f);

    auto qnnCosInput = std::make_shared<Tensor>(qnnBackend);
    qnnCosInput->reshape(1, 1, 8, 2);
    qnnCosInput->setDtype(MLLM_TYPE_F32);
    qnnCosInput->setName("qnn_cos_input");
    qnnCosInput->alloc();
    qnnCosInput->fullData(1.f);

    // sin rope
    OpParam sinParam;
    sinParam["type"] = VISIONROPESIN;
    sinParam["dim"] = 2;
    sinParam["spatial_merge_size"] = 2;
    auto ropeSin = cpuBackend->opCreate(sinParam, "calSin", 4);
    ropeSin->reshape({image_size}, {qnnSinInput});
    ropeSin->setUp({image_size}, {qnnSinInput});
    ropeSin->execute({image_size}, {qnnSinInput});

    // cos rope
    OpParam cosParam;
    cosParam["type"] = VISIONROPECOS;
    cosParam["dim"] = 2;
    cosParam["spatial_merge_size"] = 2;
    auto ropeCos = cpuBackend->opCreate(cosParam, "calCos", 4);

    ropeCos->reshape({image_size}, {qnnCosInput});
    image_size->printShape();
    qnnCosInput->printShape();
    ropeCos->setUp({image_size}, {qnnCosInput});
    ropeCos->execute({image_size}, {qnnCosInput});

    auto qnn_rope_output = std::make_shared<Tensor>(qnnBackend);
    qnn_rope_output->reshape(1, 1, 4, 4);
    qnn_rope_output->setDtype(MLLM_TYPE_F32);
    qnn_rope_output->setName("qnn_output");
    qnn_rope_output->alloc();
    qnn_rope_output->setTtype(GRAPH_OUTPUT);

    // qnn build graph
    vector<shared_ptr<Tensor>> inputs = {qnnInput, qnnSinInput, qnnCosInput}, outputs{qnn_rope_output};
    qnnBackend->onSetUpStart(inputs, outputs, "test_graph");
    std::cout << "------after setup" << std::endl;

    OpParam param;
    param["type"] = ROPESIMPLE;

    auto qnn_rope_op = qnnBackend->opCreate(param, "rope");

    qnn_rope_op->reshape(inputs, outputs);
    qnn_rope_op->setUp(inputs, outputs);

    qnnBackend->pushOutputBuffers(qnn_rope_output->hostPtr<uint8_t>());

    qnnBackend->onSetUpEnd(inputs, outputs, "test_graph");
    qnnBackend->onExecuteStart(inputs, outputs);

    qnn_rope_output->printData<float>();

    return 0;
}