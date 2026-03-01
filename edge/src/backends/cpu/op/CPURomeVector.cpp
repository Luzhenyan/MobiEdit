
#include "CPURomeVector.hpp"
#include <cstddef>
#include <iostream>
#include "ZeroOrderOptimize.hpp"

namespace mllm {

CPURomeVector::CPURomeVector(Backend *bn, string opName, int batch, int head, int seq, int dim, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    batch_ = batch;
    head_ = head;
    seq_ = seq;
    dim_ = dim;
    romeTensor.setBackend(bn);
}

ErrorCode CPURomeVector::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (outputs[0]->masterTensor() == nullptr) {
        outputs[0]->shallowCopyFrom(&romeTensor, false);
    }

    outputs[0]->reshape(batch_, head_, seq_, dim_);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPURomeVector::load(AbstructLoader &loader) {
    romeTensor.setName(name());
    romeTensor.reshape(batch_, head_, seq_, dim_);
    romeTensor.alloc();

    optim::ZeroOrderOptimizer::registerWeight(romeTensor);

    return Op::load(loader);
}

ErrorCode CPURomeVector::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (outputs[0]->masterTensor()->name() != romeTensor.name()) {
        if (outputs[0]->masterTensor() == nullptr) {
            for (int n = 0; n < outputs[0]->batch(); ++n) {
                for (int c = 0; c < outputs[0]->head(); ++c) {
                    for (int h = 0; h < outputs[0]->sequence(); ++h) {
                        for (int w = 0; w < outputs[0]->dimension(); ++w) {
                            outputs[0]->setDataAt<float>(n, c, h, w, romeTensor.dataAt<float>(n, c, h, w));
                        }
                    }
                }
            }
        } else {
            if (romeTensor.batch() == 1) {
                auto off = outputs[0]->shapeOffset();
                auto off_b = off[0];
                auto off_h = off[1];
                auto off_s_ = off[2];
                auto off_d = off[3];
                for (int n = 0; n < outputs[0]->masterTensor()->batch(); ++n) {
                    for (int c = 0; c < outputs[0]->head(); ++c) {
                        for (int h = 0; h < outputs[0]->sequence(); ++h) {
                            for (int w = 0; w < outputs[0]->dimension(); ++w) {
                                outputs[0]->masterTensor()->setDataAt<float>(n + off_b, c + off_h, h + off_s_, w + off_d, romeTensor.dataAt<float>(0, c, h, w));
                            }
                        }
                    }
                }
            }
        }
    }

    return Op::execute(inputs, outputs);
}

ErrorCode CPURomeVector::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    romeTensor.free();
    return Op::free(inputs, outputs);
}

ErrorCode CPURomeVector::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->shallowCopyFrom(&romeTensor, false);
    return MLLM_NO_ERROR;
}
} // namespace mllm
