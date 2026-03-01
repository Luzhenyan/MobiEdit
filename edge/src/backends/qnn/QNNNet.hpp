#ifndef MLLM_QNNNET_H
#define MLLM_QNNNET_H

#include "Types.hpp"
#include "legacy/Net.hpp"
#include "express/ExpressBase.hpp"

namespace mllm {
class QNNNet : public Net {
public:
    explicit QNNNet(BackendConfig config, express::Context *ctx);
    virtual ~QNNNet() = default;

    virtual void convert(vector<express::NetParameter> &param, BackendType backend_type = BackendType::MLLM_QNN, int threadCount = 4) override {};
    virtual void convert(express::Context* ctx, BackendType backend_type = BackendType::MLLM_QNN, int threadCount = 4);

private:
    express::Context *ctx_;
};

} // namespace mllm

#endif // MLLM_NET_H