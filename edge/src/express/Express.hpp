#ifndef MLLM_EXPRESS_H
#define MLLM_EXPRESS_H

#include "ExpressBase.hpp"
#include "Types.hpp"
#include <string>
#include <vector>
using namespace mllm;
void displayExpress(express::Context *c);

void _SubgraphBegin(express::Context *ctx, BackendType backend = MLLM_CPU);

express::NetTensor *_Input(express::Context *ctx, vector<int> dims = {}, string name = "", DataType type = MLLM_TYPE_F32);
express::NetTensor *_Parameter(express::Context *ctx, std::vector<express::NetTensor *> inputs, int batch, int seq, int head, int dim, string name = "", DataType type = MLLM_TYPE_F32);
express::NetTensor *_Range(express::Context *ctx, std::vector<express::NetTensor *> inputs, int start, int end, string name = "", DataType type = MLLM_TYPE_F32);
express::NetTensor *_Add(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_Causalmask(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_SiLU(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_SuperSiLU(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_Quantize(std::vector<express::NetTensor *> inputs, bool isNSHD = true, string name = "");
express::NetTensor *_Dequantize(std::vector<express::NetTensor *> inputs, bool isNSHD = true, string name = "", bool isFP32 = true);
express::NetTensor *_Softmax(std::vector<express::NetTensor *> inputs, int axis, int do_causal_mask, string name = "");
express::NetTensor *_Matmul(std::vector<express::NetTensor *> inputs, bool transpose0, bool transpose1, string name = "");
express::NetTensor *_RMSNorm(std::vector<express::NetTensor *> inputs, int norm_size, float epsilon = 1e-6, string name = "", bool isFP32 = true);
express::NetTensor *_RoPE(std::vector<express::NetTensor *> inputs, int pose_type, string name = "", int rope_theta = 10000, int max_position_embeddings = 16384);
express::NetTensor *_IRoPE(std::vector<express::NetTensor *> inputs, int pose_type, string name = "", int rope_theta = 10000, int max_position_embeddings = 16384);
express::NetTensor *_QNNRoPE(std::vector<express::NetTensor *> inputs, int pose_type, string name = "", int rope_theta = 10000, int max_position_embeddings = 16384, bool isFP32 = true);
express::NetTensor *_QNNIRoPE(std::vector<express::NetTensor *> inputs, int pose_type, string name = "", int rope_theta = 10000, int max_position_embeddings = 16384, bool isFP32 = true);
express::NetTensor *_PositionalEmbedding(std::vector<express::NetTensor *> inputs, int max_num, int hidden_dim, string name = "");
express::NetTensor *_Scale(std::vector<express::NetTensor *> inputs, float scale, float bias, bool bias_after_scale, string name);
express::NetTensor *_Linear(std::vector<express::NetTensor *> inputs, int in_features, int out_features, bool bias, string name = "");
express::NetTensor *_LinearINT8(std::vector<express::NetTensor *> inputs, int in_features, int out_features, bool bias, string name = "");
vector<express::NetTensor *> _LinearINT8ShadowMerge(std::vector<express::NetTensor *> inputs, int in_features, int out_features, bool bias, string name = "");
express::NetTensor *_LinearINT8ShadowCPU(std::vector<express::NetTensor *> inputs, int in_features, int out_features, int max_position = 1024, bool bias = false, string name = "");
express::NetTensor *_Embedding(std::vector<express::NetTensor *> inputs, int vocab_size, int hidden_size, string name = "");
express::NetTensor *_Mul(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_KVCache(std::vector<express::NetTensor *> inputs, int cache_max, string name = "");
express::NetTensor *_KVCache(std::vector<express::NetTensor *> inputs, int n_rep, int cache_max, string name = "");
express::NetTensor *_KVCacheNPU(std::vector<express::NetTensor *> inputs, int cache_max, string name = "");
express::NetTensor *_KVCache(std::vector<express::NetTensor *> inputs, int n_rep, bool share_input, int cache_max, string name = "");
express::NetTensor *_ReLU(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_ReLUSquaredActivation(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_GELU(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_QuickGELU(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_LayerNorm(std::vector<express::NetTensor *> inputs, int norm_size, bool bias = true, float epsilon = 1e-6, string name = "");
vector<express::NetTensor *> _Split(std::vector<express::NetTensor *> inputs, int split_num, Chl split_dim, int split_dim_size = -1, string name = "");
express::NetTensor *_Gather(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_Convolution2D(std::vector<express::NetTensor *> inputs, int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias = false, string name = "");
express::NetTensor *_Convolution3D(std::vector<express::NetTensor *> inputs, int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias = false, string name = "");
express::NetTensor *_AvgPool2D(std::vector<express::NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name = "");
express::NetTensor *_MaxPool2D(std::vector<express::NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name = "");
express::NetTensor *_Cat(std::vector<express::NetTensor *> inputs, Chl axis, string name = "");
express::NetTensor *_Division(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_Replace(std::vector<express::NetTensor *> inputs, string name = "");
express::NetTensor *_SparseLinear(std::vector<express::NetTensor *> inputs, int in_dim, int out_dim, string name = "");
express::NetTensor *_SparseIdLinear(std::vector<express::NetTensor *> inputs, int in_dim, int out_dim, string name = "");
express::NetTensor *_Predictor(std::vector<express::NetTensor *> inputs, int in_dim, int out_dim, string name = "");
express::NetTensor *_WNop(std::vector<express::NetTensor *> inputs, int sync_type, string name = "");
vector<express::NetTensor *> _MergeOutput(std::vector<express::NetTensor *> inputs, string name = "");
vector<express::NetTensor *> _SplitInput(std::vector<express::NetTensor *> inputs, bool isPrompt, int num, string name = "");
express::NetTensor *_Transpose(std::vector<express::NetTensor *> inputs, std::vector<int> perm, string name = "");

#endif // MLLM_EXPRESS_H