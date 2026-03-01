#include "ZeroOrderOptimize.hpp"
#include <random>

namespace mllm {
namespace optim {

vector<Tensor *> ZeroOrderOptimizer::weights_to_optimize;

vector<vector<vector<float>>> ZeroOrderOptimizer::all_layer_perts;

int ZeroOrderOptimizer::group_size = 1; // Default group size for optimization

}
} // namespace mllm::optim