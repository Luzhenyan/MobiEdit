#include "Context.hpp"
#include "ZeroOrderOptimize.hpp"
#include "QNNBackend.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen_npu_rome.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "processor/PostProcess.hpp"
#include <nlohmann/json.hpp>

using namespace mllm;
using json = nlohmann::json;

int main(int argc, char **argv) {
    std::ifstream f("../assets/rome_example.json");
    json sample_data = json::parse(f);

    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen2.5_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen2.5_merges.txt");
    cmdParser.add<string>("qnn-model", 'm', "specify mllm model path", false, "../models/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm");
    cmdParser.add<string>("decoding-model", '\0', "specify mllm model path", false, "../models/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm");
    cmdParser.add<string>("billion", 'b', "[0.5B | 1.8B | 1.5B | [1.5B, 1.8B]-rotated]", false, "1.5B-rotated");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<int>("train-steps", 's', "num of training steps", false, 100);
    cmdParser.add<int>("group-size", 'g', "num of perturbations per steps", false, 5);

    cmdParser.parse_check(argc, argv);

    const int chunk_size = 32; // Set the chunk size for the model

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("qnn-model");
    string decoding_model_path = cmdParser.get<string>("decoding-model");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    int train_steps = cmdParser.get<int>("train-steps");
    int group_size = cmdParser.get<int>("group-size");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");
    mllm::optim::ZeroOrderOptimizer::group_size = group_size;

    auto tokenizer = QWenTokenizer(vocab_path, merge_path);
    QWenNPUConfig config(tokens_limit, "1.5b-rotated", RoPEType::HFHUBROPE);
    auto model = rome_fwd::QWenForCausalLM_NPU(config, chunk_size);

    Context::Instance().initBackend(MLLM_QNN);

    model.load(model_path);
    // auto decoding_model = QWenForCausalLM(config);
    // decoding_model.load(decoding_model_path);

    mllm::optim::ZeroOrderOptimizer optimizer(0.05, 1e-3, train_steps);

    for (int i = 0; i < sample_data.size(); ++i) {
        // auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        std::string prompt = sample_data[i]["prompt"];
        std::string subject_str = sample_data[i]["subject"];
        std::string target_new_str = sample_data[i]["target_new"];
        std::string input_str = prompt + " " + target_new_str;

        std::cout << "[Prompt]: " << prompt << std::endl;
        std::cout << "[Subject]: " << subject_str << std::endl;
        std::cout << "[Target New]: " << target_new_str << std::endl;
        std::cout << "[Input]: " << input_str << std::endl;

        // find substr starting 0 and ending with subject_str in prompt
        if (input_str.find(subject_str) == std::string::npos) {
            std::cerr << "Subject string not found in input string." << std::endl;
        }
        std::string prefix_str = input_str.substr(0, input_str.find(subject_str) + subject_str.length());
        std::cout << "Prefix: " << prefix_str << std::endl;
        auto prefix_token = tokenizer.tokenize(prefix_str, "prefix");

        int edit_idx = prefix_token.sequence() - 1;
        std::cout << "Edit index: " << edit_idx << std::endl;

        optimizer.setVectorIdx(edit_idx);

        auto pre_target_token = tokenizer.tokenize(prompt + " ", "pre_target");
        std::cout << "[Pre-target token]: " << prompt + " " << std::endl;
        int target_start_idx = pre_target_token.sequence();
        std::cout << "[Target start index]: " << target_start_idx << std::endl;

        auto [real_seq_length, input_tensor] = tokenizer.tokenizeWithPadding(input_str, chunk_size, config.vocab_size);
        std::cout << "[real_seq_length]: " << real_seq_length << std::endl;
        int target_end_length = real_seq_length;
        std::cout << "[Target end length]: " << target_end_length << std::endl;

        // generate the target tensor mask
        vector<int> mock_target(chunk_size, -100);
        for (int i = target_start_idx; i < target_end_length; ++i) {
            if (i == 0) continue; // skip the first token to avoid out of bounds error
            // NOTE: LLM performs causal prediction, so the target is the next token
            mock_target[i - 1] = (int)input_tensor.d<float>(0, i, 0, 0);
            std::cout << "Mock target[" << i - 1 << "]: " << mock_target[i - 1] << std::endl;
        }

        // always turn on switching
        Context::Instance().inference_state().toggleSwitching();
        Context::Instance().inference_state().setExecutionType(PROMPT);

        // before training, run a forward to register the weights
        model({input_tensor});

        if (!std::filesystem::exists("qnn_context.bin")) {
            Context::Instance().globalBackends<QNNBackend>(MLLM_QNN)->saveQNNContext();
        }
        // freeze the QNN graph for inference, to avoid repeated tensor registration
        Context::Instance().inference_state().setQnnGraphFrozen(true);

        Context::Instance().inference_state().setCurSequenceLength(0);
        std::cout << "[test before fwd] " << prompt;
        auto [_, origin_input] = tokenizer.tokenizeWithPadding(prompt, chunk_size, config.vocab_size);
        LlmTextGeneratorOpts pre_opt{
            .max_new_tokens = 10,
            .do_sample = false,
            .is_padding = true,
            .seq_before_padding = target_start_idx - 1,
        };
        model.generate(origin_input, pre_opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { return false; }
            std::cout << output_string << std::flush;
            return true;
        });
        std::cout << std::endl;

        // NOTE: set the input tensor type to INPUT_TENSOR for refresh the tensor map
        input_tensor.setTtype(INPUT_TENSOR);

        std::cout << "[Training] steps: " << train_steps << std::endl;
        auto group_size = mllm::optim::ZeroOrderOptimizer::group_size;
        for (int j = 0; j < train_steps; j++) {
            double loss_plus = 0.0, loss_minus = 0.0;
            for (int k = 0; k < group_size; ++k) {
                optimizer.initRandomVector();
                // h-v forward
                // reset sequence length and execution type
                Context::Instance().inference_state().setCurSequenceLength(0);
                optimizer.applyPerturbation(mllm::optim::PERTUR_TYPE::ADD);
                auto result = model({input_tensor});

                loss_plus += optimizer.compute_nll_loss(result[0], mock_target);

                optimizer.removePerturbation(mllm::optim::PERTUR_TYPE::ADD);

                // h+v forward
                // reset sequence length and execution type
                Context::Instance().inference_state().setCurSequenceLength(0);
                optimizer.applyPerturbation(mllm::optim::PERTUR_TYPE::SUB);
                result = model({input_tensor});

                loss_minus += optimizer.compute_nll_loss(result[0], mock_target);

                optimizer.removePerturbation(mllm::optim::PERTUR_TYPE::SUB);

                optimizer.group_idx = (optimizer.group_idx + 1) % mllm::optim::ZeroOrderOptimizer::group_size;
            }

            std::cout << "step " << j << ": loss_plus = " << loss_plus
                      << ", loss_minus = " << loss_minus << ", loss_diff = " << (loss_plus - loss_minus) << std::endl;

            optimizer.mobiedit_zero_order_optimization(loss_plus / group_size, loss_minus / group_size);
        }

        // validate the edit result
        std::cout << "[test] " << prompt;
        Context::Instance().inference_state().setCurSequenceLength(0);
        LlmTextGeneratorOpts opt{
            .max_new_tokens = 10,
            .do_sample = false,
            .is_padding = true,
            .seq_before_padding = target_start_idx - 1,
        };
        for (int i = target_start_idx - 1; i < chunk_size; ++i) {
            input_tensor.setDataAt(0, 0, i, 0, (float)config.vocab_size);
        }
        model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { return false; }
            std::cout << output_string << std::flush;
            return true;
        });
        std::cout << std::endl;
    }
}
