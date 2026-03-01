//
// Created by Rongjie Yi on 2024/1/28 0028.
//

#ifndef MODULE_HPP
#define MODULE_HPP
#include "Context.hpp"
#include "Generate.hpp"
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"
#include "Timing.hpp"
#include "Trace.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include <any>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory/SystemMemoryManager.hpp>
#include <memory/MemoryPoolManager.hpp>
#include <memory>
#include <ostream>
#include <stack>
#include <utility>
#include <vector>
#include <unordered_map>

namespace mllm {

namespace utils {
// get the closest factors of a number, used in NPU part2 view to speed up the QNN linear
inline std::pair<int, int> closestFactors(int n) {
    int root = static_cast<int>(sqrt(n));
    for (int i = root; i > 0; --i) {
        if (n % i == 0) {
            return {i, n / i};
        }
    }
    return {1, n};
}
} // namespace utils

class Module {
protected:
    std::shared_ptr<LlmTextGenerator> text_generator_ = nullptr;

public:
    double load_time_;
    int prefilling_token_size_ = 0;
    int decoding_token_size_ = 0;
    vector<double> inference_times_;

    map<string, shared_ptr<Tensor>> activation_tensors;
    map<string, int> activation_tensors_num;
    AbstructLoader *loader;
    bool doLoad = false;
    bool op_transposed_flag = false;

    static Module *llm_model_ptr;

    static int listIdx;
    static std::stack<int> listIdxStack;
    // static int runlistIdx;

    static BackendType tmp_device;

    // static std::unordered_map<string, shared_ptr<Op>> tensor_func_ops; // use for QNN

private:
    template <typename... Args>
    vector<std::any> convertArgsToAnyVector(Args... args) {
        return vector<std::any>{std::any(args)...};
    }

    // 递归终止函数
    template <typename T>
    static auto change_last(T value) {
        return std::make_tuple(value + std::to_string(listIdx) + ".");
    }
    // 递归函数
    template <typename T, typename... Args>
    static auto change_last(T head, Args... tail) {
        auto tail_tuple = change_last(tail...);
        return std::tuple_cat(std::make_tuple(head), tail_tuple);
    }

public:
    Module() {}
    virtual ~Module() = default;

    void load(string path) {
        loader = new ParamLoader(std::move(path), true); // TODO mmap
        load(*loader);
    }
    void load(AbstructLoader &param_loader) {
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        mllm_time_init();

        loader = &param_loader;
        doLoad = true;
        vector<Tensor> tmps;
        int max_in_size = 5;
        for (int i = 0; i < max_in_size; ++i) {
            Tensor t(Context::Instance().globalBackends(MLLM_CPU));
            t.setName("input" + std::to_string(i));
            t.reshape(1, 1, 1, 10);
            t.alloc();
            t.setModule(this);
            tmps.push_back(t);
        }
        llm_model_ptr = this;
        vector<std::any> alternate_args = {
            {},
            vector<int>{0, 0},
            std::vector<std::vector<int>>(32, std::vector<int>(2))};
        uint64_t time_start = 0;
        for (auto args : alternate_args) {
            time_start = mllm_time_us();
            try {
                operator()(tmps, args);
                break;
            } catch (const std::exception &e) {
#if not defined(__ARM_NEON)
                if (std::string("bad any_cast") != e.what()) {
                    MLLM_LOG_ERROR_STREAM << e.what() << std::endl;
                    exit(0);
                }
#endif
            } catch (...) {
                MLLM_LOG_ERROR_STREAM << "load error" << std::endl;
                exit(0);
            }
        }
        uint64_t time_end = mllm_time_us();
        load_time_ = (time_end - time_start) / 1000.0F; // ms
        doLoad = false;
    }

    virtual vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) = 0;

    template <typename... Args>
    vector<Tensor> operator()(vector<Tensor> inputs, Args... args) {
        vector<std::any> anyArgs = convertArgsToAnyVector(args...);
        auto backend = inputs.empty() ? Context::Instance().globalBackends(MLLM_CPU) : inputs[0].backend();
        // TODO: multi backend dispatch
        if (Context::Instance().globalBackends(MLLM_QNN) != nullptr) {
            backend = Context::Instance().globalBackends(MLLM_QNN);
        }
        return backend->runForward(this, inputs, anyArgs);
    }

    template <typename T, typename... Args>
    static vector<T> List(int n, Args &&...args) {
        static_assert(std::is_base_of<Module, T>::value, "T must be a subclass of Module");
        if (listIdx) {
            listIdxStack.push(listIdx);
        }
        listIdx = 0;
        vector<T> modules;
        for (int i = 0; i < n; i++) {
            auto new_args = change_last(args...);
            modules.push_back(std::move(T(std::apply([&](auto &&...args) { return T(std::forward<decltype(args)>(args)...); }, new_args))));
            listIdx++;
        }
        if (!listIdxStack.empty()) {
            listIdx = listIdxStack.top();
            listIdxStack.pop();
        } else {
            listIdx = 0;
        }
        return modules;
    }

    void free() {
        activation_tensors.clear();
    }

    void setNoLoadWeightsDtype(DataType dtype) {
        llm_model_ptr = this;
        Op::noLoadWeightsDtype() = dtype;
    }
    virtual void clear_kvcache() {
        ;
    }
    vector<double> profiling(string name = "");
    virtual void generate(
        Tensor &input_ids, const LlmTextGeneratorOpts &opt, const std::function<bool(unsigned int)> &call_back = [](unsigned int) -> bool { return true; });

    vector<unsigned> generate(Tensor &input_ids, const LlmTextGeneratorOpts &opt, int end_token = -1);
};

class CPUModuleWrapper : public Module {
public:
    vector<shared_ptr<Callable>> traces_;

    void addOp(Op *op,
               vector<shared_ptr<Tensor>> inputs,
               vector<shared_ptr<Tensor>> outputs) {
        auto callable = std::make_shared<Callable>(CallableType::OP);
        callable->opInputs = inputs;
        callable->opOutputs = outputs;
        callable->op = op;
        traces_.push_back(callable);
    }

    void addTensorFunction(TensorFunction *func,
                           vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs, vector<float> args) {
        auto callable = std::make_shared<Callable>(CallableType::TENSOR_FUNC);
        callable->tensorFunc = func;
        callable->tensorInputs = inputs;
        callable->tensorOutputs = outputs;
        for (auto arg : args) {
            callable->args.push_back(arg);
        }
        traces_.push_back(callable);
    }

    virtual vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        // get chunk_id from args
        int chunk_id = std::any_cast<int>(args[0]);
        if (chunk_id != 0) {
            for (auto &callable : traces_) {
                callable->reshape();
                callable->setUp();
            }
        }

        for (int i = 0; i < traces_.size(); i++) {
            traces_[i]->execute();
        }
        return {};
    }

    vector<shared_ptr<Tensor>> result() {
        return traces_.back()->outputs();
    }
};

class QNNModuleWrapper : public Module {
public:
    string name_;
    vector<shared_ptr<Tensor>> inputs_;
    vector<shared_ptr<Tensor>> outputs_;

    virtual vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        Context::Instance().globalBackends(MLLM_QNN)->onExecuteStart(inputs_, outputs_, name_);
        Context::Instance().globalBackends(MLLM_QNN)->onExecuteEnd(outputs_, name_);
        return {};
    }
};

} // namespace mllm

#endif // MODULE_HPP
