# Zero Order Optimizer

## main函数逻辑

`examples/demo_qwen_rome_fwd.cpp` 为可执行程序，使用qwen 2.5 1.5B, billion参数为1.5B-rotated。

60：92行为prompt处理逻辑，tokenizer处理后比较seq长度进行匹配。

104：105设置Switching全局状态，用于CPU attenntion部分每次推理时更新rope和kvcache的sequence状态（一直保持1*chunk大小）。**在训练前，进行一次推理，将指定位置的扰动输入注册进ZeroOrderOptimizer**

ZeroOrderOptimizer包含随机扰动的生成和扰动向量（input tensor）的维护。扰动通过注册进ZeroOrderOptimizer::weights_to_optimize的Tensor，作为QNN的输入，进行Tensor加法实现。ZeroOrderOptimizer::all_layer_perts含义为 `层-group-扰动`。调用optimizer.applyPerturbation时，会将ZeroOrderOptimizer::all_layer_perts按照index加/减到注册的Tensor上，因此需要再调用optimizer.removePerturbation。

119：132以及172：191进行QNN上的decoding，src/models/qwen/modeling_qwen_npu_rome.hpp中根据传入的LlmTextGeneratorOpts.seq_before_padding进行生成。

137：170实现了零阶优化器的训练过程，对每个训练步，进行group_size次随机扰动。

NOTE: 参考demo_qwen_rome_fwd.cpp:135行，如果model的input_tensor发生更新，需要重新进行input_tensor.setTtype(INPUT_TENSOR)，否则model不会更新之前的input tensor指针。

## loss计算和更新逻辑

执行group_size次随机扰动，并对loss_minus和loss_plus求和，再取平均值。更新函数实现了cos学习率，更新直接加在注册的input tensor上。