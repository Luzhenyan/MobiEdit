# WISE配置
method = "WISE"
model_name = ""
model_class = ""
tokenizer_class = ""
tokenizer_name = ""
cls_name = ""
cls_class = ""

# 训练参数
batch_size = 1
max_length = 128
n_iter = 100
edit_lr = 0.1
norm_constraint = 0.5

# 激活值约束参数
alpha = 0.1
beta = 0.1
gamma = 0.1
mask_ratio = 0.5

# 记忆重放参数
retrieve = True
replay = True
save_freq = 10
merge_freq = 50
merge_alg = 'slerp'

# 零阶优化参数
use_zo = False  # 是否使用零阶优化
zo_eps = 1e-5   # 零阶优化的扰动大小 