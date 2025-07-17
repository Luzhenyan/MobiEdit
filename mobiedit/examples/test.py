from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型本地路径
# model_name = "../../models--Qwen--Qwen2.5-3B-Instruct/snapshots/Qwen2.5-3B-Instruct"
model_name = "../../models--meta-llama--Llama-3.2-3B-Instruct/snapshots/Llama3.2-3B-Instruct"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32).eval()

# 输入问题
question = "What notable cultural institution was located in Shanghai?"

# 对于Qwen的Instruct模型，通常建议添加指令（可选，如未定制训练流程可省略prompt标记）
prompt = f"<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成答案
with torch.no_grad():
    output = model.generate(
        **inputs, 
        max_new_tokens=64,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

# 解码答案
answer = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("Q:", question)
print("A:", answer.strip())