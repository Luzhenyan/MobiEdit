from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path_tokenizer = '/home/u2023040027/llama3-instruction-8b'
path = '/home/u2023040027/EasyEdit/examples/output/edited_model.pth'
# 加载保存的模型和分词器
tokenizer = AutoTokenizer.from_pretrained(path_tokenizer)
# 加载模型结构
model = AutoModelForCausalLM.from_pretrained(path_tokenizer)
# model = AutoModelForCausalLM.from_pretrained(path)
# 加载模型权重
model.load_state_dict(torch.load(path))


# 进入聊天循环
input_text = "hello, "

while True:
    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # 使用模型生成回复
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=150)
    
    # 解码并输出回复
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("模型回复:", response)
    
    # 获取用户的新输入
    input_text = input("你的问题: ")
