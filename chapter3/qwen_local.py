"""第三章 3.2.3 练习：在本地调用开源大语言模型 Qwen1.5-0.5B-Chat。

参考教程 code/chapter3/Qwen.py，针对 Apple Silicon Mac 调整了设备选择。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型 ID（约 5 亿参数的小型对话模型，适合本地跑）
model_id = "Qwen/Qwen1.5-0.5B-Chat"

# 设备选择：Mac 没有 CUDA，但 Apple Silicon 可用 mps 加速，否则退回 cpu
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# 加载分词器和模型（首次运行会自动从 Hugging Face 下载，约 1GB）
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
print("模型和分词器加载完成！")

# 准备对话输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请介绍你自己。"},
]

# 用分词器的对话模板格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# 编码为模型输入（Token ID），并移到目标设备
model_inputs = tokenizer([text], return_tensors="pt").to(device)
print("编码后的输入文本:")
print(model_inputs)

# 生成回答，max_new_tokens 控制最多生成多少新 Token
generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    max_new_tokens=512,
)

# 截掉输入部分，只保留新生成的 Token
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码回人类可读文本
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n模型的回答:")
print(response)
