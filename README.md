# Hello-Agents 练习

跟随 [Datawhale Hello-Agents](https://github.com/datawhalechina/hello-agents) 教程做的个人练习。

教程原仓库在 `~/Desktop/hello-agent/hello-agents`（只读参考，不在此提交）。本仓库只保存我自己写的练习代码。

## 进度

- [ ] 第一章 初识智能体 —— ReAct 范式旅行助手 (`chapter1/`)
- [x] 第三章 3.2.3 本地调用开源大语言模型 —— Qwen1.5-0.5B-Chat (`chapter3/`)

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai requests tavily-python python-dotenv
cp .env.example .env   # 然后填入你自己的密钥
```

## 运行

```bash
python chapter1/first_agent.py
```
