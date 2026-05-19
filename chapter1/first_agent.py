"""第一章练习：ReAct 范式旅行助手 Agent。

参考教程 code/chapter1/FirstAgentTest.py，改为从 .env 读取密钥，
并在启动时校验必需配置。
"""

import os
import re

import requests
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

MAX_LOOPS = 5


def get_weather(city: str) -> str:
    """通过 wttr.in API 查询真实天气。"""
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]
        weather_desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        return f"{city}当前天气：{weather_desc}，气温{temp_c}摄氏度"
    except requests.exceptions.RequestException as e:
        return f"错误：查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """根据城市和天气，用 Tavily 搜索景点推荐。"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置 TAVILY_API_KEY。"
    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        if response.get("answer"):
            return response["answer"]
        results = [f"- {r['title']}: {r['content']}" for r in response.get("results", [])]
        if not results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        return "根据搜索，为您找到以下信息：\n" + "\n".join(results)
    except Exception as e:
        return f"错误：执行 Tavily 搜索时出现问题 - {e}"


available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


class OpenAICompatibleClient:
    """调用任何兼容 OpenAI 接口的 LLM 服务。"""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        print("正在调用大语言模型...")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False
            )
            print("大语言模型响应成功。")
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用 LLM API 时发生错误: {e}")
            return "错误：调用语言模型服务时出错。"


def load_config() -> dict:
    """启动时读取并校验必需配置，缺失则快速失败。"""
    config = {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "base_url": os.environ.get("OPENAI_BASE_URL"),
        "model_id": os.environ.get("MODEL_ID"),
    }
    missing = [k for k, v in config.items() if not v]
    if missing:
        raise SystemExit(
            f"缺少必需的环境变量: {', '.join(missing)}。"
            f"请复制 .env.example 为 .env 并填写。"
        )
    return config


def run_agent(user_prompt: str, llm: OpenAICompatibleClient) -> None:
    prompt_history = [f"用户请求: {user_prompt}"]
    print(f"用户输入: {user_prompt}\n" + "=" * 40)

    for i in range(MAX_LOOPS):
        print(f"--- 循环 {i + 1} ---\n")
        full_prompt = "\n".join(prompt_history)
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

        match = re.search(
            r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
            llm_output,
            re.DOTALL,
        )
        if match and match.group(1).strip() != llm_output.strip():
            llm_output = match.group(1).strip()
            print("已截断多余的 Thought-Action 对")
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            obs = "错误: 未能解析到 Action 字段。请严格遵循 'Thought: ... Action: ...' 格式。"
            print(f"Observation: {obs}\n" + "=" * 40)
            prompt_history.append(f"Observation: {obs}")
            continue

        action_str = action_match.group(1).strip()
        if action_str.startswith("Finish"):
            final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
            print(f"任务完成，最终答案: {final_answer}")
            return

        tool_name = re.search(r"(\w+)\(", action_str).group(1)
        args_str = re.search(r"\((.*)\)", action_str).group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误：未定义的工具 '{tool_name}'"
        print(f"Observation: {observation}\n" + "=" * 40)
        prompt_history.append(f"Observation: {observation}")

    print("已达到最大循环次数，未能完成任务。")


if __name__ == "__main__":
    cfg = load_config()
    client = OpenAICompatibleClient(
        model=cfg["model_id"], api_key=cfg["api_key"], base_url=cfg["base_url"]
    )
    run_agent(
        "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。",
        client,
    )
