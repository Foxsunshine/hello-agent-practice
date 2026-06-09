import ast
from react_agent import HelloAgentsLLM


# ═══════════════════════════════════════════════════════════
# 提示词模板
# ═══════════════════════════════════════════════════════════

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

EXECUTOR_PROMPT_TEMPLATE = """
你是一个AI执行专家。你的任务是根据给定的计划逐步解决用户的问题。
你需要根据当前的步骤和历史记录来执行任务。

原始问题: {question}

完整计划:
{plan}

历史执行记录:
{history}

当前需要执行的步骤: {current_step}

请根据以上信息，执行当前步骤并给出结果。如果这是最后一步，请给出最终的完整答案。
"""


# ═══════════════════════════════════════════════════════════
# 1. 规划器
# ═══════════════════════════════════════════════════════════

class Planner:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """根据用户问题生成一个行动计划。"""
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""

        print(f"✅ 计划已生成:\n{response_text}")

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


# ═══════════════════════════════════════════════════════════
# 2. 执行器
# ═══════════════════════════════════════════════════════════

class Executor:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """根据计划逐步执行并解决问题。"""
        history = ""
        response_text = ""

        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan="\n".join(f"{j+1}. {s}" for j, s in enumerate(plan)),
                history=history if history else "无",
                current_step=step,
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages) or ""

            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

        return response_text


# ═══════════════════════════════════════════════════════════
# 3. Plan-and-Solve Agent
# ═══════════════════════════════════════════════════════════

class PlanAndSolveAgent:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str) -> str | None:
        """先规划，后执行。"""
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        plan = self.planner.plan(question)

        if not plan:
            print("\n--- 任务终止 --- 无法生成有效的行动计划。")
            return None

        final_answer = self.executor.execute(question, plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        return final_answer


# ═══════════════════════════════════════════════════════════
# 4. 启动入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    llm = HelloAgentsLLM()
    agent = PlanAndSolveAgent(llm)
    agent.run("一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？")
