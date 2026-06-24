import os
import asyncio
import time
from typing import List, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
import json
from config import Config

# 并发限制
MAX_BATCH_WORKERS = 8      # 批量摘要最大并行数
MAX_COMPARE_DOCS = 10      # 对比分析最大文档数


# ── 评估器结构化输出模型 ──
class ErrorItem(BaseModel):
    field: str
    wrong: str
    correct: str
    reason: str


class LLMProcessor:
    def __init__(self):
        self.llm = self._init_llm()
        self.agent_chain = None

    @staticmethod
    def _is_noise_correction(err: "ErrorItem") -> bool:
        """判断一条修正是否为 OCR 噪声导致的伪修正。

        典型：wrong='S323047090'（纯字母数字）→ correct='S3平23047090'（混入汉字）。
        规则：若 wrong 是合法的字母数字串（允许 -/.:），
        而 correct 在其中插入了 CJK 字符，则判定为噪声，丢弃。
        """
        import re as _re
        wrong = (err.wrong or "").strip()
        correct = (err.correct or "").strip()
        if not wrong or not correct:
            return False

        cjk = r'[\u4e00-\u9fff]'
        is_id_like = bool(_re.fullmatch(r'[A-Za-z0-9\-/.:]+', wrong))
        correct_has_cjk = bool(_re.search(cjk, correct))

        # wrong 像编号/学号，但 correct 却混入汉字 → 噪声修正
        if is_id_like and correct_has_cjk:
            return True

        # 进一步：去掉 CJK 后两者相等，说明 correct 只是多塞了汉字
        if _re.sub(cjk, '', correct) == wrong and correct_has_cjk:
            return True

        return False
    
    def _init_llm(self):
        model_type = Config.LLM_MODEL
        
        if model_type.startswith("gpt"):
            if not Config.OPENAI_API_KEY:
                raise ValueError("使用OpenAI模型需要设置OPENAI_API_KEY环境变量")
            return ChatOpenAI(
                model=model_type,
                temperature=0.3,
                api_key=Config.OPENAI_API_KEY
            )
        elif model_type.startswith("deepseek"):
            if not Config.DEEPSEEK_API_KEY:
                raise ValueError("使用DeepSeek模型需要设置DEEPSEEK_API_KEY环境变量")
            try:
                from langchain_deepseek import ChatDeepSeek
                return ChatDeepSeek(
                    model=model_type,
                    temperature=0.3,
                    api_key=Config.DEEPSEEK_API_KEY
                )
            except ImportError:
                raise ValueError("需要安装 langchain-deepseek 包以使用 DeepSeek 模型")
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _init_agent_chain(self, tools: List[BaseTool]):
        """初始化工具调用代理链（自动模式）"""
        try:
            from langchain.agents import create_tool_calling_agent
            from langchain.agents import AgentExecutor
            
            # 创建工具调用提示词
            system_prompt = f"""
你是一个智能助手，可以使用以下工具：

{[tool.name + ": " + tool.description for tool in tools]}

请根据用户的问题，决定是否需要使用工具：
1. 如果问题需要计算、日期处理或文件保存，请调用相应的工具
2. 如果问题可以直接回答，请直接回答，不需要调用工具

请按照以下格式输出：
<task_type>
<content>

task_type 可以是：
- DIRECT_ANSWER: 直接回答问题
- TOOL_CALL: 需要调用工具

如果需要调用工具，请在 content 中指定工具名称和参数。
"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # 创建工具调用 Agent
            agent = create_tool_calling_agent(self.llm, tools, prompt)
            
            # 创建 Agent 执行器
            self.agent_chain = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True
            )
        except Exception as e:
            raise ValueError(f"Agent 初始化失败: {str(e)}")
    
    def generate_summary(self, content: str) -> str:
        """串行模式：一个 Prompt 提取所有内容"""
        prompt = Config.SUMMARY_PROMPT.format(content=content)
        messages = [HumanMessage(content=prompt)]
        result = self.llm.invoke(messages)
        return self.evaluate_and_refine(content, result.content)
    
    def generate_summary_parallel(self, content: str) -> str:
        """并行模式：4 个独立 Prompt 并发调用，最后合并结果"""
        return asyncio.run(self._generate_summary_parallel_async(content))
    
    async def _generate_summary_parallel_async(self, content: str) -> str:
        """异步并行提取 4 类信息"""
        async def extract_one(name: str, prompt_template: str) -> str:
            prompt = prompt_template.format(content=content)
            messages = [HumanMessage(content=prompt)]
            result = await self.llm.ainvoke(messages)
            return result.content
        
        # 4 个任务并发执行
        tasks = [
            extract_one(name, Config.PARALLEL_PROMPTS[name])
            for name in ["核心摘要", "关键条款", "截止时间", "待办事项"]
        ]
        results = await asyncio.gather(*tasks)
        return "\n\n".join(results)
    
    def generate_summary_parallel_sync(self, content: str) -> str:
        """同步版本：使用 threading 实现并行（兼容不支持 asyncio 的场景）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def extract_one(name: str) -> str:
            prompt = Config.PARALLEL_PROMPTS[name].format(content=content)
            messages = [HumanMessage(content=prompt)]
            result = self.llm.invoke(messages)
            return result.content
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(extract_one, name): name
                for name in ["核心摘要", "关键条款", "截止时间", "待办事项"]
            }
            # 按提交顺序收集结果
            ordered_results = {}
            for future in as_completed(futures):
                name = futures[future]
                ordered_results[name] = future.result()
        
        # 按固定顺序拼接
        combined = "\n\n".join(
            ordered_results[name]
            for name in ["核心摘要", "关键条款", "截止时间", "待办事项"]
        )
        # 评估器-优化器：检查并修正事实错误（如学号、金额、日期等数字串）
        return self.evaluate_and_refine(content, combined)
    
    def evaluate_and_refine(self, content: str, summary: str,
                             eval_prompt_template: str = None,
                             opt_prompt_template: str = None) -> str:
        """评估器-优化器模式：检查并修正事实错误（可跨分析类型复用）。

        Args:
            content: 原始文档全文
            summary: 待校验的分析结果（摘要/对比/问答等）
            eval_prompt_template: 评估 Prompt 模板（默认 Config.EVALUATOR_PROMPT）
            opt_prompt_template: 优化 Prompt 模板（默认 Config.OPTIMIZER_PROMPT）

        流程：
          评估器 LLM → JSON → ErrorItem 对象（JSON 即刻丢弃）
          → 优化器 LLM（收自然语言错误描述，不收 JSON）
          → 干净文本

        关键：JSON 字符串在 json.loads 后从不存在于任何文本管道中，
        路径 A（优化器输出含 JSON）和路径 B（Agent 复述 JSON）同时消灭。
        """
        self._eval_details_html = None

        # 默认使用摘要专用 Prompt，调用方可覆盖为对比专用
        _eval_prompt = eval_prompt_template or Config.EVALUATOR_PROMPT
        _opt_prompt = opt_prompt_template or Config.OPTIMIZER_PROMPT

        try:
            # ═══ Step 1: 评估器 → JSON → Python 对象 ═══
            eval_prompt = _eval_prompt.format(
                content=content[:4000],
                summary=summary,
            )
            raw = self.llm.invoke([HumanMessage(content=eval_prompt)])
            eval_text = raw.content.strip()

            if "```" in eval_text:
                eval_text = eval_text.split("```")[1]
                if eval_text.startswith("json"):
                    eval_text = eval_text[4:]
            eval_text = eval_text.strip()

            try:
                eval_data = json.loads(eval_text)
            except json.JSONDecodeError:
                return summary + "\n\n> ℹ️ 评估器未能完成校验，结果可能包含误差，请人工复核。"

            errors = [
                ErrorItem(
                    field=e.get("field", ""),
                    wrong=e.get("wrong", ""),
                    correct=e.get("correct", ""),
                    reason=e.get("reason", ""),
                )
                for e in eval_data.get("errors", [])
            ]

            # ── 噪声防护：过滤"把汉字塞进结构化字段"的伪修正 ──
            errors = [e for e in errors if not self._is_noise_correction(e)]

            has_errors = eval_data.get("has_errors", False) and len(errors) > 0

            if not has_errors:
                return summary + "\n\n> ✅ 评估器核对通过，未发现事实错误。"

            # ═══ Step 2: 优化器 LLM → 智能重写 ═══
            # 用 ErrorItem 对象格式化自然语言错误清单（不是 JSON！）
            errors_text = "\n".join(
                f"- {e.field}: 错误值「{e.wrong}」→ 正确值「{e.correct}」（{e.reason}）"
                for e in errors
            )

            opt_prompt = _opt_prompt.format(
                content=content[:4000],
                summary=summary,
                errors=errors_text,
            )
            opt_raw = self.llm.invoke([HumanMessage(content=opt_prompt)])
            corrected = opt_raw.content.strip()

            # 删除优化器可能的开场白
            for marker in ("##", "【", "核心摘要", "关键条款", "📋"):
                idx = corrected.find(marker)
                if idx > 0:
                    preamble = corrected[:idx].strip()
                    # 是口水话才切（不是真正的内容）
                    if any(preamble.startswith(w) for w in ("好的", "作为", "以下", "我已", "根据", "明白")):
                        corrected = corrected[idx:].strip()
                        break

            # ═══ Step 2.5: 清理优化器输出 ═══
            # a) 移除可能泄漏的 JSON 块（平衡括号法，与 app_streamlit.strip_eval_json 一致）
            import re as _re
            while True:
                key = corrected.find('"has_errors"')
                if key < 0:
                    break
                start = corrected.rfind('{', 0, key)
                if start < 0:
                    break
                depth = 0
                end = start
                for i, ch in enumerate(corrected[start:], start):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end <= start:
                    break
                corrected = corrected[:start] + corrected[end:]
            corrected = _re.sub(r'\n\s*\n\s*\n+', '\n\n', corrected)

            # b) 确保 ## 前有换行（token 级流式下 ## 若紧跟文字会失效）
            corrected = _re.sub(r'(?<!\n)(#{1,6}\s)', r'\n\n\1', corrected)

            # ═══ Step 3: 修正详情 HTML（旁路展示，不进文本流）═══
            detail_rows = "\n".join(
                f"| {e.field} | {e.wrong} | {e.correct} | {e.reason} |"
                for e in errors
            )
            self._eval_details_html = (
                "\n\n<details>\n"
                "<summary>📋 展开修正详情（{} 处）</summary>\n\n"
                "| 字段 | 错误值 | 修正值 | 原因 |\n"
                "|------|--------|--------|------|\n"
                "{}\n"
                "</details>"
            ).format(len(errors), detail_rows)

            return corrected + "\n\n__EVAL_DETAILS__"

        except Exception as e:
            return summary + f"\n\n> ⚠️ 评估器运行异常（{str(e)}），结果未经校验，请人工复核。"
    
    def compare_documents(self, documents: List[dict]) -> str:
        total = len(documents)
        warning = ""
        
        if total > MAX_COMPARE_DOCS:
            warning = (f"\n\n> ⚠️ **注意**：共上传 {total} 个文档，超过建议上限（{MAX_COMPARE_DOCS} 个），"
                       f"仅对比前 {MAX_COMPARE_DOCS} 个文档的内容。\n")
            documents = documents[:MAX_COMPARE_DOCS]
        
        doc_list = ""
        all_full_text = ""
        for i, doc in enumerate(documents, 1):
            content = doc['content'][:2000]
            doc_list += f"\n【文档{i}】{doc['name']}\n{content}{'...' if len(doc['content']) > 2000 else ''}"
            all_full_text += f"\n=== 文档{i}: {doc['name']} ===\n{doc['content'][:4000]}\n"
        
        prompt = Config.COMPARE_PROMPT.format(documents=doc_list)
        messages = [HumanMessage(content=prompt)]
        result = self.llm.invoke(messages)
        comparison = warning + result.content
        # 评估器-优化器：用对比专用 Prompt 校验跨文档事实
        return self.evaluate_and_refine(
            all_full_text,
            comparison,
            eval_prompt_template=Config.COMPARE_EVALUATOR_PROMPT,
            opt_prompt_template=Config.COMPARE_OPTIMIZER_PROMPT,
        )
    
    def summarize_batch(self, contents: List[str], filenames: List[str]) -> List[str]:
        results = []
        for content, filename in zip(contents, filenames):
            try:
                summary = self.generate_summary(content)
                results.append({"filename": filename, "summary": summary})
            except Exception as e:
                results.append({"filename": filename, "summary": f"处理失败: {str(e)}"})
        return results
    
    def summarize_batch_parallel(self, contents: List[str], filenames: List[str]) -> List[dict]:
        """协调器-工作者模式：多个文档并行生成摘要"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def worker(idx: int, content: str, filename: str) -> dict:
            """工作者：处理单个文档（内部用 4 路并行提取）"""
            try:
                summary = self.generate_summary_parallel_sync(content)
                return {"idx": idx, "filename": filename, "summary": summary, "error": None}
            except Exception as e:
                return {"idx": idx, "filename": filename, "summary": "", "error": str(e)}
        
        n = len(contents)
        workers = min(n, MAX_BATCH_WORKERS)  # 限制最大并行数，避免线程爆炸
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(worker, i, contents[i], filenames[i]): i
                for i in range(n)
            }
            results = [None] * n
            for future in as_completed(futures):
                r = future.result()
                results[r["idx"]] = r
        
        return results
    
    def run_with_tools(self, query: str, tools: List[BaseTool]) -> str:
        """使用 Agent 自动调用工具执行查询"""
        try:
            # 确保 Agent 链已初始化
            if not self.agent_chain:
                self._init_agent_chain(tools)
            
            # 使用 Agent 执行查询
            result = self.agent_chain.invoke({"input": query})
            
            # 返回结果
            if isinstance(result, dict) and "output" in result:
                return result["output"]
            return str(result)
        
        except Exception as e:
            # 如果 Agent 调用失败，返回原始错误信息
            return f"工具调用失败: {str(e)}"
