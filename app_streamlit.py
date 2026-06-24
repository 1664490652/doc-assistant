# -*- coding: utf-8 -*-
"""办公文档智能助手 - Streamlit 版（ChatGPT 风格）

【本次修复清单】
────────────────────────────────────────────
BUG-1 摘要重复输出（最严重）
  根源：chat_with_agent 把对话历史（含上一轮完整摘要）传给 LLM，
        加上 system prompt "调用工具后原样展示结果"，
        LLM 看到历史摘要就直接复读。
  修复：
    a. 对话历史只传最近 3 轮（6 条消息），由 agent_graph.get_recent_turns() 提供
    b. assistant 消息存入短期记忆前截断到 300 字（agent_graph.append_turn 自动处理）
    c. system prompt 删除"原样展示"规则，改为"不要在工具结果外重复描述"
    d. stream_response 不再自己读写历史，全权交给 agent_graph

BUG-2 thread_id 永远是 "default"
  根源：cur_conv.get("conv_id", "default")，字典键是 "id" 不是 "conv_id"
  修复：统一改为 cur_conv["id"]，封装成 _get_thread_id(cur_conv) 工具函数

BUG-3 长期记忆（Memory Tree）混入聊天流水
  根源：stream_response 把每次完整 AI 回复存进 agent_graph（旧 save_to_memory）
  修复：agent_graph 只管短期记忆轮次；
        Memory Tree 只由 tools.py 的 memory_write 工具在用户明确要求时写入

BUG-4 两套 system prompt 不一致
  根源：_build_system_prompt() 和 chat_with_agent() 内部各有一套，规则互相矛盾
  修复：合并为唯一入口 _build_system_prompt()，chat_with_agent 直接调用
────────────────────────────────────────────
"""
import os
import sys
import json
import logging
import tempfile
import shutil
import traceback
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import streamlit as st
from config import Config

st.set_page_config(
    page_title="文档智能助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOG_FILE = "app_streamlit.log"

# 统一日志：强制把 Python logging 也写入同一个文件
# Streamlit 内部会先调用 basicConfig，所以不能依赖 basicConfig，直接加 handler
_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(LOG_FILE)
           for h in _root.handlers):
    _fh = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
    _fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    _root.addHandler(_fh)

def log_message(msg):
    logging.info(msg)


# ========== PII 脱敏（双层渲染策略）==========

def apply_pii_mask(text: str) -> str:
    """对文本中的邮箱、手机号、身份证号做脱敏处理。

    stream_mode="messages" 的 token 流不会经过 PIIMiddleware（那作用于 state 副本），
    所以在流式结束后做一次正则后处理，确保最终渲染内容不含 PII。

    规则（与 middleware_config.py 的 PII_TYPES 保持一致）：
      - 邮箱 → 完全替换为 [邮箱]
      - 手机号 → 138****5678（保留首3尾4）
      - 身份证 → 110*************1234（保留首3尾4）
    """
    import re

    # 邮箱：完全替换
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[邮箱]', text)

    # 中国手机号：保留首3尾4
    text = re.sub(
        r'(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)',
        r'\1****\3',
        text,
    )

    # 身份证号（18位）：保留首3尾4
    text = re.sub(
        r'(?<!\d)(\d{3})(\d{11})([\dXx]{4})(?!\d)',
        r'\1***********\3',
        text,
    )

    return text


def strip_eval_json(text: str) -> str:
    """移除评估器泄漏的 JSON 块（最终防线）。

    平衡括号法：搜索 "has_errors" 回溯到 {，然后匹配配对 }。
    无论 JSON 来自优化器输出、工具返回值还是 LLM 复述，一律清除。
    """
    import re
    while True:
        key = text.find('"has_errors"')
        if key < 0:
            break
        start = text.rfind('{', 0, key)
        if start < 0:
            break
        depth = 0
        end = start
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end <= start:
            break
        text = text[:start] + text[end:]
    # 可能残留的 JSON 尾逗号空白
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    # 确保 markdown 标题前有换行，避免 "前缀。## 核心摘要" 粘连
    text = re.sub(r'(?<!\n)(#{1,6}\s)', r'\n\n\1', text)
    return text


def _stream_safe_prefix(text: str) -> str:
    """返回不含「未闭合花括号尾部」的前缀，把可能是 JSON 的片段缓冲住。

    Agent 前缀（如"我先为文档1生成结构化摘要。"）之后可能紧跟着
    {"has_errors":...} JSON 块。流式输出时遇到未闭合的 { 就截断，
    等闭合后由 strip_eval_json 删掉再 yield 干净的后缀。
    """
    depth = 0
    start_idx = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0:
                    start_idx = None
    if depth > 0 and start_idx is not None:
        return text[:start_idx]
    return text



def _stream_write_typewriter(placeholder, chunks, speed: float = 0.018):
    """真正的打字机效果：逐字渲染到 st.empty() 占位符。

    小 chunk（≤3字，AI token）直接追加 — LLM 本身逐个 token 产出已有自然节奏。
    大 chunk（工具结果）逐字打字 — 模拟逐字显示效果。

    Args:
        placeholder: st.empty() 返回的占位符
        chunks: stream_response 生成器产生的文本块迭代器
        speed: 每字符延迟秒数，默认 0.018s（约55字/秒）
    Returns:
        full_text: 完整文本
    """
    import time as _twtime
    full = ""
    cursor = "▌"
    for chunk in chunks:
        if chunk is None:
            continue
        s = str(chunk)
        # 小 AI token 直接输出（自带流式节奏），大块工具结果逐字打字
        if len(s) <= 3:
            full += s
            placeholder.markdown(full + cursor)
        else:
            for ch in s:
                full += ch
                placeholder.markdown(full + cursor)
                _twtime.sleep(speed)
    # 完成后去掉光标
    placeholder.markdown(full)
    return full


# ========== 会话状态初始化 ==========

def init_session():
    defaults = {
        "conversations": {},
        "current_conv_id": None,
        "llm_processor": None,
        "initialized": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ========== thread_id 工具函数（修复 BUG-2）==========

def _get_thread_id(cur_conv: dict) -> str:
    """从对话字典安全地取 thread_id。
    
    旧版用 cur_conv.get("conv_id", "default")，但字典键是 "id"，
    导致所有会话共用 "default" 这一个记忆文件。
    """
    return cur_conv.get("id") or cur_conv.get("conv_id") or "default"


# ========== 对话历史持久化 ==========

HISTORY_DIR = "conversations"
FILES_DIR = "files"


def save_conversations():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    for conv_id, conv in st.session_state.conversations.items():
        path = os.path.join(HISTORY_DIR, f"{conv_id}.json")
        save_data = {
            "id": conv_id,
            "title": conv.get("title", "未命名对话"),
            "messages": conv.get("messages", []),
            "doc_context": conv.get("doc_context", []),
            "created_at": conv.get("created_at", ""),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)


def load_conversations():
    if not os.path.exists(HISTORY_DIR):
        return {}
    convs = {}
    for fname in os.listdir(HISTORY_DIR):
        if fname.endswith(".json"):
            path = os.path.join(HISTORY_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                doc_ctx = data.get("doc_context", [])
                if isinstance(doc_ctx, dict):
                    data["doc_context"] = [doc_ctx] if doc_ctx else []
                convs[data["id"]] = data
            except Exception:
                pass
    return convs


def new_conversation():
    conv_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    st.session_state.conversations[conv_id] = {
        "id": conv_id,
        "title": "新对话",
        "messages": [],
        "doc_context": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.current_conv_id = conv_id
    # 新对话开始时清空该 thread 的短期记忆
    from agent_graph import clear_thread
    clear_thread(conv_id)
    save_conversations()
    return conv_id


def delete_conversation(conv_id):
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
        path = os.path.join(HISTORY_DIR, f"{conv_id}.json")
        if os.path.exists(path):
            os.remove(path)
        file_dir = os.path.join(FILES_DIR, conv_id)
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir, ignore_errors=True)
        # 清除短期记忆
        from agent_graph import clear_thread
        clear_thread(conv_id)
        # 新增：清除该对话产生的长期记忆（Memory Tree）
        try:
            from memory_tree import get_memory_store
            get_memory_store().delete_by_conv(conv_id)
        except Exception as e:
            log_message(f"[WARN] 删除长期记忆失败: {e}")
    if st.session_state.current_conv_id == conv_id:
        if st.session_state.conversations:
            st.session_state.current_conv_id = list(st.session_state.conversations.keys())[0]
        else:
            st.session_state.current_conv_id = None


# ========== LLM 初始化 ==========

def get_llm_processor():
    if st.session_state.llm_processor is None:
        try:
            from llm_processor import LLMProcessor
            st.session_state.llm_processor = LLMProcessor()
            log_message("[INFO] LLM处理器初始化成功")
        except Exception as e:
            log_message(f"[ERROR] LLM初始化失败: {e}")
            raise
    return st.session_state.llm_processor


def get_tools(summary_func=None, compare_func=None):
    from tools import get_tools as _get_tools
    return _get_tools(summary_func=summary_func, compare_func=compare_func)


# ========== 文档处理 ==========

def process_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    data = uploaded_file.getbuffer()
    return _parse_bytes(data, suffix)


def _parse_bytes(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        from document_parser import DocumentParser
        return DocumentParser.parse_file(tmp_path)
    except Exception as e:
        log_message(f"[ERROR] 文件解析失败: {e}")
        return f"[解析错误: {e}]"
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ========== System Prompt（唯一入口，修复 BUG-4）==========

def _build_system_prompt(doc_context: list) -> str:
    """构建 system prompt，全局唯一。

    RAG 长度路由：
    - 短文档（<2000 字）：全文注入 prompt，LLM 直接引用
    - 长文档：不注入全文，LLM 必须通过 rag_search 工具按需检索
    """
    SHORT_DOC_THRESHOLD = 2000

    doc_info_lines = []
    short_doc_texts = []
    long_doc_names = []
    has_long_docs = False

    if doc_context:
        for i, doc in enumerate(doc_context, 1):
            content = doc.get("content", "")
            if content.startswith("[解析错误:") or content.startswith("[OCR错误:"):
                continue
            name = doc.get("filename", f"文档{i}")
            if len(content) <= SHORT_DOC_THRESHOLD:
                doc_info_lines.append(f"\n【文档{i}】{name}\n{content}（全文已加载，可直接引用）")
                short_doc_texts.append(content)
            else:
                # 长文档：注入首段概览，避免"标题/作者"等元信息依赖检索命中
                overview = ""
                try:
                    from rag_engine import get_doc_overview, _current_conv_id
                    cid = _current_conv_id
                    if cid:
                        overview = get_doc_overview(cid, name)
                except Exception:
                    pass
                if overview:
                    doc_info_lines.append(
                        f"\n【文档{i}】{name}\n概览：{overview}\n"
                        f"（长文档，全文需通过 rag_search 检索）"
                    )
                else:
                    doc_info_lines.append(
                        f"\n【文档{i}】{name}（长文档，{len(content)} 字，需通过 rag_search 工具检索）"
                    )
                long_doc_names.append(name)
                has_long_docs = True

    doc_info = ("\n当前已上传文档：" + "".join(doc_info_lines)) if doc_info_lines else "（暂无上传文档）"

    # RAG 指令
    rag_instruction = ""
    if has_long_docs:
        rag_instruction = f"""
【长文档检索规则】
以下文档为长文档，全文不在上下文中，你必须通过 rag_search 工具按需检索：
{chr(10).join(f'- {n}' for n in long_doc_names)}

当用户对这些长文档提问时（如"某章节说了什么"、"得分多少"、"作者是谁"），
必须先调用 rag_search(query=用户问题) 检索相关片段，再基于检索结果回答。
检索结果会标注来源，回答时请引用文档名和段落位置。
禁止在未检索的情况下猜测或编造长文档内容。"""

    return f"""你是专业的办公文档智能助手。{doc_info}
{rag_instruction}

可用工具：
- calculator: 数学计算
- date_processor: 日期处理
- file_exporter: 保存分析结果为 Markdown 文件（将内容作为 content 参数传入）
- web_search: 搜索互联网
- spreadsheet_query: 对上传的 Excel/CSV 表格执行数据查询（筛选、排序、统计、分组等）
- rag_search: 对长文档进行语义检索，查找用户问题的相关段落
- generate_document_summary: 为指定文档生成结构化摘要（参数 doc_index: '1'/'2'/'all'）
- compare_documents: 对比分析多个文档的差异（内部含自动评估修正，无需参数）
- memory_write: 将分析结果保存到长期记忆（摘要、对比结果、关键发现等均应保存）
- memory_search: 搜索长期记忆
- memory_list: 列出所有长期记忆
- memory_get: 获取指定长期记忆详情
- memory_delete: 删除指定长期记忆
- memory_clear_all: 清空所有长期记忆
- memory_stats: 长期记忆统计

【重要规则】
1. 完成文档分析（摘要或对比）后，必须同时调用 file_exporter 保存为文件，并调用 memory_write 保存到长期记忆。
2. 如果用户明确要求"保存"，必须调用 file_exporter 和 memory_write。
3. 不要遗漏保存步骤，哪怕分析结果很长也要完整保存。

行为规则：
1. 用户要求生成摘要时，必须调用 generate_document_summary，禁止自行输出摘要文字
2. 用户要求对比/比较/差异分析时，必须调用 compare_documents，禁止自行输出对比结果
3. 文档编号对应【文档N】，"第一个文档"→ doc_index='1'，"所有文档"→ doc_index='all'
4. 调用工具前不输出任何过渡语（"好的，我来…"），直接调用
5. 工具已返回的内容不要再用文字重复描述——工具结果会直接展示给用户
6. 生成摘要或完成对比后，必须同时调用 file_exporter 和 memory_write 保存到文件及长期记忆
7. 【输出规则】当用户要求"对比/总结"并"保存"时，你必须：
   1) 对比/摘要正文由对应工具（compare_documents / generate_document_summary）直接产出并展示，你不要重复输出；
   2) 随后调用 file_exporter，把工具刚产出的完整结果作为 content 参数传入；
   3) 再调用 memory_write 保存到长期记忆（doc_name 用文档名，summary 用对比/摘要结果）；
   4) 最终只补一句"已保存"，无需复述分析内容。
   绝不允许只回复"已保存"而不展示分析内容。
8. 用户明确要求对比分析时才做综合总结，不主动合并
9. 从对话历史中识别并记住用户信息（如姓名、偏好）
10. 不编造内容，不确定时明确告知
11. 【重要】只处理用户【当前这一条】消息的需求；不要主动继续或重做历史消息里
    已处理过的任务，除非当前消息明确再次要求。例如用户只是自我介绍（"我叫xx"），
    就只回应该信息，绝不重新生成摘要或对比分析。
12. 【重要】历史对话中的摘要、分析正文仅作背景参考，严禁原样复述或重新输出；
    当前消息若无新的摘要需求，不要调用 generate_document_summary。
13. 【表格查询】用户对 Excel/CSV 表格提问（筛选、排序、统计等）时，
    必须调用 spreadsheet_query 工具，不要自行根据文档内容推测答案。

必须通过 tool_calls 调用工具，禁止自己输出大段工具应输出的内容。"""


# ========== Agent 对话（修复 BUG-1、BUG-4）==========

def _resolve_eval_details(response_text: str) -> str:
    """将 __EVAL_DETAILS__ 占位符替换为实际的修正详情 HTML。

    evaluate_and_refine() 设置 processor._eval_details_html，并在返回文本中留下
    __EVAL_DETAILS__ 标记。此函数用实际 HTML 替换标记（而非追加到末尾），
    若 HTML 不可用则移除标记，避免裸露的 __EVAL_DETAILS__ 文字显示给用户。
    """
    processor = get_llm_processor()
    real_html = getattr(processor, "_eval_details_html", None)
    if real_html:
        response_text = response_text.replace("__EVAL_DETAILS__", real_html, 1)
        processor._eval_details_html = None
    else:
        response_text = response_text.replace("__EVAL_DETAILS__", "")
    return response_text


def chat_with_agent(
    question: str,
    doc_context: list,
    thread_id: str = "default",
    resume_value=None,
):
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command
    from middleware_config import build_middleware
    from agent_graph import get_checkpointer, get_recent_turns

    processor = get_llm_processor()

    from tools import SummaryGenerator
    summary_gen = SummaryGenerator(
        doc_context,
        summarize_fn=processor.generate_summary_parallel_sync,
    )

    def _compare_fn():
        docs = [
            {"name": d.get("filename") or d.get("name", ""),
             "content": d.get("content", "")}
            for d in doc_context
        ]
        return processor.compare_documents(docs)

    tools = get_tools(summary_func=summary_gen, compare_func=_compare_fn)

    try:
        system_prompt = _build_system_prompt(doc_context)
        import importlib, middleware_config
        importlib.reload(middleware_config)
        mw_list = middleware_config.build_middleware()

        try:
            from langchain.agents import create_agent
        except ImportError:
            try:
                from langgraph.prebuilt import create_react_agent as create_agent
            except ImportError:
                raise ImportError("无法导入 create_agent，请确认 langgraph 已安装")

        agent = create_agent(
            model=processor.llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=mw_list,
            checkpointer=get_checkpointer(),
        )

        config = {"configurable": {"thread_id": thread_id}}

        if resume_value is not None:
            stream_input = Command(resume=resume_value)
            prior_messages = []  # resume 时无需重新注入历史
        else:
            # 注入近期对话历史，确保 Agent 有上下文记忆
            prior_messages = []
            for role, msg_content in get_recent_turns(thread_id):
                if role == "user":
                    prior_messages.append(HumanMessage(content=msg_content))
                else:
                    from langchain_core.messages import AIMessage as _AIMsg
                    prior_messages.append(_AIMsg(content=msg_content))
            stream_input = {
                "messages": prior_messages + [HumanMessage(content=question)]
            }

        log_message(f"[AGENT] thread={thread_id}, "
                    f"history={len(prior_messages)}, "
                    f"resume={'yes' if resume_value else 'no'}")

        _pending_tool = None
        # 工具结果直出后开启压制，AI 复述即重复
        _suppress_agent_text = False

        from langchain_core.messages import AIMessage, ToolMessage

        def _stream_with_retry():
            try:
                for chunk in agent.stream(stream_input, config=config,
                                          stream_mode="messages"):
                    yield chunk
                return
            except Exception as e:
                err_msg = str(e)
                if "tool_calls" in err_msg and "tool messages" in err_msg:
                    log_message(f"[AGENT] checkpointer 损坏，用新 thread_id 重试")
                    import uuid
                    fresh_id = f"{thread_id}_retry_{uuid.uuid4().hex[:8]}"
                    fresh_config = {"configurable": {"thread_id": fresh_id}}
                    fresh_agent = create_agent(
                        model=processor.llm,
                        tools=tools,
                        system_prompt=system_prompt,
                        middleware=mw_list,
                        checkpointer=get_checkpointer(),
                    )
                    for chunk in fresh_agent.stream(
                        stream_input, config=fresh_config,
                        stream_mode="messages"
                    ):
                        yield chunk
                else:
                    raise

        for chunk in _stream_with_retry():
            msg, metadata = chunk

            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )

            # ── ToolMessage ──
            if isinstance(msg, ToolMessage):
                tool_name = (getattr(msg, "name", None) or _pending_tool or "")
                _pending_tool = None
                if not content:
                    continue

                if tool_name == "rag_search":
                    yield "\n\n" + content

                elif tool_name in ("generate_document_summary", "compare_documents"):
                    yield "\n\n" + content
                    _suppress_agent_text = True

                elif tool_name == "file_exporter":
                    if content.strip():
                        yield f"\n\n>  {content.strip()}"
                    # 不再解除压制：工具返回已含"保存成功"确认，AI 复述即重复

                elif tool_name == "memory_write":
                    if content.strip():
                        yield f"\n\n>  {content.strip()}"
                    # 不再解除压制：工具返回已含"记忆保存"确认，AI 复述即重复

                continue

            # ── AIMessage / AIMessageChunk ──
            elif isinstance(msg, AIMessage):
                tc_chunks = getattr(msg, "tool_call_chunks", None)
                tc_calls = getattr(msg, "tool_calls", None)
                tc = tc_chunks or tc_calls

                if tc:
                    for t in (tc if isinstance(tc, list) else [tc]):
                        tn = t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
                        if tn:
                            _pending_tool = tn
                        # 对比/摘要工具：一旦 Agent 决定调用，立即开启压制
                        # 防止 Agent 在工具输出后再复述同样的内容
                        if tn in ("compare_documents", "generate_document_summary"):
                            _suppress_agent_text = True
                    continue  # 携带 tool_call 的 AI 消息：不展示文本

                if _suppress_agent_text:
                    continue  # 压制所有 AI 文本

                if content:
                    yield content  # 正常流式输出

        # 兜底中断检查
        try:
            state = agent.get_state(config)
            if state.next:
                tasks = getattr(state, "tasks", []) or []
                for task in tasks:
                    interrupts = getattr(task, "interrupts", []) or []
                    for intr in interrupts:
                        log_message(f"[AGENT] 检测到中断: value={intr.value}")
                        yield {"type": "interrupt", "value": intr.value}
                        return
        except Exception:
            pass

    except Exception as e:
        log_message(f"[ERROR] Agent对话失败: {traceback.format_exc()}")
        yield f"对话出错: {str(e)}"


def stream_response(question: str, doc_context: list, thread_id: str = "default",
                    resume_value=None):
    """流式生成回复，结束后把本轮问答存入短期记忆。

    【改动 v3】
    - 新增 resume_value，透传给 chat_with_agent 用于中断恢复
    - 检测 interrupt 事件，保存到 session_state 供 UI 渲染确认按钮
    - 中断时不记录 final response
    """
    from agent_graph import append_turn

    # 只有新一轮（非 resume）才存用户问题到短期记忆
    if resume_value is None:
        append_turn(thread_id, "user", question)

    full_response = ""
    emitted = 0
    for chunk in chat_with_agent(question, doc_context, thread_id, resume_value=resume_value):
        # ── 中断事件：保存到 session_state，停止流式输出 ──
        if isinstance(chunk, dict) and chunk.get("type") == "interrupt":
            try:
                import streamlit as _st
                _st.session_state._pending_interrupt = {
                    "thread_id": thread_id,
                    "value": chunk["value"],
                    "question": question,
                    "doc_context": doc_context,
                }
                yield "\n\n---\n⚠️ **安全确认** — 此操作需要您的授权，请在下方确认。\n---"
            except Exception:
                yield "\n\n⚠️ 操作需要安全确认，请在下方点击按钮。"
            return

        # ── 流式实时过滤：先剥已闭合 JSON，再缓冲未闭合尾部 ──
        full_response += chunk
        cleaned = strip_eval_json(full_response)
        safe = _stream_safe_prefix(cleaned)
        if len(safe) > emitted:
            yield safe[emitted:]
            emitted = len(safe)

    # flush 收尾：补上被缓冲住但最终合法的内容
    final = strip_eval_json(full_response)
    if len(final) > emitted:
        yield final[emitted:]

    # 回复完成后存入短期记忆（中断时不走到这里）
    if full_response.strip():
        is_summary_like = (
            "__EVAL_DETAILS__" in full_response
            or "核心摘要" in full_response
            or "关键条款" in full_response
            or "评估器" in full_response
            or len(full_response) > 280
        )
        if is_summary_like:
            mem_text = full_response[:800] + ("…" if len(full_response) > 800 else "")
        else:
            mem_text = full_response
        append_turn(thread_id, "assistant", mem_text)


# ========== CSS 样式 ==========

def inject_css():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f7f7f8; }
    .sidebar-header { font-size: 18px; font-weight: bold; padding: 10px 0; color: #333; }
    .main-header { text-align: center; padding: 10px 0; }
    .stChatMessage { font-size: 16px; }
    .compact-uploader { margin-bottom: 0px; }
    .compact-uploader [data-testid="stFileUploader"] { min-height: 40px; }
    .compact-uploader [data-testid="stFileUploader"] section { padding: 4px 8px; }
    .attachment-msg {
        display: inline-block; padding: 3px 10px;
        background: #e8f0fe; border-radius: 12px;
        font-size: 13px; color: #1a73e8; margin: 2px 0;
    }
    .inline-upload [data-testid="stFileUploadDropzone"] {
        padding: 5px 10px; min-height: 40px;
        border: 1px dashed #ccc; border-radius: 8px;
    }
    .inline-upload [data-testid="stFileUploadDropzone"] small { display: inline; }
    .inline-upload [data-testid="stFileUploadDropzone"] span { font-size: 13px; }
    </style>
    """, unsafe_allow_html=True)


# ========== 侧边栏 ==========

def render_sidebar():
    st.sidebar.markdown('<div class="sidebar-header">📚 文档智能助手</div>', unsafe_allow_html=True)

    if st.sidebar.button("＋ 新对话", use_container_width=True, type="primary"):
        new_conversation()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 对话历史")

    if not st.session_state.conversations:
        st.sidebar.caption("暂无对话记录")
    else:
        sorted_convs = sorted(
            st.session_state.conversations.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True,
        )
        for conv_id, conv in sorted_convs:
            title = conv.get("title", "未命名对话")
            created = conv.get("created_at", "")
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                label = f"{title} ({created[:10]})" if created else title
                is_current = conv_id == st.session_state.current_conv_id
                button_type = "secondary" if is_current else "tertiary"
                if st.button(label, key=f"conv_{conv_id}", use_container_width=True, type=button_type):
                    st.session_state.current_conv_id = conv_id
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{conv_id}", help="删除此对话"):
                    delete_conversation(conv_id)
                    st.rerun()

    st.sidebar.markdown("---")

    cur_conv = st.session_state.conversations.get(st.session_state.current_conv_id)
    if cur_conv and cur_conv.get("doc_context"):
        st.sidebar.markdown("### 📄 已上传文档")
        for doc in cur_conv["doc_context"]:
            filename = doc.get("filename", "未知")
            saved_path = doc.get("saved_path", "")
            st.sidebar.info(filename)
            if saved_path and os.path.exists(saved_path):
                with open(saved_path, "rb") as fh:
                    st.sidebar.download_button(
                        label=f"💾 下载 {filename}",
                        data=fh,
                        file_name=filename,
                        mime="application/octet-stream",
                        use_container_width=True,
                    )

    # Memory Tree 管理
    try:
        from memory_tree import get_memory_store
        store = get_memory_store()
        stats = store.get_stats()
        total = stats.get("total_documents", 0)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧠 长期记忆（Memory Tree）")

        show_all = st.checkbox("📋 显示所有记忆", value=False, key="show_all_memories")

        if show_all and total > 0:
            memories = store.list_all(limit=100)
            for mem in memories:
                col1, col2 = st.sidebar.columns([4, 1])
                with col1:
                    st.write(f"**{mem.doc_name[:25]}{'...' if len(mem.doc_name) > 25 else ''}**")
                    st.caption(f"  {mem.doc_index} | {mem.created_at[:10] if mem.created_at else ''}")
                with col2:
                    if st.button("🗑", key=f"del_mem_{mem.id}", help="删除此记忆"):
                        store.delete(memory_id=mem.id)
                        st.rerun()
                st.divider()
            st.sidebar.markdown("---")
            if st.sidebar.button("🗑️ 清空所有长期记忆", use_container_width=True):
                deleted = store.clear_all()
                st.sidebar.success(f"已清空 {deleted} 条记忆")
                st.rerun()
        else:
            newest = stats.get("newest_memory", "")
            newest_str = f"（最近: {newest[:10]}）" if newest else ""
            st.sidebar.caption(f"已保存 **{total}** 条文档记忆 {newest_str}")
            if total > 0:
                for mem in store.get_recent(limit=3):
                    st.sidebar.write(f"- {mem.doc_name[:25]}{'...' if len(mem.doc_name) > 25 else ''}")
                if total > 3:
                    st.sidebar.caption(f"... 还有 {total - 3} 条")
                st.sidebar.markdown("---")
                if st.sidebar.button("🗑️ 清空所有长期记忆", use_container_width=True):
                    deleted = store.clear_all()
                    st.sidebar.success(f"已清空 {deleted} 条记忆")
                    st.rerun()
    except Exception as e:
        st.sidebar.error(f"Memory Tree 加载失败: {str(e)}")

    return cur_conv


# ========== 文件上传处理 ==========

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}


def handle_files_upload(uploaded_files: list, cur_conv, skip_assistant_message: bool = False):
    if not uploaded_files:
        return cur_conv
    if cur_conv is None:
        new_conversation()
        cur_conv = st.session_state.conversations[st.session_state.current_conv_id]

    existing_names = {d.get("filename", "") for d in cur_conv.get("doc_context", [])}
    new_files, skipped = [], []

    for f in uploaded_files:
        name = f.name if hasattr(f, "name") else f["name"]
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped.append(f"{name} (不支持的类型)")
            continue
        if name in existing_names:
            skipped.append(f"{name} (已存在)")
            continue
        new_files.append(f)
        existing_names.add(name)

    if skipped:
        st.toast(f"已跳过: {', '.join(skipped)}", icon="ℹ️")
    if not new_files:
        return cur_conv

    conv_id = cur_conv["id"]
    file_dir = os.path.join(FILES_DIR, conv_id)
    os.makedirs(file_dir, exist_ok=True)

    uploaded_summary = []
    for f in new_files:
        if isinstance(f, dict):
            name, data = f["name"], f["data"]
            content = _parse_bytes(data, os.path.splitext(name)[1])
        else:
            name, data = f.name, f.getbuffer()
            content = process_uploaded_file(f)

        saved_path = os.path.join(file_dir, name)
        with open(saved_path, "wb") as f_dst:
            f_dst.write(data)

        # 解析失败时不存入 doc_context，避免把错误信息当作文档内容
        if content.startswith("[解析错误:") or content.startswith("[OCR错误:"):
            st.toast(f"⚠️ {name} 解析失败，请检查依赖后刷新页面重试", icon="⚠️")
            continue

        cur_conv["doc_context"].append({
            "filename": name,
            "content": content,
            "parsed": True,
            "saved_path": saved_path,
        })

        # ── RAG 自动索引：长文档（≥2000字）才做向量化，短文档全文已在 prompt 里 ──
        RAG_MIN_LENGTH = 2000
        if len(content) >= RAG_MIN_LENGTH:
            try:
                from rag_engine import index_document
                idx_result = index_document(conv_id, name, content)
                log_message(f"[RAG] 索引 {name}: {idx_result}")
            except Exception as e:
                log_message(f"[RAG] 索引失败 {name}: {e}")
        else:
            log_message(f"[RAG] 跳过 {name}（{len(content)}字 < {RAG_MIN_LENGTH}，全文已注入 prompt）")
        icon = {"pdf": "📕", "docx": "📄", "doc": "📄", "txt": "📝",
                "xlsx": "📊", "xls": "📊", "csv": "📊"}.get(
            name.rsplit(".", 1)[-1].lower(), "🖼️"
        )
        uploaded_summary.append(f"{icon} **{name}**（{len(content)} 字符）")

    # 全部解析失败时直接返回，不更新标题和消息
    if not uploaded_summary:
        return cur_conv

    first_name = new_files[0].name if hasattr(new_files[0], "name") else new_files[0]["name"]
    cur_conv["title"] = (
        first_name.rsplit(".", 1)[0]
        if len(new_files) == 1
        else f"{first_name.rsplit('.', 1)[0]} 等 {len(new_files)} 个文档"
    )

    cur_conv["messages"].append({
        "role": "user",
        "content": "已上传：\n" + "\n".join(f"- {s}" for s in uploaded_summary),
    })
    if not skip_assistant_message:
        cur_conv["messages"].append({
            "role": "assistant",
            "content": f"已加载 {len(new_files)} 个文档，共 {sum(len(d.get('content','')) for d in cur_conv['doc_context'])} 字符。你可以直接提问或输入「生成摘要」开始分析。",
        })
    save_conversations()
    return cur_conv


# ========== 主对话区 ==========

def _run_chat_turn(prompt: str, cur_conv: dict):
    """执行一轮对话并更新消息列表。
    
    抽取为独立函数，避免在 render_chat 中多处复制粘贴同一段逻辑
    （旧版在欢迎页、pending_prompt、正常输入三处各写了一遍，容易漂移出 bug）。

    【v3 改动】支持 HumanInTheLoopMiddleware 中断/恢复：
    - 中断信号从 stream_response 经 session_state._pending_interrupt 传出
    - 保存中断上下文，rerun 后在 render_chat 顶部渲染确认按钮

    【RAG】设置当前会话 ID，使 rag_search 工具能查到正确的向量库
    """
    thread_id = _get_thread_id(cur_conv)
    doc_ctx = cur_conv.get("doc_context", [])

    # 注入当前会话 ID → rag_search 工具通过 rag_engine._current_conv_id 读取
    try:
        import rag_engine
        rag_engine.set_conv_id(thread_id)
    except Exception:
        pass

    # 检查是否是恢复中断
    resume_value = st.session_state.pop("_resume_value", None)

    if resume_value is None:
        cur_conv["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = _stream_write_typewriter(
            placeholder,
            stream_response(prompt, doc_ctx, thread_id, resume_value=resume_value)
        )
        # 双层渲染：流式结束后 PII 脱敏 + JSON 清理 + 评估详情
        masked = apply_pii_mask(response_text)
        masked = strip_eval_json(masked)
        response_text = _resolve_eval_details(masked)
        placeholder.markdown(response_text, unsafe_allow_html=True)
        st.session_state._last_summary = response_text

    # ── 中断分支：不存 assistant 消息，等用户确认后再 resume ──
    pending = st.session_state.pop("_pending_interrupt", None)
    if pending:
        st.session_state._interrupt_confirm = {
            "thread_id": thread_id,
            "question": pending["question"],
            "doc_ctx": doc_ctx,
            "interrupt_value": pending["value"],
        }
        # 存一条草稿消息用于 UI 展示，但不标记为最终回复
        if not cur_conv["messages"] or cur_conv["messages"][-1].get("content") != response_text:
            cur_conv["messages"].append({
                "role": "assistant",
                "content": response_text,
            })
        save_conversations()
        st.rerun()

    # ── 正常分支 ──
    cur_conv["messages"].append({"role": "assistant", "content": response_text})
    save_conversations()
    st.rerun()


def _resume_interrupt(action: str):
    """恢复被中断的 Agent 执行。

    action: "approve" 或 "reject"（对齐 HumanInTheLoopMiddleware 的 allowed_decisions）
    resume 格式: {"decisions": [{"type": "approve"}]} / {"decisions": [{"type": "reject"}]}
    """
    state = st.session_state.pop("_interrupt_confirm", None)
    if not state:
        st.toast("没有待确认的操作", icon="ℹ️")
        return

    thread_id = state["thread_id"]
    question = state["question"]
    doc_ctx = state["doc_ctx"]

    # ── RAG conv_id 注入 ──
    try:
        import rag_engine
        rag_engine.set_conv_id(thread_id)
    except Exception:
        pass

    resume_value = {"decisions": [{"type": action}]}
    log_message(f"[MW DEBUG] 恢复中断: action={action}, resume_value={resume_value}")

    # 拿到当前对话
    cur_conv = st.session_state.conversations.get(st.session_state.current_conv_id)

    # 追加一条"用户同意/拒绝"的系统消息
    if cur_conv:
        label = "同意执行" if action == "approve" else "拒绝执行"
        cur_conv["messages"].append({"role": "user", "content": f"（{label}安全确认操作）"})
        save_conversations()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        if action == "reject":
            # 拒绝时跳过 stream，直接显示干净提示
            response_text = "操作已取消，未执行任何变更。"
            placeholder.markdown(response_text)
        else:
            response_text = _stream_write_typewriter(
                placeholder,
                stream_response(question, doc_ctx, thread_id, resume_value=resume_value)
            )
            masked = apply_pii_mask(response_text)
            masked = strip_eval_json(masked)
            response_text = _resolve_eval_details(masked)
            placeholder.markdown(response_text, unsafe_allow_html=True)

    if cur_conv:
        cur_conv["messages"].append({"role": "assistant", "content": response_text})
        save_conversations()
    st.rerun()


def render_chat():
    st.markdown('<div class="main-header"><h2>📚 文档智能助手</h2></div>', unsafe_allow_html=True)

    cur_conv = st.session_state.conversations.get(st.session_state.current_conv_id)
    if cur_conv is None:
        if st.session_state.conversations:
            st.session_state.current_conv_id = list(st.session_state.conversations.keys())[0]
            cur_conv = st.session_state.conversations[st.session_state.current_conv_id]

    # ===== 空状态：欢迎页 =====
    if cur_conv is None:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #888;">
        <h3>欢迎使用文档智能助手</h3>
        <p>点击输入框左侧 🔗 上传文档开始分析</p>
        </div>
        """, unsafe_allow_html=True)

        welcome_input = st.chat_input(
            "输入问题，或点击左侧 🔗 上传文档（可多选）...",
            accept_file="multiple",
            key="welcome_chat_input",
        )
        if welcome_input is not None:
            has_files = bool(welcome_input.files)
            has_text = bool(welcome_input.text and welcome_input.text.strip())

            if not cur_conv:
                conv_id = new_conversation()
                cur_conv = st.session_state.conversations[conv_id]

            if has_files:
                pending = [{"name": f.name, "data": f.getvalue()} for f in welcome_input.files]
                file_names = "、".join(f.name for f in welcome_input.files)
                cur_conv["messages"].append({"role": "user", "content": f"已上传 {file_names}"})
                cur_conv["messages"].append({"role": "assistant", "content": "⏳ 正在处理文档，请稍候..."})
                save_conversations()
                st.session_state._pending_uploads = pending
                # 修复：上传同时提问时，保存问题，待文档处理完再执行
                if has_text:
                    st.session_state._pending_prompt_after_upload = welcome_input.text.strip()
                st.rerun()
            elif has_text:
                _run_chat_turn(welcome_input.text.strip(), cur_conv)
        return

    # ===== 中断确认栏（HumanInTheLoopMiddleware）=====
    _interrupt = st.session_state.get("_interrupt_confirm")
    if _interrupt:
        intr_val = _interrupt.get("interrupt_value", {})
        # 提取工具名并映射为人类可读名称
        requests = intr_val.get("action_requests", [])
        if requests:
            tool_names = [r.get("name", "未知操作") for r in requests]
            TOOL_LABELS = {
                "memory_clear_all": "清空所有长期记忆",
                "memory_cleanup_old": "清理过期记忆",
                "memory_delete": "删除指定记忆",
            }
            labels = [TOOL_LABELS.get(n, n) for n in tool_names]
            desc = "、".join(labels)
        else:
            desc = "执行敏感操作"

        with st.container(border=True):
            st.warning(f"⚠️ **安全确认**\n\n即将执行：**{desc}**\n\n此操作不可撤销，请确认是否继续。")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 同意执行", use_container_width=True, type="primary",
                             key="interrupt_approve"):
                    _resume_interrupt("approve")
            with c2:
                if st.button("❌ 取消", use_container_width=True, key="interrupt_reject"):
                    _resume_interrupt("reject")
        return  # 确认完成前不渲染后续交互

    # ===== 渲染历史消息 =====
    for msg in cur_conv.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # ===== 延迟文件处理（Phase 2）=====
    if cur_conv and st.session_state.get("_pending_uploads"):
        pending = st.session_state.pop("_pending_uploads")
        msgs = cur_conv["messages"]
        if len(msgs) >= 2 and "正在处理" in msgs[-1].get("content", ""):
            msgs.pop()
        if len(msgs) >= 1 and "已上传" in msgs[-1].get("content", "") and "正在处理" not in msgs[-1].get("content", ""):
            msgs.pop()

        has_pending_prompt = bool(st.session_state.get("_pending_prompt_after_upload"))
        handle_files_upload(pending, cur_conv, skip_assistant_message=has_pending_prompt)

        pending_prompt = st.session_state.pop("_pending_prompt_after_upload", None)
        if pending_prompt:
            # _run_chat_turn 内部会调 st.rerun()，直接执行即可，不需要外层再 rerun
            _run_chat_turn(pending_prompt, cur_conv)
        else:
            # 没有附带文字时，文件处理完后刷新一次展示结果
            st.rerun()

    # ===== 补齐未回复的 user 消息 =====
    if cur_conv["messages"] and cur_conv["messages"][-1]["role"] == "user":
        pending_msg = cur_conv["messages"].pop()["content"]  # 取出，_run_chat_turn 会重新 append
        _run_chat_turn(pending_msg, cur_conv)

    # ===== 快捷操作栏 =====
    doc_ctx = cur_conv.get("doc_context", [])
    user_input = st.chat_input("输入问题或指令...", accept_file="multiple")

    if user_input is None:
        if doc_ctx:
            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                if st.button("📋 生成摘要", use_container_width=True, key="btn_sum"):
                    cur_conv["messages"].append({"role": "user", "content": "请生成文档摘要"})
                    save_conversations()
                    st.rerun()
            with _c2:
                saved_path = doc_ctx[0].get("saved_path", "")
                if saved_path and os.path.exists(saved_path):
                    with open(saved_path, "rb") as fh:
                        st.download_button(
                            label="📥 下载原文件",
                            data=fh,
                            file_name=doc_ctx[0]["filename"],
                            mime="application/octet-stream",
                            use_container_width=True,
                            key="btn_dl",
                        )
            with _c3:
                if st.button("💾 保存结果", use_container_width=True, key="btn_save"):
                    cur_conv["messages"].append({"role": "user", "content": "请将当前文档的分析结果保存为Markdown文件"})
                    save_conversations()
                    st.rerun()
        return

    # ===== 处理新输入 =====
    prompt = user_input.text.strip() if user_input.text else ""
    has_files = bool(user_input.files)
    has_text = bool(prompt)

    if has_files:
        pending = [{"name": f.name, "data": f.getvalue()} for f in user_input.files]
        file_names = "、".join(f.name for f in user_input.files)
        cur_conv["messages"].append({"role": "user", "content": f"已上传 {file_names}"})
        cur_conv["messages"].append({"role": "assistant", "content": "⏳ 正在处理文档，请稍候..."})
        save_conversations()
        st.session_state._pending_uploads = pending
        if prompt:
            st.session_state._pending_prompt_after_upload = prompt
        st.rerun()

    if has_text:
        _run_chat_turn(prompt, cur_conv)


# ========== 主入口 ==========

def main():
    init_session()
    inject_css()

    if not st.session_state.initialized:
        st.session_state.conversations = load_conversations()
        if st.session_state.conversations:
            st.session_state.current_conv_id = list(st.session_state.conversations.keys())[0]
        st.session_state.initialized = True

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()