# -*- coding: utf-8 -*-
"""
短期记忆模块 — 会话级对话上下文管理

设计原则：
- 短期记忆 = 当前会话内最近 N 轮对话，LLM 每次调用时注入
- 长期记忆 = 跨会话的文档摘要，由 memory_tree.py 的 DocumentMemoryStore 负责
- 两者职责分离，互不干扰

【改动说明 vs 旧版】
1. 彻底移除 save_to_memory(thread_id, full_response)：
   旧版把整个 AI 回复（可能几千字摘要）存进 last_summary，
   下次对话读出来塞进历史，LLM 看到后直接复述 → 重复输出根源。
   新版短期记忆只存结构化的 (role, content) 轮次列表，且 content 超长时截断。

2. 新增 ShortTermMemory 类，管理每个 thread 的近 N 轮历史：
   - append_turn(role, content)：追加一轮
   - get_recent(n)：取最近 n 轮，assistant 内容超过 SUMMARY_TRUNCATE_LEN 自动截断
   - clear(thread_id)：清空指定会话

3. thread_id 来源修正：调用方统一用 conv["id"]，不再用 conv["conv_id"]（旧 bug）
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Tuple

from langgraph.checkpoint.memory import InMemorySaver

# ========== 配置 ==========
MEMORY_DIR = os.path.join(os.getcwd(), "memory", "conversations")
os.makedirs(MEMORY_DIR, exist_ok=True)

# assistant 消息超过此长度时，存入短期记忆前截断，避免撑爆 context window
SUMMARY_TRUNCATE_LEN = 300
# 短期记忆最多保留的轮次（user+assistant 各算1条）
MAX_TURNS = 6  # 即最近 3 轮对话


# ========== 文件路径工具 ==========

def _get_thread_file(thread_id: str) -> str:
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)
    return os.path.join(MEMORY_DIR, f"{safe_id}.json")


def _load_thread(thread_id: str) -> dict:
    path = _get_thread_file(thread_id)
    if not os.path.exists(path):
        return {"turns": [], "metadata": {}, "updated_at": ""}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"turns": [], "metadata": {}, "updated_at": ""}


def _save_thread(thread_id: str, data: dict) -> bool:
    try:
        path = _get_thread_file(thread_id)
        data["updated_at"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[ShortTermMemory] 保存失败: {e}")
        return False


# ========== 短期记忆 API ==========

def append_turn(thread_id: str, role: str, content: str) -> bool:
    """追加一轮对话到短期记忆。
    
    assistant 消息超过 SUMMARY_TRUNCATE_LEN 时自动截断，
    防止长摘要被 LLM 当作上下文复读。

    Args:
        thread_id: 对话 ID（来自 conv["id"]）
        role: "user" 或 "assistant"
        content: 消息内容
    """
    data = _load_thread(thread_id)
    turns: list = data.get("turns", [])

    # assistant 长内容（摘要/分析）不存正文，只存中性占位标记。
    # 旧版截断后加"如需完整内容请重新生成"，反而诱导模型在下一轮重做摘要 → 复读 bug。
    stored_content = content
    if role == "assistant" and len(content) > SUMMARY_TRUNCATE_LEN:
        stored_content = "[本轮已生成并展示文档摘要/分析结果，无需重复输出]"

    turns.append({
        "role": role,
        "content": stored_content,
        "ts": datetime.now().isoformat(),
    })

    # 保留最近 MAX_TURNS 条
    if len(turns) > MAX_TURNS:
        turns = turns[-MAX_TURNS:]

    data["turns"] = turns
    return _save_thread(thread_id, data)


def get_recent_turns(thread_id: str, n: int = MAX_TURNS) -> List[Tuple[str, str]]:
    """获取最近 n 条对话记录，格式为 [(role, content), ...]。
    
    供 chat_with_agent 注入 LLM 上下文使用。
    """
    data = _load_thread(thread_id)
    turns = data.get("turns", [])
    recent = turns[-n:] if len(turns) > n else turns
    return [(t["role"], t["content"]) for t in recent]


def clear_thread(thread_id: str) -> bool:
    """清空指定会话的短期记忆（新对话开始时调用）。"""
    path = _get_thread_file(thread_id)
    if os.path.exists(path):
        try:
            os.remove(path)
            return True
        except Exception:
            return False
    return True


def save_metadata(thread_id: str, metadata: dict) -> bool:
    """保存会话级元数据（如用户名、偏好），不影响对话轮次。"""
    data = _load_thread(thread_id)
    data.setdefault("metadata", {}).update(metadata)
    return _save_thread(thread_id, data)


def get_metadata(thread_id: str) -> dict:
    """读取会话元数据。"""
    return _load_thread(thread_id).get("metadata", {})


def list_conversations() -> list:
    """列出所有有短期记忆的会话（调试用）。"""
    result = []
    try:
        for fname in os.listdir(MEMORY_DIR):
            if fname.endswith(".json"):
                thread_id = fname[:-5]
                data = _load_thread(thread_id)
                result.append({
                    "thread_id": thread_id,
                    "turn_count": len(data.get("turns", [])),
                    "updated_at": data.get("updated_at", ""),
                    "metadata": data.get("metadata", {}),
                })
        return sorted(result, key=lambda x: x["updated_at"], reverse=True)
    except Exception:
        return []


# ========== 兼容旧接口（防止其他文件 import 报错）==========
# 旧版 save_to_memory / get_from_memory 已废弃，保留空壳避免 ImportError

def save_to_memory(thread_id: str, last_summary: str, metadata: dict = None) -> bool:
    """
    [已废弃] 旧版接口，保留兼容。
    
    旧版把整个 AI 回复存入 last_summary，导致下次对话 LLM 复读。
    现在请改用：
        append_turn(thread_id, "assistant", content)  # 追加对话轮次
        save_metadata(thread_id, metadata)             # 保存元数据
    """
    # 不再存储，只保存元数据（如有）
    if metadata:
        return save_metadata(thread_id, metadata)
    return True


def get_from_memory(thread_id: str, key: str = "last_summary") -> str:
    """[已废弃] 旧版接口，保留兼容。返回空字符串。"""
    return ""


def get_last_summary(thread_id: str = "default") -> str:
    """[已废弃] 旧版接口，保留兼容。返回空字符串。"""
    return ""


# ========== Checkpointer（供 LangGraph Agent 中断/恢复使用）==========

_CHECKPOINTER = None


def get_checkpointer() -> InMemorySaver:
    """获取模块级单例 checkpointer。

    HumanInTheLoopMiddleware 依赖 LangGraph 的 interrupt() 机制，
    这要求 Agent 图必须配置 checkpointer，否则中断状态无法持久化/恢复。

    使用模块级单例确保 Streamlit rerun 时不丢失中断状态。
    （如需跨进程持久化，可替换为 SqliteSaver）
    """
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        _CHECKPOINTER = InMemorySaver()
    return _CHECKPOINTER
