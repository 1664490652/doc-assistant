# -*- coding: utf-8 -*-
"""
中间件配置 — 统一管理 LangChain Agent Middleware

设计原则：
  - 所有中间件配置集中在此文件，不散落到各处
  - 对外暴露单一函数 build_middleware()，调用方无需了解内部实现
  - 每个中间件可独立禁用（设置对应 env 即可），不影响其他组件

三个中间件：
  1. ModelFallbackMiddleware — 主模型不可用时自动降级
  2. PIIMiddleware            — 自动脱敏输出中的个人隐私（邮箱/电话）
  3. HumanInTheLoopMiddleware  — 关键操作（清空记忆/删对话）前拦截确认
"""

import os
import logging
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
logger = logging.getLogger(__name__)

# ============================================================================
# 配置项（可通过 .env 覆盖）
# ============================================================================

# ---------- ModelFallback ----------
# 主模型在 llm_processor 中配置（LLM_MODEL），备用模型在此配置
FALLBACK_MODEL = os.getenv("MIDDLEWARE_FALLBACK_MODEL", "qwen-turbo")
ENABLE_FALLBACK = bool(FALLBACK_MODEL)

# ---------- PII ----------
# (pii_type, strategy, apply_to_output, detector_regex_or_None)
PII_TYPES: list = [
    # 内置类型：直接用 pii_type 字符串
    ("email",    "redact", True,  None),
    # 自定义类型：提供 detector 正则（中国手机号）
    ("phone",    "mask",   True,  r"(?<!\d)(1[3-9]\d{9})(?!\d)"),
    # 身份证号（18位）
    ("id_card",  "mask",   True,  r"(?<!\d)(\d{17}[\dXx])(?!\d)"),
]
ENABLE_PII = os.getenv("MIDDLEWARE_ENABLE_PII", "1") == "1"

# ---------- HumanInTheLoop ----------
# 需要二次确认的工具名列表
INTERRUPT_TOOLS: dict = {
    "memory_clear_all":     {"allowed_decisions": ["approve", "reject"]},
    "memory_cleanup_old":   {"allowed_decisions": ["approve", "reject"]},
    "memory_delete":        {"allowed_decisions": ["approve", "reject"]},
}
ENABLE_HUMAN_LOOP = os.getenv("MIDDLEWARE_ENABLE_HUMAN_LOOP", "1") == "1"


# ============================================================================
# 工厂函数
# ============================================================================

def build_middleware() -> list:
    """构建当前启用的中间件列表，供 create_agent(middleware=...) 使用。

    Returns:
        list: 中间件实例列表（可能为空）
    """
    result: list = []

    # ── 1. ModelFallbackMiddleware ──
    result.extend(_build_fallback())

    # ── 2. PIIMiddleware ──
    result.extend(_build_pii())

    # ── 3. HumanInTheLoopMiddleware ──
    result.extend(_build_human_loop())

    logger.info(f"build_middleware → {len(result)} 个中间件: "
                f"{[type(m).__name__ for m in result]}")
    return result


# ============================================================================
# 内部构建函数（每个中间件独立，互不引用）
# ============================================================================

def _build_fallback() -> list:
    """构建 ModelFallback 中间件列表。

    ModelFallbackMiddleware 接受 str 或 BaseChatModel 实例。
    由于 qwen-turbo 不在 LangChain 内置提供商列表，需要传入实例对象。
    """
    if not ENABLE_FALLBACK:
        return []
    try:
        from langchain.agents.middleware import ModelFallbackMiddleware
    except ImportError as e:
        logger.warning(f"[Fallback] 导入失败（中间件未启用）: {e}")
        return []
    try:
        # 确保 .env 已加载
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # 构造主模型（DeepSeek）
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("[Fallback] 跳过（未配置 DEEPSEEK_API_KEY）")
            return []
        primary_llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.3,
        )
        # 构造备用模型（阿里云百炼 qwen-turbo）
        from langchain_community.chat_models.tongyi import ChatTongyi
        ds_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not ds_key:
            logger.warning("[Fallback] 跳过（未配置 DASHSCOPE_API_KEY）")
            return []
        fallback_llm = ChatTongyi(model="qwen-turbo", dashscope_api_key=ds_key)

        m = ModelFallbackMiddleware(primary_llm, fallback_llm)
        logger.info(f"[Fallback] 已启用: 主=deepseek-chat, 备用=qwen-turbo")
        return [m]
    except Exception as e:
        logger.error(f"[Fallback] 创建失败: {e}")
        return []


def _build_pii() -> list:
    """构建 PII 脱敏中间件列表。"""
    if not ENABLE_PII:
        return []
    try:
        from langchain.agents.middleware import PIIMiddleware
        instances = []
        for pi_type, strategy, apply_to_output, detector in PII_TYPES:
            kwargs = dict(
                pii_type=pi_type,
                strategy=strategy,
                apply_to_input=False,
                apply_to_output=apply_to_output,
            )
            if detector is not None:
                kwargs["detector"] = detector  # 裸字符串即可，PIIMiddleware 内部会处理
            instances.append(PIIMiddleware(**kwargs))
        logger.info(f"[PII] 已启用 {len(instances)} 个脱敏规则")
        return instances
    except ImportError as e:
        logger.warning(f"[PII] 导入失败（中间件未启用）: {e}")
        return []
    except Exception as e:
        logger.error(f"[PII] 创建失败: {e}")
        return []


def _build_human_loop() -> list:
    """构建 HumanInTheLoop 中间件。"""
    if not ENABLE_HUMAN_LOOP:
        logger.info("[HITL] 已禁用（MIDDLEWARE_ENABLE_HUMAN_LOOP=0）")
        return []
    try:
        from langchain.agents.middleware import HumanInTheLoopMiddleware
    except ImportError as e:
        logger.warning(f"[HITL] 导入失败（中间件未启用）: {e}")
        return []
    try:
        m = HumanInTheLoopMiddleware(
            interrupt_on=INTERRUPT_TOOLS,
            description_prefix="[安全确认] 即将执行破坏性操作",
        )
        logger.info(f"[HITL] 已启用: interrupt_on={INTERRUPT_TOOLS}, "
                    f"instance={type(m).__name__}")
        return [m]
    except Exception as e:
        logger.error(f"[HITL] 创建失败: {e}")
        return []
