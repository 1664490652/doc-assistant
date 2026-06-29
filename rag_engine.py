# -*- coding: utf-8 -*-
"""RAG 引擎 — 文档分块、向量化、语义检索

设计要点：
- ChromaDB 磁盘持久化，零配置
- 按 conv_id 隔离，不同对话互不干扰
- MD5 去重，相同内容不重复 embedding
- 中文最优分块（句号/问号/感叹号优先断句）
- 检索结果附带来源引用（doc_name + chunk_index）
- 相关性阈值过滤低质量片段
"""
import os
import hashlib
import logging
import warnings
from typing import List, Optional, Dict, Any

# 屏蔽 langchain-community 的弃用警告（DashScopeEmbeddings 仍正常工作）
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

# 加载 .env（确保 DASHSCOPE_API_KEY 可用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import Config

logger = logging.getLogger(__name__)

# ========== 配置 ==========
CHROMA_DIR = os.path.join(os.getcwd(), "memory", "chroma")
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
# 中文优先分隔符：段落 → 换行 → 句末标点 → 逗号/分号 → 字符
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "，", "；", "  ", " ", ""]
# Embedding 模型（默认阿里云百炼 text-embedding-v4，国内直连、中文质量好；
# 设置 EMBEDDING_BASE_URL 后切换为任意 OpenAI 兼容 Embedding 提供方）
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
TOP_K = 5
# 相关性阈值（ChromaDB l2 distance，越小越相似）
SCORE_THRESHOLD = 0.3

_embedding_model = None  # 单例缓存

# ========== 当前会话 ID（工具注入用）==========
_current_conv_id: Optional[str] = None


def set_conv_id(conv_id: str) -> None:
    """设置当前会话 ID，供 rag_search 工具读取。"""
    global _current_conv_id
    _current_conv_id = conv_id


def _get_embedding_model():
    """延迟初始化 embedding 模型（单例）。

    提供方选择：
    - 设置 EMBEDDING_BASE_URL → 任意 OpenAI 协议兼容 Embedding 提供方
    - 未设置（默认）→ 阿里云百炼 DashScope 原生协议
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    api_key = Config.EMBEDDING_API_KEY
    base_url = Config.EMBEDDING_BASE_URL
    model = Config.EMBEDDING_MODEL

    if not api_key:
        raise ValueError(
            "未配置 Embedding API Key。请设置 EMBEDDING_API_KEY"
            "（或向后兼容的 DASHSCOPE_API_KEY）环境变量。"
        )

    if base_url:
        # OpenAI 协议兼容 Embedding 提供方
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError("需要安装 langchain-openai: uv pip install langchain-openai")
        _embedding_model = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        logger.info(f"[RAG] Embedding 使用 OpenAI 兼容提供方: model={model}, base_url={base_url}")
    else:
        # 阿里云百炼 DashScope 原生协议（默认）
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
        except ImportError:
            raise ImportError("需要安装 dashscope: uv pip install dashscope")
        _embedding_model = DashScopeEmbeddings(
            model=model,
            dashscope_api_key=api_key,
        )
        logger.info(f"[RAG] Embedding 使用 DashScope: model={model}")

    return _embedding_model


def _get_chroma():
    """获取同步 ChromaDB 客户端，确保目录存在。"""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    try:
        import chromadb
    except ImportError:
        raise ImportError("需要安装 chromadb: uv pip install chromadb")
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _collection_name(conv_id: str) -> str:
    """每个会话独立 collection，命名规则：rag_{safe_conv_id}"""
    safe = "rag_" + "".join(c if c.isalnum() or c in "-_" else "_" for c in conv_id)
    return safe[:63]  # ChromaDB collection 名限制 63 字符


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ========== 公开 API ==========

def index_document(conv_id: str, doc_name: str, text: str) -> Dict[str, Any]:
    """将文档内容分块、向量化后存入 ChromaDB。

    同一 (conv_id, doc_name) 重新上传时：
    - MD5 不同 → 删除旧 chunks 后重新索引
    - MD5 相同 → 跳过，不浪费 embedding

    Returns:
        dict: {"status": "ok"|"skipped", "chunks": int, "doc_hash": str}
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document as LCDocument
    except ImportError:
        raise ImportError("需要安装 langchain-text-splitters: uv pip install langchain-text-splitters")

    if not text or not text.strip():
        return {"status": "error", "reason": "文档内容为空"}

    doc_hash = _md5(text)
    embedder = _get_embedding_model()
    chroma = _get_chroma()
    collection = chroma.get_or_create_collection(name=_collection_name(conv_id))

    # ── MD5 去重：检查是否已有相同内容 ──
    try:
        existing = collection.get(
            where={"$and": [
                {"doc_name": {"$eq": doc_name}},
                {"conv_id": {"$eq": conv_id}},
            ]},
            include=["metadatas"],
        )
        if existing.get("metadatas"):
            old_hash = existing["metadatas"][0].get("doc_hash", "")
            if old_hash == doc_hash:
                logger.info(f"[RAG] 跳过重复索引: {doc_name} (MD5={doc_hash[:8]})")
                return {"status": "skipped", "reason": "内容未变化", "chunks": 0, "doc_hash": doc_hash}
    except Exception:
        pass  # 首次索引没有数据，get 可能报错，安全跳过

    # ── 删除旧 chunks ──
    try:
        collection.delete(where={"$and": [
            {"doc_name": {"$eq": doc_name}},
            {"conv_id": {"$eq": conv_id}},
        ]})
    except Exception:
        pass

    # ── 分块 ──
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
    )
    lc_docs = splitter.create_documents([text])
    if not lc_docs:
        return {"status": "error", "reason": "分块失败"}

    # ── 构建 ChromaDB 记录 ──
    ids = []
    embeddings_list = []
    metadatas = []
    documents = []

    for i, doc in enumerate(lc_docs):
        chunk_id = f"{conv_id}_{doc_name}_{i}"
        ids.append(chunk_id)
        documents.append(doc.page_content)
        metadatas.append({
            "conv_id": conv_id,
            "doc_name": doc_name,
            "chunk_index": i,
            "doc_hash": doc_hash,
            "char_start": i * (CHUNK_SIZE - CHUNK_OVERLAP),
        })

    # 批量 embedding
    embeddings_list = embedder.embed_documents(documents)

    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=documents,
    )

    logger.info(f"[RAG] 已索引: {doc_name} → {len(lc_docs)} chunks (MD5={doc_hash[:8]})")
    return {"status": "ok", "chunks": len(lc_docs), "doc_hash": doc_hash}


def search(conv_id: str, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """语义检索文档内容。

    Args:
        conv_id: 当前会话 ID（隔离用）
        query: 用户查询文本
        top_k: 返回片段数

    Returns:
        list[dict]: 每条包含 text, doc_name, chunk_index, score
    """
    embedder = _get_embedding_model()
    chroma = _get_chroma()
    collection_name = _collection_name(conv_id)

    try:
        collection = chroma.get_collection(name=collection_name)
    except Exception:
        return []  # 无数据，静默返回空

    query_embedding = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"conv_id": conv_id},
        include=["documents", "metadatas", "distances"],
    )

    items = []
    if results.get("documents") and results["documents"][0]:
        for i, doc_text in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

            # ChromaDB distance 范围取决于 metric：默认 l2，越小越相似
            # 简单阈值：distance > 1.5 通常相关性弱
            if distance > 1.5:
                continue

            items.append({
                "text": doc_text,
                "doc_name": metadata.get("doc_name", "未知文档"),
                "chunk_index": metadata.get("chunk_index", 0),
                "score": round(max(0, 1.0 - distance * 0.5), 4),  # 归一化到 ~0-1
            })

    return items


def get_indexed_docs(conv_id: str) -> List[str]:
    """获取当前会话已索引的文档名列表。"""
    chroma = _get_chroma()
    try:
        collection = chroma.get_collection(name=_collection_name(conv_id))
        data = collection.get(include=["metadatas"])
        if data.get("metadatas"):
            return sorted(set(m.get("doc_name", "") for m in data["metadatas"] if m.get("doc_name")))
    except Exception:
        pass
    return []


def get_doc_overview(conv_id: str, doc_name: str, max_chars: int = 300) -> str:
    """获取文档首段概览（标题+摘要），供 System Prompt 使用。

    语义检索中"标题/作者"等元信息查询难以命中 Chunk 0，
    因此通过此函数提前注入，LLM 无需检索即可获知文档基本信息。
    """
    chroma = _get_chroma()
    try:
        collection = chroma.get_collection(name=_collection_name(conv_id))
        results = collection.get(
            where={"$and": [
                {"conv_id": {"$eq": conv_id}},
                {"doc_name": {"$eq": doc_name}},
                {"chunk_index": {"$eq": 0}},
            ]},
            include=["documents"],
        )
        if results.get("documents"):
            return results["documents"][0][:max_chars]
    except Exception:
        pass
    return ""


def clear_conv(conv_id: str) -> bool:
    """清空指定会话的向量库。"""
    chroma = _get_chroma()
    try:
        chroma.delete_collection(name=_collection_name(conv_id))
        logger.info(f"[RAG] 已清空向量库: {conv_id}")
        return True
    except Exception:
        return False
