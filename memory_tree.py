# -*- coding: utf-8 -*-
"""Memory Tree 模块 - 文档记忆存储

参考 OpenHuman 的 Memory Tree 思想：
- 数据存储到本地 SQLite，持久化保存
- 按主题/来源构建层级记忆
- 支持关键词搜索和按时间/名称检索
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class DocumentMemory:
    """文档记忆条目
    
    参考 OpenHuman 的设计：
    - memory_key 是唯一标识符，用于 upsert 去重
    - key = f"{doc_name}_{doc_index}"，相同文档+编号会覆盖
    """
    id: Optional[int] = None
    memory_key: str = ""       # 唯一标识符，格式: doc_name_doc_index
    doc_name: str = ""
    doc_index: str = ""       # 文档编号，如 "1", "2", "all"
    summary: str = ""
    key_points: str = ""       # 关键点列表，JSON 字符串
    metadata: str = ""         # 其他元数据，JSON 字符串
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("key_points"):
            try:
                d["key_points"] = json.loads(d["key_points"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d


class DocumentMemoryStore:
    """文档记忆存储器 - SQLite-backed Memory Tree
    
    类似于 OpenHuman 的 Memory Tree：
    - 数据持久化到本地 SQLite，重启后不丢失
    - 按文档名称和索引组织记忆
    - 支持关键词搜索
    - 生成 Obsidian 兼容的 Markdown 文件
    """

    def __init__(self, db_path: str = "./memory/documents.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        """获取数据库连接，自动确保表存在（防止 DB 文件被外部删除）。"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_memory'")
        if cursor.fetchone() is None:
            self._init_db()
        return conn

    def _init_db(self):
        """初始化数据库表（包含迁移逻辑）"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_memory'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # 检查是否需要迁移（是否有 memory_key 列）
            cursor.execute("PRAGMA table_info(document_memory)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'memory_key' not in columns:
                # 迁移：添加 memory_key 列
                cursor.execute('ALTER TABLE document_memory ADD COLUMN memory_key TEXT')
                
                # 填充已有的 memory_key（基于现有 doc_name + doc_index）
                cursor.execute('''
                    UPDATE document_memory 
                    SET memory_key = doc_name || '_' || doc_index
                    WHERE memory_key IS NULL
                ''')
                conn.commit()
            # 新增：source_conv_id 列迁移
            if 'source_conv_id' not in columns:
                cursor.execute('ALTER TABLE document_memory ADD COLUMN source_conv_id TEXT DEFAULT ""')
                conn.commit()
        else:
            # 新建表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_key TEXT UNIQUE NOT NULL,
                    doc_name TEXT NOT NULL,
                    doc_index TEXT DEFAULT '',
                    summary TEXT NOT NULL,
                    key_points TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    source_conv_id TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        # 创建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_memory_key 
            ON document_memory(memory_key)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_doc_name 
            ON document_memory(doc_name)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created 
            ON document_memory(created_at DESC)
        ''')
        conn.commit()
        conn.close()

    def write(self, doc_name: str, summary: str, doc_index: str = "",
              key_points: str = "[]", metadata: Dict[str, Any] = None,
              source_conv_id: str = "") -> int:
        """写入文档记忆（OpenHuman 风格的 upsert 去重）
        
        参考 OpenHuman 的 memory_store 设计：
        - memory_key = f"{doc_name}_{doc_index}" 是唯一标识符
        - 相同 key 会自动覆盖（upsert），不会重复插入
        
        Args:
            doc_name: 文档名称
            summary: 摘要内容
            doc_index: 文档编号（1, 2, all 等）
            key_points: 关键点列表（JSON 字符串）
            metadata: 其他元数据
            source_conv_id: 产生该记忆的对话 ID
            
        Returns:
            记录的 ID（新增或更新）
        """
        # 生成唯一 key（参考 OpenHuman 的 namespace/key 模式）
        memory_key = f"{doc_name}_{doc_index}" if doc_index else doc_name
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Upsert: INSERT ... ON CONFLICT ... DO UPDATE
        # 相同 key 时更新内容，ID 保持不变
        cursor.execute('''
            INSERT INTO document_memory 
            (memory_key, doc_name, doc_index, summary, key_points, metadata, source_conv_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_key) DO UPDATE SET
                summary = excluded.summary,
                key_points = excluded.key_points,
                metadata = excluded.metadata,
                source_conv_id = excluded.source_conv_id,
                created_at = excluded.created_at
        ''', (
            memory_key,
            doc_name,
            doc_index,
            summary,
            key_points if isinstance(key_points, str) else json.dumps(key_points, ensure_ascii=False),
            json.dumps(metadata or {}, ensure_ascii=False),
            source_conv_id,
            datetime.now().isoformat()
        ))
        conn.commit()
        
        # 获取 record_id（插入时返回新ID，更新时查询）
        cursor.execute('SELECT id FROM document_memory WHERE memory_key = ?', (memory_key,))
        row = cursor.fetchone()
        record_id = row[0] if row else None
        conn.close()
        
        # 同时生成 Obsidian 兼容的 Markdown 文件
        self._save_as_markdown(record_id, doc_name, summary, doc_index, key_points, metadata)
        
        return record_id

    def _save_as_markdown(self, record_id: int, doc_name: str, summary: str,
                          doc_index: str, key_points: str, metadata: Any) -> str:
        """保存为 Obsidian 兼容的 Markdown 文件
        
        模仿 OpenHuman 的 Obsidian-style Wiki 格式
        """
        md_dir = self.db_path.parent / "markdown"
        md_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件名用 memory_key（稳定），同一文档覆盖同一文件，避免无限累积
        safe_key = "".join(c if c.isalnum() or c in " -_" else "_"
                           for c in (f"{doc_name}_{doc_index}" if doc_index else doc_name))
        filename = f"{safe_key}.md"
        filepath = md_dir / filename
        
        # 解析 key_points
        if isinstance(key_points, str):
            try:
                kp_list = json.loads(key_points)
            except (json.JSONDecodeError, TypeError):
                kp_list = []
        else:
            kp_list = key_points or []
        
        # 解析 metadata
        if isinstance(metadata, str):
            try:
                meta_dict = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                meta_dict = metadata or {}
        else:
            meta_dict = metadata or {}
        
        # 生成 Markdown
        lines = [
            "---",
            f"type: document-summary",
            f"doc_name: {doc_name}",
            f"doc_index: {doc_index}",
            f"created: {datetime.now().isoformat()}",
            f"memory_id: {record_id}",
        ]
        for k, v in meta_dict.items():
            lines.append(f"{k}: {v}")
        lines.append("---")
        lines.append("")
        lines.append(f"# {doc_name}")
        lines.append("")
        lines.append("## 摘要")
        lines.append(summary)
        lines.append("")
        if kp_list:
            lines.append("## 关键点")
            for i, point in enumerate(kp_list, 1):
                lines.append(f"{i}. {point}")
            lines.append("")
        lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> Memory ID：{record_id}")
        
        filepath.write_text("\n".join(lines), encoding="utf-8")
        return str(filepath)

    def search(self, query: str, limit: int = 5) -> List[DocumentMemory]:
        """搜索记忆
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
            
        Returns:
            匹配的 DocumentMemory 列表
        """
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM document_memory
            WHERE doc_name LIKE ? OR summary LIKE ? OR key_points LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]

    def get_by_doc(self, doc_name: str, doc_index: str = "") -> Optional[DocumentMemory]:
        """根据文档名和编号获取记忆（用于去重检查）
        
        Args:
            doc_name: 文档名称
            doc_index: 文档编号
            
        Returns:
            DocumentMemory 对象，不存在则返回 None
        """
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM document_memory
            WHERE doc_name = ? AND doc_index = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (doc_name, doc_index))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_memory(row)
        return None

    def list_all(self, limit: int = 20) -> List[DocumentMemory]:
        """列出所有记忆
        
        Args:
            limit: 返回数量限制
            
        Returns:
            DocumentMemory 列表
        """
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM document_memory
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]

    def get_by_index(self, doc_index: str) -> List[DocumentMemory]:
        """按文档编号获取记忆
        
        Args:
            doc_index: 文档编号（1, 2, all 等）
            
        Returns:
            DocumentMemory 列表
        """
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if doc_index == "all":
            cursor.execute('''
                SELECT * FROM document_memory
                ORDER BY doc_index, created_at DESC
            ''')
        else:
            cursor.execute('''
                SELECT * FROM document_memory
                WHERE doc_index = ? OR doc_index LIKE ?
                ORDER BY created_at DESC
            ''', (doc_index, f'{doc_index},%'))
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]

    def get_recent(self, limit: int = 5) -> List[DocumentMemory]:
        """获取最近的记忆
        
        Args:
            limit: 返回数量限制
            
        Returns:
            DocumentMemory 列表
        """
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM document_memory
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]

    def _row_to_memory(self, row: sqlite3.Row) -> DocumentMemory:
        """将数据库行转换为 DocumentMemory 对象"""
        return DocumentMemory(
            id=row["id"],
            memory_key=row["memory_key"],
            doc_name=row["doc_name"],
            doc_index=row["doc_index"],
            summary=row["summary"],
            key_points=row["key_points"],
            metadata=row["metadata"],
            created_at=row["created_at"]
        )

    def get_stats(self) -> Dict[str, int]:
        """获取记忆统计"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM document_memory')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM document_memory')
        row = cursor.fetchone()
        oldest = row[0] if row else None
        newest = row[1] if row else None
        
        conn.close()
        return {
            "total_documents": total,
            "oldest_memory": oldest,
            "newest_memory": newest
        }

    def delete(self, memory_id: int = None, memory_key: str = None) -> bool:
        """删除指定记忆
        
        参考 OpenHuman 的 memory_forget 工具设计
        
        Args:
            memory_id: 记忆 ID
            memory_key: 记忆 key（doc_name_doc_index 格式）
            
        Returns:
            是否删除成功
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        if memory_id:
            cursor.execute('DELETE FROM document_memory WHERE id = ?', (memory_id,))
        elif memory_key:
            cursor.execute('DELETE FROM document_memory WHERE memory_key = ?', (memory_key,))
        else:
            conn.close()
            return False
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def delete_by_conv(self, conv_id: str) -> int:
        """删除某个对话产生的所有长期记忆（对话被删除时调用）。"""
        if not conv_id:
            return 0
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # 先查出来，顺便删对应 markdown 文件
        cursor.execute('SELECT doc_name, doc_index FROM document_memory WHERE source_conv_id = ?', (conv_id,))
        rows = cursor.fetchall()
        cursor.execute('DELETE FROM document_memory WHERE source_conv_id = ?', (conv_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        md_dir = self.db_path.parent / "markdown"
        for r in rows:
            safe_key = "".join(c if c.isalnum() or c in " -_" else "_"
                               for c in (f"{r['doc_name']}_{r['doc_index']}" if r['doc_index'] else r['doc_name']))
            f = md_dir / f"{safe_key}.md"
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass
        return deleted

    def clear_all(self) -> int:
        """清空所有记忆（参考 OpenHuman 的 clear_namespace）
        
        Returns:
            删除的记录数
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM document_memory')
        total = cursor.fetchone()[0]
        cursor.execute('DELETE FROM document_memory')
        conn.commit()
        conn.close()
        return total

    def cleanup_old(self, days: int = 30) -> int:
        """清理指定天数之前的记忆
        
        参考 OpenHuman 的自动清理策略
        
        Args:
            days: 保留最近多少天的记忆，默认 30 天
            
        Returns:
            删除的记录数
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM document_memory 
            WHERE created_at < datetime('now', '-' || ? || ' days')
        ''', (days,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted


# 全局单例
_memory_store: Optional[DocumentMemoryStore] = None


def get_memory_store(db_path: str = "./memory/documents.db") -> DocumentMemoryStore:
    """获取记忆存储单例"""
    global _memory_store
    if _memory_store is None:
        _memory_store = DocumentMemoryStore(db_path)
    return _memory_store
