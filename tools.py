# -*- coding: utf-8 -*-
"""工具定义 - 使用 @tool 装饰器，简洁高效"""
from langchain.tools import tool
from datetime import datetime, timedelta
import re
import os
import sys
import json   # 修复：memory_get 用到 json.loads，原来未导入


# ========== 计算器工具 ==========
@tool
def calculator(expression: str) -> str:
    """执行数学计算，支持加减乘除、幂运算、括号、平方根等。
    示例：'(3 + 5) * 2', '2 ** 10', 'sqrt(144)'

    Args:
        expression: 数学表达式字符串，如 '467 ** 0.5' 或 'sqrt(529)'
    """
    try:
        # 替换数学函数名
        clean_expr = expression.strip()
        clean_expr = clean_expr.replace('sqrt', 'math.sqrt')
        clean_expr = clean_expr.replace('^', '**')

        # 只允许安全的字符
        allowed = r'[^0-9+\-*/().%\s,math.sqrt]'
        if re.search(allowed, clean_expr.replace('math.sqrt', '')):
            return "错误：表达式包含非法字符"

        if not clean_expr.strip():
            return "错误：无效的数学表达式"

        import math
        result = eval(clean_expr, {"__builtins__": {}}, {"math": math})
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


# ========== 日期处理工具 ==========
@tool
def date_processor(query: str) -> str:
    """处理日期相关任务：获取当前日期、计算日期差、日期加减、查星期几等。
    支持格式：'今天'、'明天'、'30天后'、'2024-01-15 到 2024-02-20 相差几天'、'2024-03-08 是星期几'

    Args:
        query: 日期查询字符串，描述你需要知道的日期信息
    """
    try:
        today = datetime.now()

        if "今天" in query or "当前日期" in query:
            return f"今天是：{today.strftime('%Y年%m月%d日 %H:%M:%S')}"

        if "明天" in query:
            tomorrow = today + timedelta(days=1)
            return f"明天是：{tomorrow.strftime('%Y年%m月%d日')}"

        if "昨天" in query:
            yesterday = today - timedelta(days=1)
            return f"昨天是：{yesterday.strftime('%Y年%m月%d日')}"

        # N天后/前
        days_match = re.search(r'(\d+)\s*天', query)
        if days_match:
            days = int(days_match.group(1))
            if "后" in query or "加" in query or "+" in query:
                result_date = today + timedelta(days=days)
                return f"{days}天后是：{result_date.strftime('%Y年%m月%d日')}"
            elif "前" in query or "减" in query or "-" in query:
                result_date = today - timedelta(days=days)
                return f"{days}天前是：{result_date.strftime('%Y年%m月%d日')}"

        # 两个日期相差天数
        date_pattern = r'(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})[日号]?'
        dates = re.findall(date_pattern, query)
        if len(dates) >= 2:
            date1 = datetime(int(dates[0][0]), int(dates[0][1]), int(dates[0][2]))
            date2 = datetime(int(dates[1][0]), int(dates[1][1]), int(dates[1][2]))
            diff_days = abs((date2 - date1).days)
            return f"{date1.strftime('%Y年%m月%d日')} 到 {date2.strftime('%Y年%m月%d日')} 相差 {diff_days} 天"

        # 查星期几
        date_match = re.search(date_pattern, query)
        if date_match and "星期" in query:
            date = datetime(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
            weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            return f"{date.strftime('%Y年%m月%d日')} 是 {weekdays[date.weekday()]}"

        return f"无法理解日期查询：{query}\n支持的格式：\n- 今天/明天/昨天\n- 30天后\n- 2024年1月15日 到 2024年2月20日相差几天\n- 2024年3月8日是星期几"

    except Exception as e:
        return f"日期处理错误：{str(e)}"


# ========== 文件导出工具 ==========
@tool
def file_exporter(content: str = "") -> str:
    """将当前文档的分析结果保存为Markdown文件。

    Args:
        content: 要保存的文档分析内容。如果为空，则自动从 LangGraph 状态读取。
    """
    try:
        save_content = content.strip() if content else ''

        if not save_content:
            return "错误：保存失败——请把要保存的完整分析内容作为 content 参数显式传入 file_exporter。"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"analysis_result_{timestamp}.md"

        output_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        full_path = os.path.abspath(os.path.join(output_dir, file_name))

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(save_content)   # 修复：原为 content，导致 fallback 内容未写入

        return f"文件已成功保存到：{full_path}"

    except Exception as e:
        return f"文件保存失败：{str(e)}"


# ========== 网络搜索工具（内部使用类封装复杂逻辑）==========
class _WebSearchEngine:
    """内部搜索引擎封装"""
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    }

    @staticmethod
    def clean_query(query: str) -> str:
        """用 LLM 提取搜索关键词（比停用词匹配更可靠）"""
        try:
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = "https://api.deepseek.com/v1" if os.getenv("DEEPSEEK_API_KEY") else None

            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                messages=[{
                    "role": "user",
                    "content": f"请从以下用户输入中提取适合搜索引擎的关键词，直接返回关键词，不要解释：\n{query}"
                }],
                max_tokens=50,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            return result if result else query
        except Exception:
            return query  # 失败时返回原始query，不影响搜索

    @staticmethod
    def _clean_link(link: str) -> str:
        """清洗链接：去空白、验证格式"""
        if not link:
            return ''
        link = link.strip().replace('\n', '').replace('\r', '')
        # 只保留合法的 http/https URL
        if link.startswith(('http://', 'https://')):
            return link
        return ''

    @staticmethod
    def search_duckduckgo(query: str, max_results: int) -> list:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [{'title': r.get('title', '无标题').strip(),
                 'link': _WebSearchEngine._clean_link(r.get('href', '')),
                 'snippet': r.get('body', '无摘要').strip()} for r in results]

    @staticmethod
    def search_bing(query: str, max_results: int) -> list:
        import requests
        from urllib.parse import quote_plus
        from html.parser import HTMLParser

        url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"
        response = requests.get(url, headers=_WebSearchEngine.HEADERS, timeout=15)
        response.encoding = 'utf-8'

        class BingParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = None
                self.in_result = False
                self.in_title = False
                self.in_snippet = False

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == 'li' and 'b_algo' in attrs_dict.get('class', ''):
                    self.in_result = True
                    self.current = {'title': '', 'link': '', 'snippet': ''}
                elif tag == 'h2' and self.in_result:
                    self.in_title = True
                elif tag == 'a' and self.in_result and 'href' in attrs_dict:
                    if not self.current['link']:
                        self.current['link'] = _WebSearchEngine._clean_link(attrs_dict['href'])
                elif tag in ('p', 'span') and self.in_result:
                    self.in_snippet = True

            def handle_endtag(self, tag):
                if tag == 'li' and self.in_result:
                    self.in_result = False
                    if self.current and self.current['title']:
                        self.results.append(self.current)
                elif tag == 'h2':
                    self.in_title = False
                elif tag in ('p', 'span'):
                    self.in_snippet = False

            def handle_data(self, data):
                if self.in_title and self.current:
                    self.current['title'] += data.strip()
                elif self.in_snippet and self.current:
                    self.current['snippet'] += data.strip()

        parser = BingParser()
        parser.feed(response.text)
        return parser.results[:max_results]

    @staticmethod
    def format_results(query: str, results: list) -> str:
        items = []
        for i, r in enumerate(results[:5], 1):
            title = (r.get('title', '无标题') or '').strip()
            link = (r.get('link', '') or '').strip()
            snippet = (r.get('snippet', '无摘要') or '').strip()
            if title and link:
                items.append(
                    f"### {i}. {title}\n\n"
                    f"**摘要**: {snippet[:200]}{'...' if len(snippet) > 200 else ''}\n\n"
                    f"**来源**: [{link}]({link})\n\n---"
                )
        if items:
            return f"## 搜索结果: {query}\n\n共找到 {len(items)} 条相关信息\n\n" + "\n".join(items) + \
                   "\n\n**提示**: 以上信息来自互联网搜索，请注意核实信息的准确性。"
        return f"未找到与'{query}'相关的结果。请尝试更换搜索关键词。"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """搜索互联网获取最新信息，支持多网页分析并返回来源链接。适用于查询最新新闻、技术文档、产品信息等。

    Args:
        query: 搜索关键词，越具体越好
        max_results: 最大返回结果数，默认5个
    """
    engine = _WebSearchEngine()
    clean_query = engine.clean_query(query)

    # 尝试 DuckDuckGo
    try:
        results = engine.search_duckduckgo(clean_query, max_results)
        if results:
            return engine.format_results(clean_query, results)
    except Exception:
        pass

    # 兜底：Bing
    try:
        results = engine.search_bing(clean_query, max_results)
        if results:
            return engine.format_results(clean_query, results)
    except Exception:
        pass

    return f"未找到与'{clean_query}'相关的结果。请尝试更换搜索关键词。"


# ========== 摘要工具 ==========
class SummaryGenerator:
    """摘要生成工具 — 封装评估器修正链路，作为 LangChain Tool 注入 Agent"""

    def __init__(self, doc_context: list, summarize_fn):
        """
        Args:
            doc_context: list of dict [{filename, content}, ...]
            summarize_fn: callable(content: str) -> str，发起摘要+评估修正
        """
        self.doc_context = doc_context
        self.summarize_fn = summarize_fn

    def __call__(self, doc_index: str = None, **kwargs) -> str:
        """为指定文档生成结构化摘要

        Args:
            doc_index: 文档编号，'1'、'2'、'all' 等
        """
        import json as _json

        idx = doc_index
        if idx is None and kwargs:
            idx = kwargs.get("doc_index", "")
            if not idx and kwargs.get("__arg1"):
                idx = kwargs["__arg1"]
        if isinstance(idx, str) and idx.startswith("{"):
            try:
                idx = _json.loads(idx).get("doc_index", "")
            except _json.JSONDecodeError:
                pass
        idx = str(idx or "").strip()

        if idx.lower() == "all":
            all_content = "\n\n".join(d.get("content", "") for d in self.doc_context)
            if not all_content.strip():
                return "没有可用的文档内容。"
            return self.summarize_fn(all_content)

        try:
            i = int(idx) - 1
            if 0 <= i < len(self.doc_context):
                content = self.doc_context[i].get("content", "")
                if not content.strip():
                    return f"文档 {idx} 内容为空。"
                return self.summarize_fn(content)
            return f"文档编号 {idx} 无效，有效范围 1-{len(self.doc_context)}。"
        except ValueError:
            return f"无效的文档编号: {idx}"


# ========== 记忆工具（Memory Tree）==========
from memory_tree import get_memory_store


@tool
def memory_write(doc_name: str, summary: str, doc_index: str = "",
                 key_points: str = "", metadata: str = "{}") -> str:
    """将文档摘要写入长期记忆（Memory Tree）。

    参考 OpenHuman 的 memory_store 设计，使用 upsert 去重：
    - memory_key = f"{doc_name}_{doc_index}" 是唯一标识符
    - 相同 key 会自动覆盖（upsert），不会重复插入
    
    生成摘要后调用此工具将摘要持久化保存，以便后续跨会话查询。
    同时会生成 Obsidian 兼容的 Markdown 文件。

    Args:
        doc_name: 文档名称，如 '硕士专业学位论文评阅书.pdf'
        summary: 文档的结构化摘要内容
        doc_index: 文档编号，如 '1', '2', 'all'
        key_points: 关键点列表，JSON 数组格式，如 '["关键点1", "关键点2"]'
        metadata: 其他元数据，JSON 对象格式，如 '{"char_count": 1300, "pages": 2}'
    """
    try:
        store = get_memory_store()

        # 取当前对话 ID，写入 source_conv_id，删除对话时可按此清理
        source_conv_id = ""
        try:
            import streamlit as _st
            source_conv_id = _st.session_state.get("current_conv_id", "") or ""
        except Exception:
            pass

        existing = store.get_by_doc(doc_name, doc_index)
        is_update = existing is not None

        record_id = store.write(
            doc_name=doc_name,
            summary=summary,
            doc_index=doc_index,
            key_points=key_points,
            metadata=metadata,
            source_conv_id=source_conv_id,   # 新增
        )

        if is_update:
            return f"🔄 摘要已更新到记忆（Memory ID: {record_id}）"
        else:
            return f"✅ 摘要已保存到记忆（Memory ID: {record_id}）"
    except Exception as e:
        return f"❌ 记忆保存失败：{str(e)}"


@tool
def memory_search(query: str, limit: int = 3) -> str:
    """搜索长期记忆中的相关内容。

    当用户询问之前分析过的文档时，先用此工具搜索记忆，
    找到后可以进一步查询详细内容。

    Args:
        query: 搜索关键词，可以是文档名、关键词或摘要内容
        limit: 返回结果数量，默认 3 条
    """
    try:
        store = get_memory_store()
        results = store.search(query=query, limit=limit)
        
        if not results:
            return "🔍 未找到相关记忆。"
        
        lines = [f"📚 找到 {len(results)} 条相关记忆：\n"]
        for i, mem in enumerate(results, 1):
            lines.append(f"\n--- 记忆 {i} ---")
            lines.append(f"📄 文档：{mem.doc_name}")
            if mem.doc_index:
                lines.append(f"📑 编号：{mem.doc_index}")
            lines.append(f"📝 摘要：{mem.summary[:300]}{'...' if len(mem.summary) > 300 else ''}")
            if mem.created_at:
                lines.append(f"🕐 时间：{mem.created_at}")
            lines.append(f"🔑 记忆ID：{mem.id}")
        
        lines.append("\n\n💡 提示：可以使用 memory_get 工具获取完整摘要，doc_index 使用记忆ID。")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 记忆搜索失败：{str(e)}"


@tool
def memory_list(limit: int = 10) -> str:
    """列出所有已保存的文档记忆。

    当用户想查看之前分析过哪些文档时，调用此工具。

    Args:
        limit: 返回数量限制，默认 10 条
    """
    try:
        store = get_memory_store()
        results = store.list_all(limit=limit)
        
        if not results:
            return "📭 暂无保存的文档记忆。"
        
        stats = store.get_stats()
        lines = [f"📚 文档记忆库（共 {stats['total_documents']} 条）：\n"]
        for i, mem in enumerate(results, 1):
            lines.append(f"\n{i}. **{mem.doc_name}**")
            if mem.doc_index:
                lines.append(f"   📑 编号：{mem.doc_index}")
            if mem.summary:
                lines.append(f"   📝 {mem.summary[:100]}{'...' if len(mem.summary) > 100 else ''}")
            if mem.created_at:
                lines.append(f"   🕐 {mem.created_at[:19]}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 记忆列表获取失败：{str(e)}"


@tool
def memory_get(doc_index: str) -> str:
    """获取指定记忆的完整摘要。

    通过 memory_search 或 memory_list 获取记忆ID后，
    用此工具获取完整摘要内容。

    Args:
        doc_index: 文档编号或记忆ID（数字）
    """
    try:
        store = get_memory_store()
        memories = store.get_by_index(doc_index)
        
        if not memories:
            return f"❌ 未找到记忆 ID: {doc_index}"
        
        lines = []
        for mem in memories:
            lines.append(f"# {mem.doc_name}")
            lines.append(f"\n## 完整摘要\n{mem.summary}")
            if mem.key_points:
                try:
                    kp_list = json.loads(mem.key_points)
                    if kp_list:
                        lines.append("\n## 关键点")
                        for i, point in enumerate(kp_list, 1):
                            lines.append(f"{i}. {point}")
                except (json.JSONDecodeError, TypeError):
                    pass
            lines.append(f"\n---\n记忆ID：{mem.id} | 创建时间：{mem.created_at}")
            lines.append("\n" + "=" * 50 + "\n")
        
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 获取记忆失败：{str(e)}"


@tool
def memory_delete(memory_id: str = None, memory_key: str = None) -> str:
    """删除指定记忆（参考 OpenHuman 的 memory_forget 工具）。

    通过 memory_search 或 memory_list 获取记忆ID或memory_key后，
    用此工具删除不需要的记忆。

    Args:
        memory_id: 记忆 ID（数字），如 '1', '2'
        memory_key: 记忆 key（格式：doc_name_doc_index），如 '外审意见3.pdf_1'
        注意：memory_id 和 memory_key 至少提供一个
    """
    try:
        store = get_memory_store()
        
        # 转换为整数
        mid = int(memory_id) if memory_id else None
        
        deleted = store.delete(memory_id=mid, memory_key=memory_key)
        
        if deleted:
            if mid:
                return f"✅ 已删除记忆（ID: {mid}）"
            else:
                return f"✅ 已删除记忆（Key: {memory_key}）"
        else:
            return f"❌ 未找到指定的记忆"
    except ValueError:
        return f"❌ memory_id 必须是数字"
    except Exception as e:
        return f"❌ 删除记忆失败：{str(e)}"


@tool
def memory_clear_all() -> str:
    """清空所有记忆（参考 OpenHuman 的 clear_namespace）。

    警告：此操作不可逆，会删除所有保存的文档摘要。
    建议先使用 memory_list 查看有哪些记忆。
    """
    try:
        store = get_memory_store()
        stats = store.get_stats()
        total = stats.get("total_documents", 0)
        
        if total == 0:
            return "ℹ️ 没有需要清空的记忆"
        
        deleted = store.clear_all()
        return f"🗑️ 已清空所有记忆（共 {deleted} 条记录）"
    except Exception as e:
        return f"❌ 清空记忆失败：{str(e)}"


@tool
def memory_cleanup_old(days: int = 30) -> str:
    """清理旧记忆（参考 OpenHuman 的自动清理策略）。

    删除指定天数之前保存的记忆，默认保留最近 30 天的记忆。

    Args:
        days: 保留最近多少天的记忆，默认 30 天
    """
    try:
        store = get_memory_store()
        stats_before = store.get_stats()
        total_before = stats_before.get("total_documents", 0)
        
        if total_before == 0:
            return "ℹ️ 没有需要清理的记忆"
        
        deleted = store.cleanup_old(days=days)
        
        if deleted == 0:
            return f"ℹ️ 没有超过 {days} 天的旧记忆需要清理"
        
        return f"🧹 已清理 {days} 天前的旧记忆（删除了 {deleted} 条记录）"
    except Exception as e:
        return f"❌ 清理旧记忆失败：{str(e)}"


@tool
def memory_stats() -> str:
    """查看记忆统计信息。

    返回记忆数量、时间范围等统计信息。
    """
    try:
        store = get_memory_store()
        stats = store.get_stats()
        
        total = stats.get("total_documents", 0)
        oldest = stats.get("oldest_memory", "无")
        newest = stats.get("newest_memory", "无")
        
        lines = ["📊 记忆统计：", f"- 总记录数：{total} 条"]
        
        if oldest:                       # 修复：原为 oldest != oldest（恒为 False）
            lines.append(f"- 最早记忆：{oldest[:19] if oldest else '无'}")
        if newest:
            lines.append(f"- 最新记忆：{newest[:19] if newest else '无'}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 获取统计失败：{str(e)}"


# ========== 表格查询工具（Excel / CSV）==========
@tool
def spreadsheet_query(question: str) -> str:
    """对当前上传的 Excel (.xlsx/.xls) 或 CSV 表格执行数据查询。

    支持的操作：
    - 筛选：某列等于/大于/包含某值的行
    - 排序：按某列升序/降序排列，取前 N 行
    - 统计：某列的总和、均值、最大值、最小值、计数
    - 分组：按某列分组后统计（如各部门平均薪资）
    - 去重/唯一值查看

    Args:
        question: 对表格的自然语言问题，如：
            "销售额最高的5行是哪些？"
            "每个部门的平均薪资是多少？"
            "状态为'已完成'的有多少条？"
            "列出'产品名称'列的所有不重复值"
    """
    try:
        import pandas as pd

        # 找到当前对话的表格文件
        conv_id = ""
        file_dir = ""
        try:
            import streamlit as _st
            conv_id = _st.session_state.get("current_conv_id", "") or ""
            if conv_id:
                file_dir = os.path.join("files", conv_id)
            else:
                return "❌ 未找到当前对话，请先上传表格文件。"
        except Exception:
            return "❌ 无法获取当前对话信息。"

        if not os.path.isdir(file_dir):
            return "❌ 未找到当前对话的文件目录，请先上传表格文件。"

        # 找到表格文件
        table_files = [f for f in os.listdir(file_dir)
                       if f.lower().endswith(('.xlsx', '.xls', '.csv'))]
        if not table_files:
            return "❌ 当前对话未上传 Excel/CSV 表格文件。"

        table_path = os.path.join(file_dir, table_files[0])
        ext = os.path.splitext(table_files[0])[1].lower()

        # 加载数据
        if ext in ('.xlsx', '.xls'):
            df = pd.read_excel(table_path)
        else:
            for enc in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                try:
                    df = pd.read_csv(table_path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue

        # 用 LLM 将自然语言问题转为 pandas 操作
        col_info = "\n".join(
            f"- {c} (dtype={df[c].dtype}, "
            f"sample={list(df[c].dropna().head(3).values)})"
            for c in df.columns
        )
        prompt = (
            f"你是一个 pandas 专家。根据以下表格信息和用户问题，生成一段可直接执行的 Python 代码，"
            f"代码最后将结果赋给变量 result（DataFrame 或 Series 或标量）。"
            f"不要 import，不要定义函数，不要 print，只输出代码。"
            f"\n\n表格列名和类型：\n{col_info}"
            f"\n表格行数：{len(df)}"
            f"\n\n用户问题：{question}"
            f"\n\n代码："
        )

        try:
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = "https://api.deepseek.com/v1" if os.getenv("DEEPSEEK_API_KEY") else None
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )
            code = response.choices[0].message.content.strip()
            # 清理代码块标记
            if code.startswith("```"):
                code = code.split("\n", 1)[1]
                if code.endswith("```"):
                    code = code[:-3]
                code = code.strip()
                if code.startswith("python"):
                    code = code.split("\n", 1)[1]
        except Exception:
            return f"❌ 生成查询代码失败，请换一种方式提问。"

        # 执行生成的代码
        local_vars = {"df": df, "pd": pd}
        try:
            exec(code, {"pd": pd}, local_vars)
            result = local_vars.get("result")
            if result is None:
                return f"❌ 查询未返回结果。\n生成的代码：\n```python\n{code}\n```"
        except Exception as e:
            return f"❌ 执行查询失败：{e}\n生成的代码：\n```python\n{code}\n```"

        # 格式化返回结果
        if isinstance(result, pd.DataFrame):
            return f"📊 查询结果（{len(result)} 行）：\n\n{result.to_string(index=False)}"
        elif isinstance(result, pd.Series):
            return f"📊 查询结果：\n\n{result.to_string()}"
        else:
            return f"📊 查询结果：{result}"

    except ImportError:
        return "❌ 表格查询需要 pandas，请运行: pip install pandas openpyxl"
    except Exception as e:
        return f"❌ 表格查询失败：{str(e)}"


# ========== RAG 语义检索工具 ==========
@tool
def rag_search(query: str) -> str:
    """对已上传的文档内容进行语义检索。

    当用户对长文档的具体细节提问时（如"某条款说了什么"、"评分多少"、"作者是谁"），
    使用此工具从文档中检索相关段落。短文档可直接用上下文回答，无需调用此工具。

    Args:
        query: 用户想问的问题或关键词，如"论文得分"、"合同违约条款"
    """
    try:
        from rag_engine import _current_conv_id, search, get_indexed_docs
    except ImportError as e:
        return f"❌ RAG 引擎未加载: {e}"

    conv_id = _current_conv_id
    if not conv_id:
        return "❌ 当前无活跃会话，无法检索文档。请先上传文档。"

    # 检查是否有已索引的文档
    indexed = get_indexed_docs(conv_id)
    if not indexed:
        return "❌ 当前会话没有已索引的文档。请确认文档已上传并解析成功。"

    results = search(conv_id, query)

    if not results:
        return f"未在文档中找到与「{query}」相关的内容。文档列表：{', '.join(indexed)}"

    lines = [f"🔍 **语义检索结果**（查询: {query}）\n"]
    for i, r in enumerate(results, 1):
        source = f"据《{r['doc_name']}》第{r['chunk_index'] + 1}段"
        lines.append(f"**片段 {i}** [{source}] (相关度: {r['score']:.2f})\n> {r['text']}\n")

    return "\n".join(lines)


# ========== 工具列表 ==========
def get_tools(processor=None, summary_func=None, compare_func=None):
    """获取所有可用工具。

    Args:
        processor: LLMProcessor 实例（保留兼容）
        summary_func: callable(doc_index: str) -> str，摘要生成函数
        compare_func: callable() -> str，对比分析函数（内部已含评估修正）
    """
    tools = [
        calculator,
        date_processor,
        file_exporter,
        web_search,
        spreadsheet_query,
        rag_search,
        memory_write,
        memory_search,
        memory_list,
        memory_get,
        memory_delete,
        memory_clear_all,
        memory_cleanup_old,
        memory_stats,
    ]
    if summary_func:
        from langchain_core.tools import Tool
        _fn = summary_func
        tools.append(Tool(
            name="generate_document_summary",
            description=(
                "为指定文档生成结构化摘要（含自动评估修正）。"
                "参数 doc_index: 文档编号（如 '1' 表示第1个文档，'all' 表示所有文档合并）。"
                "当用户要求生成摘要时调用此工具，不要直接用 LLM 生成。"
            ),
            func=lambda doc_index="all", **kw: _fn(doc_index, **kw),
        ))
    if compare_func:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        _cfn = compare_func

        class _CompareArgs(BaseModel):
            pass  # 无参数

        tools.append(StructuredTool.from_function(
            name="compare_documents",
            description=(
                "对比分析当前已上传的多个文档，输出差异、统一规范、整合建议与风险"
                "（内部已含跨文档事实的自动评估修正）。"
                "当用户要求'对比/比较/差异分析'多个文档时必须调用此工具，"
                "禁止直接用 LLM 自行生成对比结果。无需参数。"
            ),
            func=lambda **kw: _cfn(),
            args_schema=_CompareArgs,
        ))
    return tools


# ========== 工具使用示例 ==========
if __name__ == "__main__":
    print(calculator.invoke({"expression": "200 * (1 + 0.15)"}))
    print(date_processor.invoke({"query": "今天"}))
    print(date_processor.invoke({"query": "30天后"}))
    print(file_exporter.invoke({}))