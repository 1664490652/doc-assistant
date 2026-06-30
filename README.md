# 文档智能助手

基于 LangChain + Streamlit 的多智能体文档分析系统，支持 PDF/Word/Excel 解析、RAG 检索、文档对比、长期记忆管理。

> **GitHub**: https://github.com/1664490652/doc-assistant

## 环境要求

- **Python ≥ 3.12, < 3.14**
- **uv** 包管理器：[安装指南](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 快速开始

**方式一：一键安装（推荐）**

右键 `setup.ps1` → "使用 PowerShell 运行"，脚本会自动：
1. 安装 uv 包管理器（如未安装）
2. 安装所有 Python 依赖
3. 检测 onnxruntime 是否正常（如失败会提示安装 VC++ 运行库）

**方式二：手动安装**

```bash
# 1. 安装依赖
uv sync

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek 和 DashScope API Key

# 3. 启动
uv run streamlit run app_streamlit.py
```

> **Windows 用户注意**：如果启动报 `onnxruntime DLL load failed`，请安装 [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) 后重试。

浏览器打开 `http://localhost:8501` 即可使用。

## API Key 获取

| 用途 | 平台 | 获取地址 |
|------|------|----------|
| 主模型 | DeepSeek | https://platform.deepseek.com/api_keys |
| 备用模型 / Embedding | 阿里云百炼 | https://dashscope.console.aliyun.com/apiKey |

两个 Key 都要填，缺一不可。

## 项目结构

```
doc-assistant/
├── app_streamlit.py      # Streamlit 主应用（UI + Agent 调度）
├── agent_graph.py        # LangGraph Agent 图管管理
├── llm_processor.py      # LLM 调用封装（摘要 / 对比 / 评估修正）
├── tools.py              # 16 个工具定义（计算 / 搜索 / 记忆 / RAG 等）
├── rag_engine.py         # ChromaDB 向量检索引擎
├── memory_tree.py        # SQLite 长期记忆存储
├── middleware_config.py  # 中间件配置（Fallback / PII / HITL）
├── document_parser.py    # 文档解析（pdfplumber + RapidOCR 回退）
├── paddle_ocr.py         # RapidOCR 引擎封装（ONNX Runtime）
├── config.py             # 全局配置加载
├── pyproject.toml        # 项目元数据与依赖
├── uv.lock               # 依赖锁文件
└── .env.example          # 环境变量模板
```

## 功能列表

- 文档解析：PDF（pdfplumber + OCR）、Word（python-docx）、Excel
- AI 摘要与对比分析（含自动评估修正）
- RAG 语义检索（长文档自动索引到 ChromaDB）
- 长期记忆（Memory Tree，SQLite 持久化）
- 中间件：模型降级 / PII 脱敏 / 高危操作确认
- 打字机效果流式输出

## 常见问题

**Q: 启动报 OCR 相关错误？**

A: OCR 基于 RapidOCR（ONNX Runtime），首次运行会自动下载模型文件（~50MB）。如果下载失败，检查网络或手动安装：
```bash
uv pip install rapidocr-onnxruntime onnxruntime
```

**Q: Memory Tree 加载失败？**

A: 首次使用会自动建表，如果已存在 `memory/documents.db` 文件损坏则删除后重启。
