# Docling-VLM2MD: 基于 Docling 和 VLM、LLM 的  PDF 、WORD 文档增强转换器



本项目是一个基于 `docling` 框架的深度 PDF 处理工具。它利用多模态模型 (VLM) 的强大能力，在将 PDF 转换为 Markdown 的过程中，对文档中的**图片、表格和公式**进行智能识别和增强。

传统的 PDF 转换工具在处理复杂布局（如图表、公式）时效果不佳。本项目通过 VLM 为这些元素生成丰富的、机器可读的描述或修复其结构，极大提升了转换后 Markdown 文档的质量和可用性。



## ✨ 项目特点 (Features)



- **公式识别 (Formula Recognition) (🔧 更新)**

  - **针对 PDF:** 利用 VLM (如 `glm-4v-plus-0111`) 识别 PDF 中的公式**图片**，并将其自动转换为 `$$...$$` 格式的 LaTeX 代码。
  - **针对 DOCX (✨ 新增):** 利用纯文本 LLM 解析 Word 文档中的 `FormulaItem.text` (例如 $$a^2 + b^2 = c^2$$)，并将其转换为标准 LaTeX。

  **图片描述 (Image Description) (🔧 更新)**

  - 自动为 **PDF 和 DOCX** 文档中的图片（`PictureItem`）调用 VLM 生成详细的描述性文字（Alt-text），并将描述作为 `DescriptionAnnotation` 注入，提升 Markdown 的无障碍访问性和可索引性。
  - (✨ 新增) 支持处理 Word 文档中嵌入的 **Visio 绘图**（通过将RGBA转为JEPG格式）。
- **表格修复 (Table Repair):** 智能检测 PDF 中结构异常或跨页的表格。
  - 对于**正常**表格，直接转换为 Markdown。
  - 对于**异常**表格（如列混乱、跨页断裂），则调用 VLM 进行“截图识别与修复”，将其转换为结构完整的 Markdown 表格。
- **基于 Docling 框架 (🔧 更新)**

  - 项目完全构建在 `docling` 管道之上，通过自定义 `VlmEnrichmentPipeline` (用于 PDF) 和 `VlmEnrichmentWordPipeline` (用于 DOCX) (✨ 新增) 以及多个增强模型实现，逻辑清晰，易于扩展。
- **并发与并行处理:**
  
  - `main.py`: 提供**同步执行**方法，使用 `ThreadPoolExecutor` 在进程内并发调用 VLM API，适用于中小型文档和调试。
  - `main_parallel.py`: 提供**多进程并行**处理方法，先将大型 PDF 拆分为多个小块，再使用 `ProcessPoolExecutor` 并行处理，极大提升大型文档的转换效率。



## 🔧 安装说明 (Installation)



本项目使用 `uv` 作为包管理和虚拟环境工具，以实现快速的依赖解析。

```bash
# 1. 克隆项目
git clone https://github.com/yjryjrhehe/docling-vlm2md.git
cd docling-vlm2md

# 2. (推荐) 使用 uv 创建虚拟环境
# 如果你还没有安装 uv, 运行: pip install uv
uv venv

# 3. 激活环境
# Windows (PowerShell):
.\.venv\Scripts\Activate
# macOS / Linux (bash):
source .venv/bin/activate

# 4. 安装所有依赖
# uv 会自动读取 pyproject.toml 和 uv.lock 文件
uv sync
```



## ⚙️ 配置说明 (Configuration)



在运行项目之前，你**必须**在项目根目录创建一个 `config.yaml` 文件，用于配置 VLM API 的相关凭据。

**`config.yaml` 文件内容示例:**

```yaml
VLM:
  api_key: "sk-your_api_key_here" # 你的 VLM API 密钥 (例如智谱AI的API Key)
  base_url: "https://open.bigmodel.cn/api/paas/v4/" # VLM API 的端点 (Endpoint)
  model: "glm-4v-plus-0111" # 你希望使用的多模态模型
  max_concurrency: 5 # 单个进程内VLM API调用的最大并发线程数
  
LLM:
  api_key: "your_api_key"
  base_url: "https://open.bigmodel.cn/api/paas/v4/"
  model: "glm-4-flashx-250414"
  max_concurrency: 5
```

> **重要提示:** `max_concurrency` 控制**单个脚本**（如 `main.py`）或**单个并行工作进程**（在 `main_parallel.py` 中）的 API 并发线程数。请根据你的 VLM API 供应商的速率限制（Rate Limit）来设置此值。



## 🚀 使用示例 (Usage)



本项目提供两种运行模式：同步处理和多进程并行处理。



### 1. 同步处理 (main.py)



此方法适用于处理中小型 PDF 文档（如 50 页以下）或进行功能调试。它将在一个进程中按顺序解析文档，并使用 `ThreadPoolExecutor` 并发调用 VLM API。

**使用方法:**

修改 `main.py` 的 `if __name__ == "__main__":` 部分，指定输入和输出路径，然后运行：

```python
# main.py
if __name__ == "__main__":
    start = time.time()
    
    # 既可以处理 PDF:
    # input_doc_path = Path(r"path/to/your/document.pdf")
    
    # 也可以处理 DOCX:
    input_doc_path = Path(r"path/to/your/document.docx")
    output_md_path = Path(r"path/to/your/output.md")
    
    docs_to_md(input_doc_path, output_md_path)
    
    end = time.time()
    log.info(f"总转换时间为：{end-start:.6f} 秒。")
```

然后执行：

```python
python main.py
```



### 2. 多进程并行处理 (main_parallel.py)



此方法专为处理大型 PDF 文档（如 50 页以上）而设计。它会先将 PDF 拆分成多个块（例如每 10 页一块），然后启动多个**工作进程**并行处理这些块。

**使用方法:**

修改 `main_parallel.py` 的 `if __name__ == "__main__":` 部分，指定路径和并行参数：

```python
# main_parallel.py
if __name__ == "__main__":
    start = time()

    input_file = Path(r"path/to/your/large_document.pdf")
    output_file = Path(r"path/to/your/output.md")

    # [可调参数] 设置每个PDF块的页数（例如10页）
    PAGES_PER_CHUNK = 10
    
    # [可调参数] 设置并行工作进程数 (默认为 CPU 核心数 / 2)
    # MAX_WORKERS = 4 

    docs_to_md_parallel(
        input_file,
        output_file,
        pages_per_chunk=PAGES_PER_CHUNK,
        # max_workers=MAX_WORKERS # 可以不传，使用默认值
    )

    end = time()
    log.info(f"总转换时间为：{end-start:.6f} 秒。")
```

然后执行：

```bash
python main_parallel.py
```



## ⚠️ 关于并行处理和 429 错误 (重要)



在使用 `main_parallel.py` 时，你必须小心计算**总并发数**，以避免超出 VLM API 供应商的速率限制（这会导致 `429 Rate Limit Exceeded` 错误）。

**总并发数 = `MAX_WORKERS` (进程数) × `max_concurrency` (每个进程的线程数)**

- `MAX_WORKERS`：在 `main_parallel.py` 中设置（或使用默认值）。
- `max_concurrency`：在 `config.yaml` 中设置。

**示例：** 如果你的 API 供应商**总并发上限**为 20 次请求/秒：

- **安全配置 1：** `MAX_WORKERS = 4` (个进程), `max_concurrency = 5` (线程/进程)
  - 总并发 = 4 * 5 = 20
- **安全配置 2：** `MAX_WORKERS = 2` (个进程), `max_concurrency = 10` (线程/进程)
  - 总并发 = 2 * 10 = 20
- **错误配置：** `MAX_WORKERS = 8`, `max_concurrency = 5`
  - 总并发 = 8 * 5 = 40 (**远超上限，将导致 429 错误**）

> **注意：** 由于 API 并发限制，多进程并行（`main_parallel.py`）确实很容易遇到 429 错误。`main_parallel.py` 中的 `process_chunk` 函数已集成了 `tenacity` 库，在遇到 API 错误时会自动进行**指数退避重试**（最多 5 次），这有助于缓解瞬时的速率限制，提高大型文档处理的成功率。



## 📁 支持的文件类型 (Supported Formats)



- **输入:**
  
  - `PDF (.pdf)`: **主要支持**，具备完整的 VLM 增强（公式、图片、表格）。
  
    `Word (.docx)`: **增强支持** (✨ 新增)。
  
    - ✅ 图片描述 (VLM)
    - ✅ Visio 绘图描述 (VLM)
    - ✅ 公式转换 (LLM 文本转 LaTeX)
    - ❌ 表格 VLM 修复 (暂不支持)
- **输出:**
  - `Markdown (.md)`: 最终输出格式。根据 `main.py` 中的设置 (`ImageRefMode.EMBEDDED`)，所有图片（包括公式）将被转换为 Base64 嵌入 Markdown 中。（也可以选择“REFERENCED”模式，会在输出路径中新建文件夹保存所有图片，在输出的md文档中仅保留图片引用路径，）



## 🏗️ 项目架构 (Project Architecture)



```
📦docling-vlm2md
 ┣ 📂src
 ┃ ┣ 📂enrichment_models           # 核心 VLM/LLM 增强模型
 ┃ ┃ ┣ 📜formula_enrichment_model.py      # (PDF) 公式VLM识别模型
 ┃ ┃ ┣ 📜formula_enrichment_model_word.py # (DOCX) 公式LLM转换模型 (✨ 新增)
 ┃ ┃ ┣ 📜pic_enrichment_model.py          # (PDF/DOCX) 图片VLM描述模型
 ┃ ┃ ┗ 📜table_enrichment_model.py        # (PDF) 表格VLM修复模型
 ┃ ┣ 📂prompt                        # VLM 提示词 (Prompts)
 ┃ ┃ ┣ 📜formula_recognition_prompt.py
 ┃ ┃ ┣ 📜table_repair_prompt.py
 ┃ ┃ ┗ 📜VLM_prompt.py
 ┃ ┣ 📜vlm_enrichment_pipeline.py         # (PDF) 自定义 Docling 管道 (✨ 更新)
 ┃ ┣ 📜vlm_enrichment_pipeline_word.py    # (DOCX) 自定义 Docling 管道 (✨ 新增)
 ┃ ┗ 📜vlm_enrichment_pipeline_options.py # 管道配置项 (VLMEnrichmentPipelineOptions)
 ┣ 📜.gitignore
 ┣ 📜.python-version
 ┣ 📜config.yaml                      # 配置文件 (需用户本地创建)
 ┣ 📜main.py                          # 同步执行入口 (支持 PDF/DOCX)
 ┣ 📜main_parallel.py                 # 多进程并行执行入口 (仅支持 PDF)
 ┣ 📜pyproject.toml
 ┣ 📜README.md                        # 本文档
 ┗ 📜uv.lock
```



## 💡 常见问题解答 (FAQ)



**Q: 为什么我运行 `main.py` 或 `main_parallel.py` 时立即失败并提示 `config.yaml` 未找到？** A: 你需要在项目根目录（与 `main.py` 同级）**手动创建**一个 `config.yaml` 文件，并按照 配置说明 部分的格式填入你的 VLM LLM API 密钥和端点。

**Q: 我遇到了 429 (Rate Limit Exceeded) 错误，该怎么办？** A: 这是因为你的总并发请求数超过了 API 限制。请参考 关于并行处理和 429 错误 (重要) 部分的说明：

1. **如果你使用 `main.py`**: 降低 `config.yaml` 中的 `max_concurrency` 值。
2. **如果你使用 `main_parallel.py`**: 确保 `MAX_WORKERS` × `max_concurrency` 的乘积在你的 API 限制之内。建议优先降低 `MAX_WORKERS`（进程数）。

**Q: 公式识别的效果不好，总是返回 "[公式识别失败]"。** A: 请检查：

1. **API 凭据:** `config.yaml` 中的 `api_key` 和 `base_url` 是否正确。
2. **网络连接:** 确认你的服务器可以访问 `base_url`。
3. **模型能力:** 确认你选用的 `model` (如 `glm-4v-plus-0111`) 具备强大的公式 OCR 能力。
4. **Prompt 调优:** `src/prompt/formula_recognition_prompt.py` 中的提示词可能需要针对你的特定模型进行微调。



## 🤝 贡献指南 (Contributing)

如果你有任何改进意见或发现了 Bug，请随时提交 Issue 或 Pull Request。



## 📄 License



本项目遵循 MIT 协议。