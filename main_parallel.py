import logging
import yaml
import os
import tempfile
import concurrent.futures
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from docling_core.types.doc import ImageRefMode
from docling.document_converter import InputFormat, DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions

from src.vlm_enrichment_pipeline_options import VLMEnrichmentPipelineOptions
from src.vlm_enrichment_pipeline import VlmEnrichmentPipeline

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def split_pdf(input_path: Path, output_dir: Path, pages_per_chunk: int) -> list[Path]:
    """
    将一个PDF文件按指定页数拆分成多个小PDF文件。

    拆分后的文件将保存在 output_dir / {input_filename}_chunks / 目录下。

    返回: 排序好的PDF块路径列表。
    """

    # 1. 获取PDF文件名 (不含.pdf后缀)
    pdf_file_name_stem = input_path.stem

    # 2. 构建新的块输出目录路径
    #    例如: output_dir / "my_large_document_chunks"
    chunks_output_dir = output_dir / f"{pdf_file_name_stem}_chunks"

    # 3. 创建目录
    try:
        chunks_output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"已创建/确保块目录存在: {chunks_output_dir}")
    except Exception as e:
        log.error(f"创建块目录 {chunks_output_dir} 失败: {e}")
        # 如果目录创建失败，抛出异常停止执行
        raise

    log.info(f"开始拆分PDF: {input_path} (每块 {pages_per_chunk} 页)")

    try:
        reader = PdfReader(input_path)
    except Exception as e:
        log.error(f"读取PDF {input_path} 失败: {e}")
        return [] 

    total_pages = len(reader.pages)
    chunk_paths = []

    for i in range(0, total_pages, pages_per_chunk):
        writer = PdfWriter()
        start_page = i
        end_page = min(i + pages_per_chunk, total_pages)

        chunk_filename = chunks_output_dir / f"chunk_{i:04d}.pdf"

        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        try:
            with open(chunk_filename, "wb") as f_out:
                writer.write(f_out)
        except Exception as e:
            log.error(f"写入PDF块 {chunk_filename} 失败: {e}")
            continue  

        chunk_paths.append(chunk_filename)
        log.info(f"已创建块: {chunk_filename} (页 {start_page+1}-{end_page})")

    return chunk_paths

class VlmApiRateLimitError(Exception):
    """自定义一个异常，用于429错误"""
    pass

def process_chunk(job_details: dict) -> Path:
    """
    处理单个PDF块并将其转换为Markdown。
    此函数将在单独的进程中运行。

    job_details 包含:
    - 'input_pdf_chunk': Path, 输入的PDF块路径
    - 'output_md_chunk': Path, 输出的MD块路径
    - 'config': dict, 全局配置
    """
    input_path = job_details["input_pdf_chunk"]
    output_path = job_details["output_md_chunk"]
    config = job_details["config"]

    # [tenacity] 将日志与块文件名关联，以便于调试
    chunk_log_prefix = f"[{input_path.name}]"

    log.info(f"开始处理块: {input_path.name}")

    # --- 将可重试的工作定义为一个内部函数 ---
    # @retry 装饰器将自动捕获此函数中的异常并根据策略重试
    @retry(
        # 等待 5s, 10s, 20s, 30s, 30s...
        wait=wait_exponential(multiplier=1, min=5, max=30),
        # 停止策略: 在 5 次尝试后停止重试
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
    )
    def _convert_with_retry():
        # [tenacity] 在每次尝试时重新记录，以便我们知道重试正在发生
        log.info(f"{chunk_log_prefix} 正在调用转换器 (重新尝试连接)...")
        # --- 1. 重新构建 VLMEnrichmentPipelineOptions ---
        # 必须在子进程中重新创建不可序列化(pickle)的对象
        pipeline_options = VLMEnrichmentPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.do_formula_vlm_recognition = True
        pipeline_options.do_table_enrichment = True
        pipeline_options.do_pic_enrichment = True
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CPU,  # 禁用GPU
            num_threads=4
        )

        # pipeline_options.do_ocr = config.get('OCR', {}).get('enabled', False)
        pipeline_options.do_ocr = False

        pipeline_options.vlm_max_concurrency = config.get("VLM", {}).get(
            "max_concurrency", 10
        )
        pipeline_options.vlm_api_key = config.get("VLM", {}).get("api_key")
        pipeline_options.vlm_base_url = config.get("VLM", {}).get("base_url")
        pipeline_options.vlm_model = config.get("VLM", {}).get("model")

        # --- 2. 重新构建 DocumentConverter ---
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmEnrichmentPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )

        # --- 3. 执行转换 ---
        res = doc_converter.convert(input_path)

        # --- 4. 保存为Markdown ---
        res.document.save_as_markdown(
            filename=output_path,
            image_mode=ImageRefMode.EMBEDDED,
        )

        log.info(f"完成处理块: {output_path.name}")
        return output_path
    
    # --- `process_chunk` 的主逻辑 ---
    try:
        # 调用这个带重试的内部函数
        result_path = _convert_with_retry()
        
        log.info(f"{chunk_log_prefix} 成功处理完成。")
        return result_path
        
    except Exception as e:
        # 如果 tenacity 在 5 次尝试后 *仍然* 失败，
        # 它将抛出它捕获到的最后一个异常。
        log.error(f"{chunk_log_prefix} 彻底失败 (已达最大重试次数): {e}", exc_info=True)
        return None


def merge_markdown_files(md_files: list[Path], output_path: Path):
    """
    按顺序合并多个Markdown文件。
    """
    log.info(f"开始合并 {len(md_files)} 个Markdown文件到 {output_path}")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for md_file in md_files:
            if md_file is None:
                log.warning("跳过一个失败的块。")
                continue

            try:
                with open(md_file, "r", encoding="utf-8") as f_in:
                    f_out.write(f_in.read())

                f_out.write("\n\n---\n\n")

            except Exception as e:
                log.error(f"合并文件 {md_file} 时出错: {e}")

    log.info(f"合并完成: {output_path}")


def docs_to_md_parallel(
    input_doc_path: Path,
    output_md_path: Path,
    pages_per_chunk: int = 10,
    max_workers: int = None,
):
    """
    并行处理PDF到Markdown的转换。
    """

    # --- 1. 加载配置 (仅在主进程中加载一次) ---
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        log.error("错误: config.yaml 文件未找到。")
        return
    except Exception as e:
        log.error(f"加载 config.yaml 出错: {e}")
        return

    if max_workers is None:
        max_workers = os.cpu_count() // 2  # 默认为CPU核心数/2
    log.info(f"使用 {max_workers} 个并行工作进程。")

    # --- 2. 创建临时目录 ---
    # 使用 tempfile.TemporaryDirectory 确保所有块文件在完成后自动清理
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        log.info(f"临时目录已创建: {temp_dir}")

        # --- 3. 步骤 1: 拆分PDF ---
        try:
            pdf_chunk_paths = split_pdf(input_doc_path, temp_dir, pages_per_chunk)
        except Exception as e:
            log.error(f"拆分PDF失败: {e}")
            return

        if not pdf_chunk_paths:
            log.warning("未生成任何PDF块，任务中止。")
            return

        # --- 4. 步骤 2: 并行处理 ---

        # 准备任务列表
        jobs = []
        for pdf_chunk_path in pdf_chunk_paths:
            # 为每个PDF块定义一个对应的MD输出路径
            md_chunk_path = pdf_chunk_path.with_suffix(".md")
            jobs.append(
                {
                    "input_pdf_chunk": pdf_chunk_path,
                    "output_md_chunk": md_chunk_path,
                    "config": config,  # 将配置字典传递给每个子进程
                }
            )

        ordered_md_paths = []

        # 使用 ProcessPoolExecutor 来管理进程池
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # 使用 executor.map 来提交所有任务
            # executor.map 会按照 jobs 列表的顺序返回结果
            try:
                # `list()` 会强制执行所有任务并等待它们完成
                ordered_md_paths = list(executor.map(process_chunk, jobs))
            except Exception as e:
                log.error(f"并行处理期间发生严重错误: {e}")
                return

        # --- 5. 步骤 3: 合并结果 ---
        # 过滤掉失败的块 (在 process_chunk 中返回了 None)
        valid_md_paths = [path for path in ordered_md_paths if path is not None]

        if not valid_md_paths:
            log.error("所有处理块均失败，无法生成最终文档。")
            return

        merge_markdown_files(valid_md_paths, output_md_path)

        log.info("所有步骤完成！")


if __name__ == "__main__":
    from time import time

    start = time()

    input_file = Path(
        r".\test\test.pdf"
    )
    output_file = Path(
        r".\test\test.md"
    )

    # 设置每个PDF块的页数（例如10页）
    PAGES_PER_CHUNK = 10

    # 设置并行数 (例如 4)
    # 这里 并行数×每个进程的API并发数量 应该 ≤ API供应商的并发请求上限
    # MAX_WORKERS = 4

    docs_to_md_parallel(
        input_file,
        output_file,
        pages_per_chunk=PAGES_PER_CHUNK,
        # max_workers=MAX_WORKERS
    )

    end = time()
    log.info(f"总转换时间为：{end-start:.6f} 秒。")
