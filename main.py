import yaml
from pathlib import Path
import logging
import time

from docling_core.types.doc import ImageRefMode
from docling.document_converter import InputFormat, DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions

# --- VLM 管道导入 ---
from src.vlm_enrichment_pipeline_options import VLMEnrichmentPipelineOptions
from src.vlm_enrichment_pipeline import VlmEnrichmentPipeline
from src.vlm_enrichment_pipeline_word import VlmEnrichmentWordPipeline


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def docs_to_md(input_doc_path: Path, output_md_path: Path):
    logging.basicConfig(level=logging.INFO)
    
    # --- 1. 加载配置  ---
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        log.error("错误: config.yaml 文件未找到。")
        return
    except Exception as e:
        log.error(f"加载 config.yaml 出错: {e}")
        return

    # --- 2. 配置 VLMEnrichmentPipelineOptions ---
    pipeline_options = VLMEnrichmentPipelineOptions()
    
    # Docling 基础配置
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True # 必须为 True，VLM 才能获取图片
    
    # 自定义增强配置
    # VlmEnrichmentPipeline (PDF) 会读取所有三个
    # VlmPicEnrichmentWordPipeline (Word) 只会读取 do_pic_enrichment 和 do_formula_vlm_recognition
    pipeline_options.do_formula_vlm_recognition = True # 对 PDF 和 Word 均生效
    pipeline_options.do_table_enrichment = True        # 仅对 PDF 生效
    pipeline_options.do_pic_enrichment = True          # 对 PDF 和 Word 均生效
    
    # 加速器配置
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU,
        num_threads=4
    )
    pipeline_options.do_ocr = False # PDF 选项

    # 注入 VLM 配置
    pipeline_options.vlm_api_key = config.get('VLM', {}).get('api_key')
    pipeline_options.vlm_base_url = config.get('VLM', {}).get('base_url')
    pipeline_options.vlm_model = config.get('VLM', {}).get('model')
    # 注入 llm 配置
    pipeline_options.llm_api_key = config.get('LLM', {}).get('api_key')
    pipeline_options.llm_base_url = config.get('LLM', {}).get('base_url')
    pipeline_options.llm_model = config.get('LLM', {}).get('model')

    # --- 3. 配置 DocumentConverter  ---
    #
    #为 PDF 和 DOCX 注册各自的 VLM 管道
    #
    log.info("配置 DocumentConverter...")
    doc_converter = DocumentConverter(
        format_options={
            # 选项 A: PDF 文件
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmEnrichmentPipeline, 
                pipeline_options=pipeline_options,
            ),
            
            # 选项 B: DOCX 文件
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=VlmEnrichmentWordPipeline, 
                pipeline_options=pipeline_options,
            ),
        }
    )

    # --- 4. 执行转换  ---
    log.info(f"开始转换: {input_doc_path.name}")
    res = doc_converter.convert(input_doc_path)
    
    log.info(f"转换完成，保存到: {output_md_path.name}")
    res.document.save_as_markdown(
        filename=output_md_path,
        image_mode=ImageRefMode.EMBEDDED # 嵌入 Base64
    )

if __name__ == "__main__":
    start = time.time()
    
    # --- 在此切换测试文件 ---
    # input_doc_path: Path = Path(r".\test\test.pdf")
    input_doc_path: Path = Path(r".\test\test.docx") # 
    
    output_md_path: Path = Path(r".\test\test_word.md")
    
    docs_to_md(input_doc_path, output_md_path)
    
    end = time.time()
    log.info(f"总转换时间为：{end-start:.6f} 秒。")