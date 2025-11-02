import yaml
from pathlib import Path
import logging

from docling_core.types.doc import ImageRefMode
from docling.document_converter import InputFormat, DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions

from src.vlm_enrichment_pipeline_options import VLMEnrichmentPipelineOptions
from src.vlm_enrichment_pipeline import VlmEnrichmentPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def docs_to_md(input_doc_path: Path, output_md_path: Path):
    logging.basicConfig(level=logging.INFO)
    # --- 1. 加载配置 ---
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        log.error("错误: config.yaml 文件未找到。请确保配置文件在脚本运行目录下。")
        return
    except Exception as e:
        log.error(f"加载 config.yaml 出错: {e}")
        return

    pipeline_options = VLMEnrichmentPipelineOptions()
    # Docling 基础配置
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    # 自定义增强配置
    pipeline_options.do_formula_vlm_recognition = True
    pipeline_options.do_table_enrichment = True
    pipeline_options.do_pic_enrichment = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU,  # 禁用GPU
        num_threads=4
    )

    pipeline_options.do_ocr = False
    # 注入 VLM 配置
    pipeline_options.vlm_api_key = config.get('VLM', {}).get('api_key')
    pipeline_options.vlm_base_url = config.get('VLM', {}).get('base_url')
    pipeline_options.vlm_model = config.get('VLM', {}).get('model')
    # 定义解析word文档的配置
    docx_pipeline_options = pipeline_options.model_copy()
    docx_pipeline_options.do_code_enrichment = True
    docx_pipeline_options.do_formula_enrichment = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmEnrichmentPipeline,
                pipeline_options=pipeline_options,
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_options=docx_pipeline_options,
            ),
        }
    )
    res = doc_converter.convert(input_doc_path)
    res.document.save_as_markdown(
            filename=output_md_path,
            image_mode=ImageRefMode.EMBEDDED
        ) 

if __name__ == "__main__":
    import time
    start = time.time()
    input_doc_path: Path = Path(r".\test\test.pdf")
    output_md_path: Path = Path(r".\test\test.md")
    docs_to_md(input_doc_path, output_md_path)
    end = time.time()
    log.info(f"总转换时间为：{end-start:.6f} 秒。")
