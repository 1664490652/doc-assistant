# -*- coding: utf-8 -*-
"""
OCR 模块（PaddleOCR 引擎）
使用 PaddleOCR 3.x 进行中文 + 英文的 OCR 识别。
"""

import io
import os
import re
import tempfile
import numpy as np
from PIL import Image


class PaddleOCREngine:
    """PaddleOCR 单例引擎，首次调用时加载模型"""

    _instance = None
    _ocr = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self):
        """延迟加载模型（首次调用时加载），自动检测 GPU"""
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
            import paddle

            # 检测 GPU 是否可用
            gpu_available = paddle.is_compiled_with_cuda()
            device = "gpu" if gpu_available else "cpu"
            if gpu_available:
                print(f"[INFO] PaddleOCR 检测到 GPU，启用 CUDA 加速")
            else:
                print(f"[INFO] PaddleOCR 未检测到 GPU，使用 CPU 推理")

            self._ocr = PaddleOCR(lang='ch', use_angle_cls=False, device=device)
            print(f"[INFO] PaddleOCR 模型加载成功 (device={device})")
        except ImportError:
            raise ImportError(
                "PaddleOCR 未安装，请运行: uv pip install paddleocr paddlepaddle"
            )

    def ocr_image(self, image_path: str) -> str:
        """对单张图片执行 OCR，返回纯文本"""
        self._ensure_loaded()

        try:
            result = self._ocr.predict(image_path)
            return self._extract_text(result)
        except Exception as e:
            return f"[OCR错误: {str(e)}]"

    def ocr_pdf(self, pdf_path: str) -> str:
        """对 PDF 执行 OCR（逐页渲染为图片后识别）"""
        self._ensure_loaded()

        try:
            import fitz
        except ImportError:
            return "[OCR错误: PyMuPDF 未安装]"

        doc = fitz.open(pdf_path)
        all_lines = []

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)  # 降低 DPI 避免推理引擎内存不足
            tmp_path = os.path.join(tempfile.gettempdir(), f"paddle_tmp_{i}.png")
            pix.save(tmp_path)

            # 限制图片尺寸，避免推理引擎崩溃
            img = Image.open(tmp_path)
            w, h = img.size
            if w > 2000 or h > 2000:
                scale = min(2000 / w, 2000 / h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                img.save(tmp_path)

            try:
                result = self._ocr.ocr(tmp_path)
                text = self._extract_text(result)
                if text:
                    all_lines.append(text)
            finally:
                try:
                    os.remove(tmp_path)
                except:
                    pass
            print(f"[INFO] PaddleOCR PDF 第 {i+1}/{len(doc)} 页完成")

        return "\n".join(all_lines)

    @staticmethod
    def _extract_text(ocr_result) -> str:
        """从 PaddleOCR 3.x 结果中提取纯文本"""
        if not ocr_result:
            return ""

        lines = []
        for page in ocr_result:
            if not isinstance(page, dict):
                continue
            rec_texts = page.get("rec_texts", [])
            for text in rec_texts:
                text = PaddleOCREngine._clean_text(text)
                if text:
                    lines.append(text)
        return "\n".join(lines)

    @staticmethod
    def _clean_text(text: str) -> str:
        """后处理清洗：去除多余空格、修复数字串"""
        if not text:
            return ""

        text = text.strip()

        # 修复数字/字母串中被错误插入的空格
        # 例如 "SS33 2233004477009900" → "SS332233004477009900"
        text = re.sub(r'(?<=[A-Za-z0-9])\s+(?=[A-Za-z0-9])', '', text)

        # 去除连续空格
        text = ' '.join(text.split())

        return text


# 便捷函数
def ocr_image(image_path: str) -> str:
    return PaddleOCREngine().ocr_image(image_path)


def ocr_pdf(pdf_path: str) -> str:
    return PaddleOCREngine().ocr_pdf(pdf_path)