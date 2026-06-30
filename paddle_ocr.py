# -*- coding: utf-8 -*-
"""
OCR 模块（RapidOCR 引擎，基于 ONNX Runtime）
"""

import io
import os
import re

import numpy as np
from PIL import Image


class PaddleOCREngine:
    """OCR 引擎（名称保持兼容，实际使用 RapidOCR）"""
    _instance = None
    _ocr = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self):
        if self._ocr is not None:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            print("[INFO] RapidOCR 模型加载成功 (ONNX Runtime)")
        except ImportError as e:
            raise ImportError(
                f"RapidOCR 未正确安装: {e}\n"
                f"请运行: uv sync"
            )

    def ocr_image(self, image_path: str) -> str:
        self._ensure_loaded()
        try:
            result, _ = self._ocr(image_path)
            return self._extract_text(result)
        except Exception as e:
            return f"[OCR错误: {str(e)}]"

    def ocr_pdf(self, pdf_path: str) -> str:
        self._ensure_loaded()
        try:
            import fitz
        except ImportError:
            return "[OCR错误: PyMuPDF 未安装]"

        doc = fitz.open(pdf_path)
        all_lines = []

        for i, page in enumerate(doc):
            # 按最大 2000px 计算 DPI，避免超大图拖慢推理
            page_rect = page.rect
            max_dim = max(page_rect.width, page_rect.height)
            dpi = min(150, int(2000 * 72 / max_dim))  # 72 points per inch

            pix = page.get_pixmap(dpi=dpi)
            # 直接转 numpy 数组传给 RapidOCR，不落盘
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            try:
                result, _ = self._ocr(img_array)
                text = self._extract_text(result)
                if text:
                    all_lines.append(text)
            except Exception as e:
                print(f"[WARN] OCR 第 {i + 1} 页失败: {e}")

            print(f"[INFO] OCR PDF 第 {i + 1}/{len(doc)} 页完成")

        doc.close()
        return "\n".join(all_lines)

    def _extract_text(self, result) -> str:
        """从 RapidOCR 结果中提取文字。
        RapidOCR 返回格式: [[box, text, score], ...] 或 None
        """
        if result is None:
            return ""
        lines = []
        for item in result:
            if len(item) >= 2:
                text = item[1]
                text = PaddleOCREngine._clean_text(text)
                if text:
                    lines.append(text)
        return "\n".join(lines)

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r'(?<=[A-Za-z0-9])\s+(?=[A-Za-z0-9])', '', text)
        text = ' '.join(text.split())
        return text


# ── 便捷函数 ──
def ocr_image(image_path: str) -> str:
    return PaddleOCREngine().ocr_image(image_path)


def ocr_pdf(pdf_path: str) -> str:
    return PaddleOCREngine().ocr_pdf(pdf_path)
