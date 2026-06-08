from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageFilter, ImageOps


class PerceptualImageFeatureExtractor:
    """Small offline visual descriptor for the design FTO MVP.

    This is intentionally lightweight so the demo works without model downloads.
    The interface can later be replaced by CLIP, SigLIP, or another multimodal
    embedding model.
    """

    def extract_from_path(self, path: str | Path) -> List[float]:
        with Image.open(path) as image:
            return self.extract(image)

    def extract_from_bytes(self, data: bytes) -> List[float]:
        with Image.open(BytesIO(data)) as image:
            return self.extract(image)

    def extract(self, image: Image.Image) -> List[float]:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        color_features = self._color_histogram(normalized)
        edge_features = self._edge_histogram(normalized)
        shape_features = self._shape_features(normalized)
        return self._normalize(color_features + edge_features + shape_features)

    def similarity(self, left: List[float], right: List[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return round(numerator / (left_norm * right_norm), 4)

    def _color_histogram(self, image: Image.Image) -> List[float]:
        resized = image.resize((96, 96))
        histogram = resized.histogram()
        total = max(sum(histogram), 1)
        bins = []
        for channel_start in (0, 256, 512):
            channel = histogram[channel_start : channel_start + 256]
            for index in range(0, 256, 32):
                bins.append(sum(channel[index : index + 32]) / total)
        return bins

    def _edge_histogram(self, image: Image.Image) -> List[float]:
        gray = ImageOps.grayscale(image.resize((96, 96)))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        histogram = edges.histogram()
        total = max(sum(histogram), 1)
        return [sum(histogram[index : index + 32]) / total for index in range(0, 256, 32)]

    def _shape_features(self, image: Image.Image) -> List[float]:
        gray = ImageOps.grayscale(image.resize((96, 96)))
        bbox = ImageOps.invert(gray).getbbox()
        width, height = image.size
        aspect_ratio = width / max(height, 1)
        if not bbox:
            return [aspect_ratio, 0.0, 0.0, 0.0]

        left, upper, right, lower = bbox
        box_width = right - left
        box_height = lower - upper
        area_ratio = (box_width * box_height) / (96 * 96)
        center_x = ((left + right) / 2) / 96
        center_y = ((upper + lower) / 2) / 96
        return [aspect_ratio, area_ratio, center_x, center_y]

    def _normalize(self, values: Iterable[float]) -> List[float]:
        values = list(values)
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]
