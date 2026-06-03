from __future__ import annotations

import json
from pathlib import Path
from typing import List

from backend.schemas.models import PatentDocument


def load_patent_corpus(path: str | Path) -> List[PatentDocument]:
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as rf:
        raw_items = json.load(rf)
    return [_validate_patent(item) for item in raw_items]


def _validate_patent(item: dict) -> PatentDocument:
    if hasattr(PatentDocument, "model_validate"):
        return PatentDocument.model_validate(item)
    return PatentDocument.parse_obj(item)
