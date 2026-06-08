from __future__ import annotations

import json
from pathlib import Path
from typing import List

from backend.schemas.models import DesignPatentImage


def load_design_corpus(corpus_dir: str | Path) -> List[DesignPatentImage]:
    root = Path(corpus_dir)
    metadata_path = root / "metadata.json"
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    images: List[DesignPatentImage] = []

    for record in records:
        for image_record in record.get("images", []):
            image_path = root / image_record["path"]
            images.append(
                DesignPatentImage(
                    image_id=f"{record['design_id']}::{image_record.get('view', 'view')}",
                    design_id=record["design_id"],
                    title=record["title"],
                    product_type=record.get("product_type", ""),
                    view=image_record.get("view", ""),
                    image_path=str(image_path),
                    assignee=record.get("assignee", ""),
                    publication_date=record.get("publication_date", ""),
                    jurisdiction=record.get("jurisdiction", ""),
                    source_url=record.get("source_url", ""),
                )
            )

    return images
