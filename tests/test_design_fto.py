from __future__ import annotations

from pathlib import Path

from backend.agents.design_fto import DesignFTOAgent
from backend.retrieval.design_corpus import load_design_corpus
from backend.retrieval.design_image_features import PerceptualImageFeatureExtractor
from backend.schemas.models import RiskLevel


ROOT = Path(__file__).resolve().parents[1]


def test_design_corpus_loads_sample_images() -> None:
    images = load_design_corpus(ROOT / "data" / "sample_design_patents")

    assert len(images) >= 4
    assert all(Path(image.image_path).exists() for image in images)
    assert {image.product_type for image in images} >= {"chair", "bottle", "lamp"}


def test_perceptual_features_make_similar_chair_images_closer() -> None:
    extractor = PerceptualImageFeatureExtractor()
    query = extractor.extract_from_path(ROOT / "data" / "sample_design_queries" / "chair_loop_query.png")
    similar = extractor.extract_from_path(ROOT / "data" / "sample_design_patents" / "chair_loop_front.png")
    different = extractor.extract_from_path(ROOT / "data" / "sample_design_patents" / "bottle_round_front.png")

    assert extractor.similarity(query, similar) > extractor.similarity(query, different)


def test_design_fto_agent_returns_visual_results_and_report() -> None:
    query_path = ROOT / "data" / "sample_design_queries" / "chair_loop_query.png"
    result = DesignFTOAgent(ROOT / "data" / "sample_design_patents").run(
        query_image_bytes=query_path.read_bytes(),
        query_image_name=query_path.name,
        product_type="chair",
        top_k=3,
    )

    assert result.search_results
    assert result.search_results[0].image.design_id == "USD-DEMO-CHAIR-001"
    assert result.search_results[0].risk_level in {RiskLevel.high, RiskLevel.medium}
    assert "Design FTO Search Report" in (result.report_markdown or "")
