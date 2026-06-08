from __future__ import annotations

from backend.agents.disclosure_parser import DisclosureParserAgent


def test_parser_extracts_english_structured_disclosure() -> None:
    text = """
Title: Predictive low-latency handover method for dense 5G networks

Technical Field:
Wireless communication and mobility management.

Background:
Conventional handover reacts after signal quality degrades, causing packet loss
and service interruption.

Technical Solution:
The method predicts link degradation from beam measurements and mobility
history, ranks target cells, and preconfigures target-cell radio resources
before handover execution.

Innovation Points:
1. Predicting link degradation before handover thresholds are crossed.
2. Combining beam quality and mobility history in target-cell ranking.
3. Preconfiguring target-cell radio resources before handover.
"""

    disclosure = DisclosureParserAgent().run(text)

    assert disclosure.parser_mode == "rule"
    assert disclosure.parse_quality == "high"
    assert disclosure.title == "Predictive low-latency handover method for dense 5G networks"
    assert "Wireless communication" in disclosure.technical_field
    assert len(disclosure.innovation_points) == 3
    assert not disclosure.warnings


def test_parser_extracts_chinese_structured_disclosure() -> None:
    text = """
发明名称：一种低时延切换方法
技术领域：本发明涉及5G无线通信和移动性管理。
背景技术：现有切换通常在信号质量下降后才触发，容易造成丢包和业务中断。
技术方案：系统根据波束测量、移动历史和业务时延需求预测链路退化，并提前预配置目标小区资源。
创新点：
1. 在阈值触发前预测链路退化。
2. 综合波束质量、移动历史和时延需求进行目标小区排序。
技术效果：降低切换中断时间并提高业务连续性。
"""

    disclosure = DisclosureParserAgent().run(text)

    assert disclosure.parser_mode == "rule"
    assert disclosure.parse_quality == "high"
    assert disclosure.title == "一种低时延切换方法"
    assert "5G无线通信" in disclosure.technical_field
    assert disclosure.innovation_points == [
        "在阈值触发前预测链路退化。",
        "综合波束质量、移动历史和时延需求进行目标小区排序。",
    ]


def test_parser_uses_plain_text_fallback_for_unstructured_input() -> None:
    text = (
        "We propose a low latency handover system for dense 5G networks. "
        "Existing handover methods react too late after signal quality drops, causing packet loss. "
        "The system predicts link degradation from beam reports and mobility history, ranks target cells, "
        "and preconfigures radio resources before handover. "
        "This improves service continuity and reduces interruption time."
    )

    disclosure = DisclosureParserAgent().run(text)

    assert disclosure.parser_mode == "plain_text_fallback"
    assert disclosure.parse_quality in {"medium", "high"}
    assert "handover" in disclosure.problem.lower()
    assert "predicts link degradation" in disclosure.solution
    assert disclosure.innovation_points
    assert any("fallback" in warning for warning in disclosure.warnings)


def test_parser_marks_low_quality_short_input() -> None:
    disclosure = DisclosureParserAgent().run("A better network handover idea.")

    assert disclosure.parser_mode == "plain_text_fallback"
    assert disclosure.parse_quality == "low"
    assert disclosure.warnings
