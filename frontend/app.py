from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.orchestrator import NoveltyOrchestrator


DATA_DIR = ROOT / "data"
SAMPLE_DISCLOSURE = DATA_DIR / "sample_disclosures" / "wireless_handover.txt"
PATENT_CORPUS = DATA_DIR / "sample_patents.json"


@st.cache_resource
def load_orchestrator() -> NoveltyOrchestrator:
    return NoveltyOrchestrator(PATENT_CORPUS)


def load_sample_text() -> str:
    return SAMPLE_DISCLOSURE.read_text(encoding="utf-8")


def main() -> None:
    st.set_page_config(page_title="IP AgentLab", layout="wide")
    st.title("IP AgentLab")
    st.caption("Patent novelty search MVP: disclosure parsing, keyword expansion, local prior art retrieval, risk analysis, and report generation.")

    with st.sidebar:
        st.header("Analysis Settings")
        limit = st.slider("Prior art candidates", min_value=3, max_value=8, value=6)
        st.info("MVP uses a local sample patent corpus. External patent APIs, Qdrant, and LLM calls are planned extension points.")

    default_text = load_sample_text()
    disclosure_text = st.text_area(
        "Technical disclosure",
        value=default_text,
        height=360,
        help="Paste a technical disclosure or use the included wireless handover sample.",
    )

    if st.button("Run novelty analysis", type="primary"):
        if not disclosure_text.strip():
            st.warning("Please provide a technical disclosure.")
            return

        with st.spinner("Running multi-agent novelty workflow..."):
            result = load_orchestrator().run(disclosure_text, limit=limit)
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        return

    st.success(f"Analysis completed: {result.task_id}")

    tab_summary, tab_keywords, tab_results, tab_report = st.tabs(
        ["Disclosure", "Keywords", "Prior Art", "Report"]
    )

    with tab_summary:
        st.subheader(result.disclosure.title)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Technical field**")
            st.write(result.disclosure.technical_field)
            st.markdown("**Problem**")
            st.write(result.disclosure.problem)
        with col2:
            st.markdown("**Solution**")
            st.write(result.disclosure.solution)

        st.markdown("**Innovation points**")
        for point in result.disclosure.innovation_points:
            st.write(f"- {point}")

    with tab_keywords:
        st.markdown("**Core terms**")
        st.write(", ".join(result.keywords.core_terms) or "N/A")
        st.markdown("**Expanded terms**")
        st.write(", ".join(result.keywords.synonyms) or "N/A")
        st.markdown("**Query groups**")
        for query in result.keywords.query_groups:
            st.code(query)
        st.markdown("**Classification hints**")
        st.write(", ".join(result.keywords.classification_hints))

    with tab_results:
        comparisons_by_id = {item.patent_id: item for item in result.comparisons}
        for idx, search_result in enumerate(result.search_results, start=1):
            doc = search_result.document
            comparison = comparisons_by_id.get(doc.patent_id)
            risk_label = comparison.risk_level.value if comparison else "N/A"
            with st.expander(f"{idx}. {doc.patent_id} | {risk_label} | {doc.title}", expanded=idx <= 3):
                st.write(doc.abstract)
                st.markdown(f"**Retrieval score:** {search_result.score}")
                st.markdown(f"**Matched terms:** {', '.join(search_result.matched_terms) or 'N/A'}")
                st.markdown(f"**Assignee:** {doc.assignee or 'N/A'}")
                st.markdown(f"**Publication date:** {doc.publication_date or 'N/A'}")
                if doc.url:
                    st.markdown(f"[Open patent]({doc.url})")

                if comparison:
                    st.markdown("**Overlaps**")
                    for item in comparison.overlaps:
                        st.write(f"- {item}")
                    st.markdown("**Differences**")
                    for item in comparison.differences:
                        st.write(f"- {item}")
                    st.markdown("**Recommendation**")
                    st.write(comparison.recommendation)

    with tab_report:
        st.download_button(
            label="Download Markdown report",
            data=result.report_markdown or "",
            file_name=f"{result.task_id}_novelty_report.md",
            mime="text/markdown",
        )
        st.markdown(result.report_markdown or "")


if __name__ == "__main__":
    main()
