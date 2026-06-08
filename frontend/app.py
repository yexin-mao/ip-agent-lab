from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.design_fto import DesignFTOAgent
from backend.agents.novelty_workflow import NoveltyWorkflow
from backend.agents.technical_fto_workflow import TechnicalFTOWorkflow


DATA_DIR = ROOT / "data"
SAMPLE_DISCLOSURE = DATA_DIR / "sample_disclosures" / "wireless_handover.txt"
PATENT_CORPUS = DATA_DIR / "sample_patents.json"
DESIGN_CORPUS = DATA_DIR / "sample_design_patents"
SAMPLE_DESIGN_QUERY = DATA_DIR / "sample_design_queries" / "chair_loop_query.png"


@st.cache_resource
def load_novelty_workflow() -> NoveltyWorkflow:
    return NoveltyWorkflow(PATENT_CORPUS)


@st.cache_resource
def load_technical_fto_workflow() -> TechnicalFTOWorkflow:
    return TechnicalFTOWorkflow(PATENT_CORPUS)


@st.cache_resource
def load_design_fto_agent() -> DesignFTOAgent:
    return DesignFTOAgent(DESIGN_CORPUS)


def load_sample_text() -> str:
    return SAMPLE_DISCLOSURE.read_text(encoding="utf-8")


def main() -> None:
    st.set_page_config(page_title="IP AgentLab", layout="wide")
    st.title("IP AgentLab")
    st.caption("AI-assisted IP review demo: novelty search, technical FTO, and design FTO workflows.")

    with st.sidebar:
        st.header("Analysis Mode")
        mode = st.radio("Workflow", ["Novelty Search", "Technical FTO", "Design FTO"], index=0)

    if mode == "Novelty Search":
        render_novelty_search()
    elif mode == "Technical FTO":
        render_technical_fto()
    else:
        render_design_fto()


def render_novelty_search() -> None:
    with st.sidebar:
        st.header("Novelty Settings")
        limit = st.slider("Prior-art candidates", min_value=3, max_value=8, value=6)
        st.info("Novelty mode asks whether the invention is disclosed by prior art. It does not generate FTO claim charts.")

    disclosure_text = st.text_area(
        "Technical disclosure",
        value=load_sample_text(),
        height=360,
        help="Paste an invention disclosure or use the included wireless handover sample.",
    )

    if st.button("Run novelty search", type="primary"):
        if not disclosure_text.strip():
            st.warning("Please provide a technical disclosure.")
            return
        with st.spinner("Running novelty search workflow..."):
            result = load_novelty_workflow().run(disclosure_text, limit=limit)
        st.session_state["novelty_result"] = result

    result = st.session_state.get("novelty_result")
    if not result:
        return

    st.success(f"Novelty search completed: {result.task_id}")
    tab_summary, tab_keywords, tab_results, tab_matrix, tab_rag, tab_report = st.tabs(
        ["Disclosure", "Keywords", "Prior Art", "Novelty Matrix", "RAG Analysis", "Report"]
    )

    with tab_summary:
        render_disclosure_summary(result.disclosure)

    with tab_keywords:
        render_keywords(result.keywords)

    with tab_results:
        render_evidence_results(result.evidence_results)

    with tab_matrix:
        render_novelty_matrix(result.novelty_matrix)

    with tab_rag:
        render_rag_analysis(result.generated_analysis)

    with tab_report:
        render_report_download(result.report_markdown or "", f"{result.task_id}_novelty_report.md")


def render_technical_fto() -> None:
    with st.sidebar:
        st.header("FTO Settings")
        limit = st.slider("Claim candidates", min_value=3, max_value=8, value=6)
        st.info("Technical FTO mode maps implementation features to claim evidence. It does not judge patent novelty.")

    implementation_text = st.text_area(
        "Implementation / product technical features",
        value=load_sample_text(),
        height=360,
        help="Paste a product implementation or technical feature description.",
    )

    if st.button("Run technical FTO", type="primary"):
        if not implementation_text.strip():
            st.warning("Please provide implementation features.")
            return
        with st.spinner("Running technical FTO workflow..."):
            result = load_technical_fto_workflow().run(implementation_text, limit=limit)
        st.session_state["technical_fto_result"] = result

    result = st.session_state.get("technical_fto_result")
    if not result:
        return

    st.success(f"Technical FTO completed: {result.task_id}")
    tab_summary, tab_keywords, tab_claims, tab_chart, tab_rag, tab_report = st.tabs(
        ["Implementation", "Queries", "Claim Evidence", "Claim Chart", "RAG Analysis", "Report"]
    )

    with tab_summary:
        render_disclosure_summary(result.implementation)

    with tab_keywords:
        render_keywords(result.keywords)

    with tab_claims:
        render_evidence_results(result.claim_evidence_results)

    with tab_chart:
        render_fto_claim_chart(result.fto_claim_chart)

    with tab_rag:
        render_rag_analysis(result.generated_analysis)

    with tab_report:
        render_report_download(result.report_markdown or "", f"{result.task_id}_technical_fto_report.md")


def render_disclosure_summary(disclosure) -> None:
    st.subheader(disclosure.title)
    st.caption(f"Parser: {disclosure.parser_mode} | Quality: {disclosure.parse_quality}")
    if disclosure.warnings:
        with st.expander("Parser warnings", expanded=disclosure.parse_quality == "low"):
            for warning in disclosure.warnings:
                st.write(f"- {warning}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Technical field**")
        st.write(disclosure.technical_field)
        st.markdown("**Problem / context**")
        st.write(disclosure.problem)
    with col2:
        st.markdown("**Solution / implementation summary**")
        st.write(disclosure.solution)

    st.markdown("**Extracted technical points**")
    for point in disclosure.innovation_points:
        st.write(f"- {point}")


def render_keywords(keywords) -> None:
    st.markdown("**Core terms**")
    st.write(", ".join(keywords.core_terms) or "N/A")
    st.markdown("**Expanded terms**")
    st.write(", ".join(keywords.synonyms) or "N/A")
    st.markdown("**Query groups**")
    for query in keywords.query_groups:
        st.code(query)
    st.markdown("**Classification hints**")
    st.write(", ".join(keywords.classification_hints) or "N/A")


def render_evidence_results(evidence_results) -> None:
    for idx, search_result in enumerate(evidence_results, start=1):
        with st.expander(f"{idx}. {search_result.patent_id} | {search_result.title}", expanded=idx <= 3):
            st.markdown(f"**Retrieval score:** {search_result.score}")
            st.markdown(f"**Assignee:** {search_result.assignee or 'N/A'}")
            st.markdown(f"**Publication date:** {search_result.publication_date or 'N/A'}")
            st.markdown(f"**CPC:** {', '.join(search_result.cpc) or 'N/A'}")
            if search_result.source_url:
                st.markdown(f"[Open patent]({search_result.source_url})")

            st.markdown("**Evidence chunks**")
            for chunk_result in search_result.evidence_chunks:
                chunk = chunk_result.chunk
                st.markdown(
                    f"- `{chunk.chunk_id}` | section: `{chunk.section}` | score: `{chunk_result.score}` | "
                    f"{chunk_result.retrieval_reason}"
                )
                if chunk_result.matched_terms:
                    st.caption(f"Matched terms: {', '.join(chunk_result.matched_terms)}")
                if chunk_result.query_sources or chunk_result.retrieval_sources:
                    st.caption(
                        "Query sources: "
                        f"{', '.join(chunk_result.query_sources) or 'N/A'} | "
                        "Retrieval sources: "
                        f"{', '.join(chunk_result.retrieval_sources) or 'N/A'}"
                    )
                st.write(chunk.text)


def render_novelty_matrix(novelty_matrix) -> None:
    if not novelty_matrix:
        st.info("No novelty matrix was generated.")
        return

    rows = [
        {
            "innovation_point": f"{row.innovation_point_id}: {row.innovation_point}",
            "patent": f"{row.patent_id} | {row.title}" if row.patent_id else "N/A",
            "coverage": row.coverage.value,
            "risk": row.risk_level.value,
            "score": row.score,
            "evidence": row.evidence_chunk_id or "N/A",
            "reasoning": row.reasoning,
        }
        for row in novelty_matrix
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_fto_claim_chart(fto_claim_chart) -> None:
    if not fto_claim_chart:
        st.info("No FTO claim chart was generated.")
        return

    rows = [
        {
            "technical_element": f"{row.element_id}: {row.technical_element}",
            "patent": f"{row.patent_id} | {row.title}" if row.patent_id else "N/A",
            "mapping": row.mapping.value,
            "risk": row.risk_level.value,
            "score": row.score,
            "claim_chunk": row.claim_chunk_id or "N/A",
            "reasoning": row.reasoning,
        }
        for row in fto_claim_chart
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    for row in fto_claim_chart:
        with st.expander(f"{row.element_id} | {row.mapping.value} | {row.patent_id or 'N/A'}"):
            st.markdown("**Technical element**")
            st.write(row.technical_element)
            st.markdown("**Reasoning**")
            st.write(row.reasoning)
            if row.matched_terms:
                st.caption(f"Matched terms: {', '.join(row.matched_terms)}")
            st.markdown("**Claim text**")
            st.write(row.claim_text or "N/A")


def render_rag_analysis(generated_analysis) -> None:
    if not generated_analysis:
        st.info("No RAG analysis was generated.")
        return

    st.caption(f"Task type: {generated_analysis.task_type} | Generation mode: {generated_analysis.generation_mode}")
    if generated_analysis.warnings:
        with st.expander("Generation warnings"):
            for warning in generated_analysis.warnings:
                st.write(f"- {warning}")

    st.markdown("**Executive summary**")
    st.write(generated_analysis.executive_summary or "N/A")
    st.markdown("**Evidence-based findings**")
    for item in generated_analysis.evidence_based_findings:
        st.write(f"- {item}")
    st.markdown("**Risk summary**")
    st.write(generated_analysis.risk_summary or "N/A")
    st.markdown("**Recommended next steps**")
    for item in generated_analysis.recommended_next_steps:
        st.write(f"- {item}")
    st.markdown("**Citations**")
    for citation in generated_analysis.citations:
        st.write(f"- `{citation.chunk_id}` ({citation.patent_id}, {citation.section}): {citation.quote}")


def render_report_download(report_markdown: str, file_name: str) -> None:
    st.download_button(
        label="Download Markdown report",
        data=report_markdown,
        file_name=file_name,
        mime="text/markdown",
    )
    st.markdown(report_markdown)


def render_design_fto() -> None:
    with st.sidebar:
        st.header("Design Settings")
        design_top_k = st.slider("Design image candidates", min_value=2, max_value=6, value=4)
        st.info("Design mode uses local sample design images and lightweight perceptual similarity. CLIP/SigLIP can replace this layer later.")

    st.subheader("Design FTO")
    st.caption("Upload a product image or use the included chair demo image to retrieve visually similar design-patent samples.")

    product_type = st.text_input("Product type", value="chair", help="Optional category hint, for example chair, bottle, lamp.")
    uploaded = st.file_uploader("Product image", type=["png", "jpg", "jpeg", "webp"])
    use_sample = st.checkbox("Use demo chair image", value=uploaded is None)

    query_bytes = None
    query_name = ""
    if uploaded is not None:
        query_bytes = uploaded.getvalue()
        query_name = uploaded.name
    elif use_sample:
        query_bytes = SAMPLE_DESIGN_QUERY.read_bytes()
        query_name = SAMPLE_DESIGN_QUERY.name

    if query_bytes:
        st.image(query_bytes, caption=f"Query image: {query_name}", width=280)

    if st.button("Run design FTO search", type="primary"):
        if not query_bytes:
            st.warning("Please upload a product image or use the demo image.")
            return
        with st.spinner("Running visual design FTO search..."):
            result = load_design_fto_agent().run(
                query_image_bytes=query_bytes,
                query_image_name=query_name,
                product_type=product_type,
                top_k=design_top_k,
            )
        st.session_state["design_result"] = result

    result = st.session_state.get("design_result")
    if not result:
        return

    st.success(f"Design FTO completed: {result.task_id}")
    tab_results, tab_report = st.tabs(["Visual Results", "Report"])

    with tab_results:
        rows = [
            {
                "design_id": item.image.design_id,
                "title": item.image.title,
                "product_type": item.image.product_type,
                "view": item.image.view,
                "similarity": item.similarity_score,
                "risk": item.risk_level.value,
                "reasoning": item.reasoning,
            }
            for item in result.search_results
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        for index, item in enumerate(result.search_results, start=1):
            image = item.image
            with st.expander(f"{index}. {image.design_id} | {item.risk_level.value} | {image.title}", expanded=index <= 2):
                cols = st.columns([1, 2])
                with cols[0]:
                    st.image(image.image_path, caption=f"{image.design_id} - {image.view}", use_container_width=True)
                with cols[1]:
                    st.markdown(f"**Similarity score:** {item.similarity_score}")
                    st.markdown(f"**Product type:** {image.product_type}")
                    st.markdown(f"**Assignee:** {image.assignee or 'N/A'}")
                    st.markdown(f"**Publication date:** {image.publication_date or 'N/A'}")
                    st.markdown("**Visual overlaps**")
                    for overlap in item.visual_overlaps:
                        st.write(f"- {overlap}")
                    st.markdown("**Visual differences**")
                    for difference in item.visual_differences:
                        st.write(f"- {difference}")
                    st.markdown("**Reasoning**")
                    st.write(item.reasoning)

    with tab_report:
        render_report_download(result.report_markdown or "", f"{result.task_id}_design_fto_report.md")


if __name__ == "__main__":
    main()
