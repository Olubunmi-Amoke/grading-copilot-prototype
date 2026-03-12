from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

import streamlit as st

from score_engine import (
    load_config,
    GradingRecord,
    SubmissionArtifacts,
)
from grading_pipeline import run_grading_pipeline
from turnitin_explain import format_similarity_summary
from brightspace_ingest import (
    load_metadata_csv,
    build_record_from_submission_folder,
)


st.set_page_config(
    page_title="DTSC 690 Grading Co-Pilot",
    page_icon="🧭",
    layout="wide",
)

st.title("DTSC 690 Grading Co-Pilot")
st.caption("Triage assistant only; human grader remains fully responsible for academic judgment.")


@st.cache_data
def load_app_config(path: str):
    return load_config(path)


@st.cache_data
def cached_load_metadata_csv(csv_path: str):
    return load_metadata_csv(csv_path)


def initialize_session_state():
    if "record" not in st.session_state:
        st.session_state.record = None


def load_record_from_folder_ui() -> Optional[GradingRecord]:
    st.sidebar.header("Load from Submission Folder")

    student_dir = st.sidebar.text_input("Student submission folder path")
    metadata_csv_path = st.sidebar.text_input("Metadata CSV path (optional)")

    load_clicked = st.sidebar.button("Load submission folder")

    if not load_clicked:
        return None

    if not student_dir.strip():
        st.sidebar.error("Please provide a student submission folder path.")
        return None

    metadata = None
    if metadata_csv_path.strip():
        try:
            metadata_map = cached_load_metadata_csv(metadata_csv_path.strip())
            folder_name = student_dir.strip().rstrip("/").rstrip("\\").split("/")[-1].split("\\")[-1]
            metadata = metadata_map.get(folder_name)
        except Exception as e:
            st.sidebar.warning(f"Could not load metadata CSV: {e}")

    try:
        record = build_record_from_submission_folder(
            student_dir=student_dir.strip(),
            metadata=metadata,
        )
        st.sidebar.success("Submission folder loaded.")
        return record
    except Exception as e:
        st.sidebar.error(f"Failed to load submission folder: {e}")
        return None


def build_record_editor(existing_record: Optional[GradingRecord]) -> GradingRecord:
    record = existing_record or GradingRecord(
        intro_score=0.0,
        depth_score=0.0,
        sources_score=0.0,
        presentation_score=0.0,
        artifacts=SubmissionArtifacts(),
    )

    artifacts = record.artifacts or SubmissionArtifacts()

    with st.sidebar:
        st.header("Submission Metadata")
        student_name = st.text_input("Student name", value=artifacts.student_name or "")
        student_id = st.text_input("Student ID", value=artifacts.student_id or "")
        submission_id = st.text_input("Submission ID", value=artifacts.submission_id or "")
        submitted_at_iso = st.text_input("Submitted at (ISO)", value=artifacts.submitted_at_iso or "")
        discussion_posted_at_iso = st.text_input("Discussion posted at (ISO)", value=artifacts.discussion_posted_at_iso or "")

        st.subheader("Artifacts")
        slideshow_pdf_path = st.text_input("Slideshow PDF path", value=artifacts.slideshow_pdf_path or "")
        script_pdf_path = st.text_input("Script PDF path", value=artifacts.script_pdf_path or "")
        video_url = st.text_input("Video URL", value=artifacts.video_url or "")
        video_file_path = st.text_input("Video file path", value=artifacts.video_file_path or "")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rubric Scores")
        intro_score = st.number_input("Introduction score (0–100)", min_value=0.0, max_value=100.0, value=float(record.intro_score), step=1.0)
        depth_score = st.number_input("Depth score (0–100)", min_value=0.0, max_value=100.0, value=float(record.depth_score), step=1.0)
        sources_score = st.number_input("Sources score (0–100)", min_value=0.0, max_value=100.0, value=float(record.sources_score), step=1.0)
        presentation_score = st.number_input("Presentation score (0–100)", min_value=0.0, max_value=100.0, value=float(record.presentation_score), step=1.0)

        st.subheader("Presentation / Delivery")
        runtime_minutes = st.number_input("Runtime (minutes)", min_value=0.0, value=float(record.runtime_minutes or 0.0), step=0.1)
        face_visible_majority = st.checkbox("Face visible for majority of presentation", value=bool(record.face_visible_majority) if record.face_visible_majority is not None else True)

        video_access_options = ["embedded_ok", "requires_download", "inaccessible"]
        default_video_access = record.video_access if record.video_access in video_access_options else "embedded_ok"
        video_access = st.selectbox(
            "Video access",
            options=video_access_options,
            index=video_access_options.index(default_video_access),
        )

        submission_instructions_issue = st.checkbox("Submission instructions issue", value=bool(record.submission_instructions_issue))
        resubmission = st.checkbox("Resubmission", value=bool(record.resubmission))

    with col2:
        st.subheader("Submission Checks")
        slideshow_pdf_present = st.checkbox("Slideshow PDF present", value=bool(record.slideshow_pdf_present))
        script_pdf_present = st.checkbox("Script PDF present", value=bool(record.script_pdf_present))
        video_present = st.checkbox("Video present", value=bool(record.video_present))

        reference_slide_in_video = st.checkbox(
            "Reference slide shown in video",
            value=bool(record.reference_slide_in_video) if record.reference_slide_in_video is not None else True,
        )
        reference_slide_in_pdf = st.checkbox(
            "Reference slide present in PDF",
            value=bool(record.reference_slide_in_pdf) if record.reference_slide_in_pdf is not None else True,
        )
        reference_format_ok = st.checkbox(
            "Reference format acceptable",
            value=bool(record.reference_format_ok) if record.reference_format_ok is not None else True,
        )

        citation_options = ["almost_every_slide", "some_slides", "few_or_none"]
        default_citation_choice = record.citations_on_slides if record.citations_on_slides in citation_options else "almost_every_slide"
        citations_on_slides = st.selectbox(
            "Citation coverage on slides",
            options=citation_options,
            index=citation_options.index(default_citation_choice),
        )

        few_slide_citations_severity_points: Optional[int] = record.few_slide_citations_severity_points
        if citations_on_slides == "some_slides":
            few_slide_citations_severity_points = st.slider(
                "Citation sparsity severity (3–8)",
                min_value=3,
                max_value=8,
                value=int(record.few_slide_citations_severity_points or 5),
                step=1,
            )
        else:
            few_slide_citations_severity_points = None

        num_sources_on_reference_slide = st.number_input(
            "Number of sources on reference slide",
            min_value=0,
            max_value=50,
            value=int(record.num_sources_on_reference_slide or 10),
            step=1,
        )

        st.subheader("Turnitin / Adjustments")
        turnitin_slide_pct = st.number_input(
            "Turnitin slides %",
            min_value=0.0,
            max_value=100.0,
            value=float(record.turnitin_slide_pct or 0.0),
            step=1.0,
        )
        turnitin_script_pct = st.number_input(
            "Turnitin script %",
            min_value=0.0,
            max_value=100.0,
            value=float(record.turnitin_script_pct or 0.0),
            step=1.0,
        )
        suspected_plagiarism = st.checkbox(
            "Suspected plagiarism after human review",
            value=bool(record.suspected_plagiarism),
        )

        late_days_assignment = st.number_input(
            "Late days (assignment)",
            min_value=0,
            max_value=30,
            value=int(record.late_days_assignment),
            step=1,
        )
        early_submission_extra_credit_eligible = st.checkbox(
            "Eligible for early submission extra credit",
            value=bool(record.early_submission_extra_credit_eligible),
        )
        manual_overall_deduction_points = st.number_input(
            "Manual overall deduction points",
            min_value=0.0,
            max_value=100.0,
            value=float(record.manual_overall_deduction_points),
            step=1.0,
        )

    updated_artifacts = SubmissionArtifacts(
        slideshow_pdf_path=slideshow_pdf_path or None,
        script_pdf_path=script_pdf_path or None,
        video_url=video_url or None,
        video_file_path=video_file_path or None,
        submitted_at_iso=submitted_at_iso or None,
        discussion_posted_at_iso=discussion_posted_at_iso or None,
        student_name=student_name or None,
        student_id=student_id or None,
        submission_id=submission_id or None,
    )

    updated_record = GradingRecord(
        intro_score=intro_score,
        depth_score=depth_score,
        sources_score=sources_score,
        presentation_score=presentation_score,
        runtime_minutes=runtime_minutes,
        face_visible_majority=face_visible_majority,
        video_access=video_access,
        slideshow_pdf_present=slideshow_pdf_present,
        script_pdf_present=script_pdf_present,
        video_present=video_present,
        submission_instructions_issue=submission_instructions_issue,
        resubmission=resubmission,
        num_sources_on_reference_slide=int(num_sources_on_reference_slide),
        reference_slide_in_video=reference_slide_in_video,
        reference_slide_in_pdf=reference_slide_in_pdf,
        citations_on_slides=citations_on_slides,
        reference_format_ok=reference_format_ok,
        turnitin_slide_pct=turnitin_slide_pct,
        turnitin_script_pct=turnitin_script_pct,
        suspected_plagiarism=suspected_plagiarism,
        turnitin_findings=record.turnitin_findings,
        late_days_assignment=int(late_days_assignment),
        early_submission_extra_credit_eligible=early_submission_extra_credit_eligible,
        few_slide_citations_severity_points=few_slide_citations_severity_points,
        manual_overall_deduction_points=manual_overall_deduction_points,
        artifacts=updated_artifacts,
        turnitin_slide_explain=record.turnitin_slide_explain,
        turnitin_script_explain=record.turnitin_script_explain,
    )

    return updated_record


def render_breakdown(breakdown):
    st.subheader("Final Result")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base score", f"{breakdown.base_score:.2f}")
    c2.metric("After point deductions", f"{breakdown.score_after_point_deductions:.2f}")
    c3.metric("Final score", f"{breakdown.final_score:.2f}")
    c4.metric("Turnitin max %", "N/A" if breakdown.turnitin_max_pct is None else f"{breakdown.turnitin_max_pct:.0f}%")

    st.subheader("Turnitin Triage")
    left, right = st.columns(2)

    with left:
        st.write(f"**Review needed:** {'Yes' if breakdown.turnitin_review_needed else 'No'}")
        st.write(f"**Flag recommended:** {'Yes' if breakdown.turnitin_flag_recommended else 'No'}")

    with right:
        st.write(f"**Flag for Kiersten:** {'Yes' if breakdown.flag_for_kiersten else 'No'}")

    if breakdown.turnitin_findings:
        pipeline_note = breakdown.turnitin_findings.get("_pipeline_note")
        if pipeline_note:
            st.info(pipeline_note)

        st.markdown("**Turnitin summary**")
        st.code(format_similarity_summary(breakdown.turnitin_findings, include_evidence=False))

        with st.expander("Show Turnitin evidence"):
            st.text(format_similarity_summary(breakdown.turnitin_findings, include_evidence=True))

    st.subheader("Point Deductions")
    if breakdown.point_deductions:
        for reason, pts in breakdown.point_deductions:
            st.write(f"- {reason}: **-{pts:.2f}**")
    else:
        st.write("No point deductions applied.")

    st.subheader("Percent Modifiers")
    if breakdown.percent_modifiers:
        for reason, mult in breakdown.percent_modifiers:
            st.write(f"- {reason}: **× {mult:.4f}**")
    else:
        st.write("No percent modifiers applied.")

    st.subheader("Flags / Reasons")
    if breakdown.flag_reasons:
        for r in breakdown.flag_reasons:
            st.write(f"- {r}")
    else:
        st.write("No flags raised.")

    with st.expander("Raw ScoreBreakdown JSON"):
        st.json(asdict(breakdown))


def main():
    initialize_session_state()

    cfg_path = st.sidebar.text_input("Config path", value="rubric_config.yaml")
    cfg = load_app_config(cfg_path)

    loaded_record = load_record_from_folder_ui()
    if loaded_record is not None:
        st.session_state.record = loaded_record

    st.subheader("Record Editor")
    record = build_record_editor(st.session_state.record)

    save_record = st.button("Save current form state")
    if save_record:
        st.session_state.record = record
        st.success("Current form state saved.")

    st.divider()
    run_clicked = st.button("Run grading pipeline", type="primary")

    if run_clicked:
        try:
            st.session_state.record = record
            breakdown = run_grading_pipeline(record, cfg)
            render_breakdown(breakdown)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()