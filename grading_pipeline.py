from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from score_engine import (
    GradingRecord,
    ScoreBreakdown,
    TurnitinExplainability,
    score_record,
)

from turnitin_explain import (
    extract_text_from_pdf,
    analyze_presentation_turnitin,
    turnitin_should_run_explain,
    max_similarity,
)


def prepare_turnitin_findings(
    record: GradingRecord,
    cfg: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate explainable Turnitin findings for slides/script if similarity
    is high enough to warrant review and the needed PDF paths are available.

    Returns:
        (findings, note)
        findings: Dict with shape {"slides": {...}, "script": {...}} or None
        note: optional message explaining why findings were not produced
    """
    mx = max_similarity(record.turnitin_slide_pct, record.turnitin_script_pct)
    threshold = cfg.get("thresholds", {}).get("turnitin_review_pct", 30)

    if not turnitin_should_run_explain(mx, threshold=threshold):
        return None, "Turnitin explainability not run: below review threshold."

    if record.artifacts is None:
        return None, "Turnitin explainability not run: no submission artifacts attached."

    slides_pdf_path = record.artifacts.slideshow_pdf_path
    script_pdf_path = record.artifacts.script_pdf_path

    if not slides_pdf_path or not script_pdf_path:
        return None, "Turnitin explainability not run: missing slideshow or script PDF path."

    try:
        slides_text = extract_text_from_pdf(slides_pdf_path)
    except Exception as e:
        return None, f"Turnitin explainability not run: could not extract slideshow PDF text ({e})."

    try:
        script_text = extract_text_from_pdf(script_pdf_path)
    except Exception as e:
        return None, f"Turnitin explainability not run: could not extract script PDF text ({e})."

    try:
        findings = analyze_presentation_turnitin(
            slides_text=slides_text,
            script_text=script_text,
            slides_similarity_pct=record.turnitin_slide_pct,
            script_similarity_pct=record.turnitin_script_pct,
            cfg=cfg,
        )
        return findings, None
    except Exception as e:
        return None, f"Turnitin explainability not run: analysis failed ({e})."


def attach_turnitin_findings_to_record(
    record: GradingRecord,
    findings: Optional[Dict[str, Any]],
) -> GradingRecord:
    """
    Attach findings to the record in both:
    1) dict form (for UI / summaries / output)
    2) typed dataclass form (for consistency with score_engine fallback behavior)
    """
    if findings is None:
        return record

    record.turnitin_findings = findings

    slides = findings.get("slides")
    script = findings.get("script")

    if slides:
        record.turnitin_slide_explain = TurnitinExplainability(
            similarity_pct=slides.get("similarity_pct"),
            reasons=slides.get("reasons", []) or [],
            evidence=slides.get("evidence", []) or [],
            metrics=slides.get("metrics", {}) or {},
        )

    if script:
        record.turnitin_script_explain = TurnitinExplainability(
            similarity_pct=script.get("similarity_pct"),
            reasons=script.get("reasons", []) or [],
            evidence=script.get("evidence", []) or [],
            metrics=script.get("metrics", {}) or {},
        )

    return record


def attach_pipeline_note_to_record(
    record: GradingRecord,
    note: Optional[str],
) -> GradingRecord:
    """
    Optionally store pipeline notes in the Turnitin findings payload
    when explainability could not be run.
    """
    if not note:
        return record

    if record.turnitin_findings is None:
        record.turnitin_findings = {}

    record.turnitin_findings["_pipeline_note"] = note
    return record


def run_grading_pipeline(
    record: GradingRecord,
    cfg: Dict[str, Any],
) -> ScoreBreakdown:
    """
    Full orchestration pipeline for one student presentation.

    Steps:
    1. Decide whether Turnitin explainability should run.
    2. If yes, extract text from PDFs and build explainable findings.
    3. Attach findings (or note) to the record.
    4. Pass the record into the deterministic scoring engine.
    5. Return the final ScoreBreakdown.
    """
    findings, note = prepare_turnitin_findings(record, cfg)
    record = attach_turnitin_findings_to_record(record, findings)
    record = attach_pipeline_note_to_record(record, note)
    breakdown = score_record(record, cfg)
    return breakdown