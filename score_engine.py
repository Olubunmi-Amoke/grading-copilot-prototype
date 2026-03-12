from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
import yaml


# ----------------------------
# Data model (one presentation)
# ----------------------------
@dataclass
class SubmissionArtifacts:
    # local paths OR URLs depending on ingest approach
    slideshow_pdf_path: Optional[str] = None
    script_pdf_path: Optional[str] = None
    video_url: Optional[str] = None
    video_file_path: Optional[str] = None  # only if downloaded locally

    # metadata that might be extracted
    submitted_at_iso: Optional[str] = None
    discussion_posted_at_iso: Optional[str] = None
    student_name: Optional[str] = None
    student_id: Optional[str] = None
    submission_id: Optional[str] = None


@dataclass
class TurnitinExplainability:
    similarity_pct: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    evidence: List[Dict[str, str]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class GradingRecord:
    # Rubric category scores (0–100)
    intro_score: float
    depth_score: float
    sources_score: float
    presentation_score: float

    # Timing / delivery
    runtime_minutes: Optional[float] = None
    face_visible_majority: Optional[bool] = None

    # Submission/access
    video_access: Optional[str] = None  # "embedded_ok" | "requires_download" | "inaccessible"
    slideshow_pdf_present: Optional[bool] = True
    script_pdf_present: Optional[bool] = True
    video_present: Optional[bool] = True
    submission_instructions_issue: Optional[bool] = False
    resubmission: Optional[bool] = False

    # Sources/citations checks
    num_sources_on_reference_slide: Optional[int] = None
    reference_slide_in_video: Optional[bool] = None
    reference_slide_in_pdf: Optional[bool] = None
    citations_on_slides: Optional[str] = None  # "almost_every_slide" | "some_slides" | "few_or_none"
    reference_format_ok: Optional[bool] = True

    # Turnitin
    turnitin_slide_pct: Optional[float] = None
    turnitin_script_pct: Optional[float] = None
    suspected_plagiarism: Optional[bool] = False
    turnitin_findings: Optional[Dict[str, Any]] = None

    # Late/extra credit (presentation assignment only)
    late_days_assignment: int = 0
    early_submission_extra_credit_eligible: bool = False

    # Manual controls
    few_slide_citations_severity_points: Optional[int] = None  # choose 3–8 if citations_on_slides=="some_slides"
    manual_overall_deduction_points: float = 0.0              # used when runtime penalty >15, or other one-offs
    
    # Store explainability outputs
    artifacts: Optional[SubmissionArtifacts] = None
    turnitin_slide_explain: Optional[TurnitinExplainability] = None
    turnitin_script_explain: Optional[TurnitinExplainability] = None


@dataclass
class ScoreBreakdown:
    base_score: float
    category_scores_used: Dict[str, float]
    category_weights: Dict[str, float]

    runtime_penalty_points: float
    runtime_penalty_applied_to: str  # "presentation_category" | "overall_manual" | "none"

    point_deductions: List[Tuple[str, float]]
    score_after_point_deductions: float

    percent_modifiers: List[Tuple[str, float]]
    final_score: float
    
    #. Turnitin triage status
    turnitin_max_pct: Optional[float]
    turnitin_review_needed: bool
    turnitin_flag_recommended: bool

    flag_for_kiersten: bool
    flag_reasons: List[str]
    
    turnitin_findings: Optional[Dict[str, Any]] = None


# ----------------------------
# Engine
# ----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def round_points(x: float) -> float:
    # Match your examples: round to nearest integer point (13.3 -> 13)
    return float(int(round(x)))


def compute_base_score(record: GradingRecord, cfg: Dict[str, Any]) -> float:
    w = cfg["weights"]
    base = (
        w["intro"] * record.intro_score +
        w["depth"] * record.depth_score +
        w["sources"] * record.sources_score +
        w["presentation"] * record.presentation_score
    )
    return float(base)


def compute_runtime_penalty_points(runtime: Optional[float], cfg: Dict[str, Any]) -> float:
    if runtime is None:
        return 0.0

    tmin = cfg["runtime_rules"]["target_min_minutes"]
    tmax = cfg["runtime_rules"]["target_max_minutes"]

    if runtime < tmin:
        penalty = ((tmin - runtime) / tmin) * 100.0
        return round_points(penalty)

    if runtime > tmax:
        penalty = ((runtime - tmax) / tmax) * 100.0
        penalty = min(penalty, cfg["caps"]["runtime_over_20_cap_points"])
        return round_points(penalty)

    return 0.0


def score_record(record: GradingRecord, cfg: Dict[str, Any]) -> ScoreBreakdown:
    # ---- base score
    base = compute_base_score(record, cfg)

    point_deductions: List[Tuple[str, float]] = []
    flag_reasons: List[str] = []

    # ---- flags: submission completeness / access
    submission_incomplete = not (record.slideshow_pdf_present and record.script_pdf_present and record.video_present)
    if submission_incomplete and cfg["flags"]["flag_if_submission_incomplete"]:
        flag_reasons.append("Submission incomplete (missing required component(s))")

    if (record.slideshow_pdf_present is False or record.script_pdf_present is False) and cfg["flags"]["flag_if_missing_required_pdfs"]:
        flag_reasons.append("Missing required PDF(s): slideshow and/or script")

    if record.video_access == "inaccessible" and cfg["flags"]["flag_if_video_inaccessible"]:
        flag_reasons.append("Video inaccessible (cannot view)")

    if record.submission_instructions_issue and cfg["flags"]["flag_if_submission_instructions_not_followed"]:
        flag_reasons.append("Submission instructions not followed (e.g., video not embedded properly)")

    # ---- Turnitin triage (separate "review" vs "flag")
    turnitin_review_needed = False
    turnitin_flag_recommended = False

    turnitin_max = max(
        [x for x in [record.turnitin_slide_pct, record.turnitin_script_pct] if x is not None],
        default=None
    )

    if turnitin_max is not None:
        if turnitin_max >= cfg["thresholds"]["turnitin_flag_pct"] and cfg["flags"]["flag_if_turnitin_ge_flag_pct"]:
            turnitin_flag_recommended = True
            flag_reasons.append(
                f"Turnitin high (>= {cfg['thresholds']['turnitin_flag_pct']}%) — flag recommended"
            )
        elif turnitin_max >= cfg["thresholds"]["turnitin_review_pct"]:
            turnitin_review_needed = True
            flag_reasons.append(
                f"Turnitin elevated (>= {cfg['thresholds']['turnitin_review_pct']}%) — review recommended"
            )

    if record.suspected_plagiarism and cfg["flags"]["flag_if_suspected_plagiarism"]:
        flag_reasons.append("Suspected plagiarism after quick glance")
        
    flag_for_kiersten = (
        submission_incomplete
        or (record.video_access == "inaccessible")
        or (record.slideshow_pdf_present is False or record.script_pdf_present is False)
        or record.submission_instructions_issue
        or (record.suspected_plagiarism is True)
        or turnitin_flag_recommended
    )

    # ---- Non-categorical deductions
    # Face visible
    if record.face_visible_majority is False:
        point_deductions.append(("Face not visible for majority of presentation", float(cfg["deductions_points"]["face_not_visible"])))

    # Video requires download
    if record.video_access == "requires_download":
        point_deductions.append(("Video required download to view", float(cfg["deductions_points"]["video_requires_download"])))

    # Reference slide rules
    if record.reference_slide_in_video is False:
        if record.reference_slide_in_pdf:
            point_deductions.append(("Reference slide not shown in video (but present in PDF)", float(cfg["deductions_points"]["reference_slide_missing_in_video_but_in_pdf"])))
        else:
            point_deductions.append(("Reference slide missing from both video and PDF", float(cfg["deductions_points"]["reference_slide_missing_in_both_video_and_pdf"])))

    # Minimum sources rule (10)
    if record.num_sources_on_reference_slide is not None:
        req = cfg["thresholds"]["sources_required_min"]
        missing = max(0, req - record.num_sources_on_reference_slide)
        if missing > 0:
            per = cfg["deductions_points"]["sources_missing_per_source"]
            point_deductions.append((f"Missing sources (needed {req})", float(missing * per)))

    # Reference format
    if record.reference_format_ok is False:
        point_deductions.append(("Reference slide not in acceptable APA/MLA format", float(cfg["deductions_points"]["sources_bad_format"])))

    # Slide citation coverage
    if record.citations_on_slides == "few_or_none":
        point_deductions.append(("Few or no citations on slides throughout presentation", float(cfg["deductions_points"]["no_slide_citations_but_refs_shown"])))
    elif record.citations_on_slides == "some_slides":
        # choose severity (3–8); default to midpoint if not provided
        lo = cfg["deductions_points"]["few_slide_citations_min"]
        hi = cfg["deductions_points"]["few_slide_citations_max"]
        sev = record.few_slide_citations_severity_points
        if sev is None:
            sev = int(round((lo + hi) / 2))
        sev = max(lo, min(hi, sev))
        point_deductions.append(("Citations present on only some slides (should be on most)", float(sev)))

    # ---- Runtime penalty
    runtime_pen = compute_runtime_penalty_points(record.runtime_minutes, cfg)
    runtime_applied_to = "none"

    if runtime_pen > 0:
        if runtime_pen > cfg["caps"]["runtime_apply_to_category_max_points"]:
            # move to overall manual deduction bucket
            record.manual_overall_deduction_points += runtime_pen
            runtime_applied_to = "overall_manual"
        else:
            # apply to the Presentation category score by reducing it proportionately (points off final score)
            # We implement as point deduction equal to (runtime_pen * presentation_weight)
            pw = cfg["weights"]["presentation"]
            point_deductions.append(("Runtime outside 15–20 minutes (time penalty)", float(runtime_pen * pw)))
            runtime_applied_to = "presentation_category"

    # ---- Manual overall deductions (already in points)
    if record.manual_overall_deduction_points and record.manual_overall_deduction_points > 0:
        point_deductions.append(("Manual overall deduction (non-categorical)", float(record.manual_overall_deduction_points)))

    # Late penalty (assignment submission)
    if record.late_days_assignment and record.late_days_assignment > 0:
        per_day = cfg["deductions_points"]["late_per_day"]
        total_late_deduction = record.late_days_assignment * per_day
        point_deductions.append(
        (f"Late submission ({record.late_days_assignment} day(s))", float(total_late_deduction))
        )

    # ---- Apply point deductions
    score_after_points = base
    for _, pts in point_deductions:
        score_after_points -= pts

    # ---- Percent modifiers
    percent_mods: List[Tuple[str, float]] = []
    final = score_after_points

    # Resubmission
    if record.resubmission:
        mult = cfg["modifiers_percent"]["resubmission_multiplier"]
        percent_mods.append(("Resubmission deduction", mult))
        final *= mult

    # Early extra credit
    if record.early_submission_extra_credit_eligible:
        mult = cfg["modifiers_percent"]["early_extra_credit_multiplier"]
        percent_mods.append(("Early submission extra credit", mult))
        final *= mult

    # ---- Clamp
    final = clamp(final, cfg["caps"]["clamp_final_min"], cfg["caps"]["clamp_final_max"])
    
    turnitin_findings = None
    if record.turnitin_slide_explain or record.turnitin_script_explain:
        turnitin_findings = {
            "slides": asdict(record.turnitin_slide_explain) if record.turnitin_slide_explain else None,
            "script": asdict(record.turnitin_script_explain) if record.turnitin_script_explain else None,
        }

    return ScoreBreakdown(
        base_score=base,
        category_scores_used={
            "intro": record.intro_score,
            "depth": record.depth_score,
            "sources": record.sources_score,
            "presentation": record.presentation_score,
        },
        category_weights=cfg["weights"],
        runtime_penalty_points=runtime_pen,
        runtime_penalty_applied_to=runtime_applied_to,
        point_deductions=point_deductions,
        score_after_point_deductions=score_after_points,
        percent_modifiers=percent_mods,
        final_score=final,
        turnitin_max_pct=turnitin_max,
        turnitin_review_needed=turnitin_review_needed,
        turnitin_flag_recommended=turnitin_flag_recommended,
        turnitin_findings=record.turnitin_findings if record.turnitin_findings is not None else turnitin_findings,
        flag_for_kiersten=flag_for_kiersten,
        flag_reasons=flag_reasons,
    )