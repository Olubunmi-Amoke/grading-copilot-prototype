from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from score_engine import GradingRecord, SubmissionArtifacts


@dataclass
class BrightspaceStudentMetadata:
    student_name: Optional[str] = None
    student_id: Optional[str] = None
    submission_id: Optional[str] = None
    submitted_at_iso: Optional[str] = None
    discussion_posted_at_iso: Optional[str] = None
    turnitin_slide_pct: Optional[float] = None
    turnitin_script_pct: Optional[float] = None
    video_url: Optional[str] = None
    resubmission: Optional[bool] = False


def safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "yes", "1", "y"}


def load_metadata_csv(csv_path: str) -> Dict[str, BrightspaceStudentMetadata]:
    """
    Load Brightspace-style metadata from CSV.

    Expected columns can include:
    - student_name
    - student_id
    - submission_id
    - submitted_at_iso
    - discussion_posted_at_iso
    - turnitin_slide_pct
    - turnitin_script_pct
    - video_url
    - resubmission

    Returns a dict keyed by student folder name or student_id if available.
    """
    out: Dict[str, BrightspaceStudentMetadata] = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta = BrightspaceStudentMetadata(
                student_name=row.get("student_name"),
                student_id=row.get("student_id"),
                submission_id=row.get("submission_id"),
                submitted_at_iso=row.get("submitted_at_iso"),
                discussion_posted_at_iso=row.get("discussion_posted_at_iso"),
                turnitin_slide_pct=safe_float(row.get("turnitin_slide_pct")),
                turnitin_script_pct=safe_float(row.get("turnitin_script_pct")),
                video_url=row.get("video_url"),
                resubmission=safe_bool(row.get("resubmission")),
            )

            key = (
                row.get("folder_name")
                or row.get("student_id")
                or row.get("student_name")
            )
            if key:
                out[key] = meta

    return out


def detect_submission_files(student_dir: str) -> Dict[str, Optional[str]]:
    """
    Best-effort detection of key files inside one student's submission folder.
    """
    p = Path(student_dir)

    slideshow_pdf_path = None
    script_pdf_path = None
    video_link_path = None
    video_file_path = None

    for file in p.iterdir():
        if not file.is_file():
            continue

        name = file.name.lower()

        if file.suffix.lower() == ".pdf":
            if "slide" in name or "slideshow" in name or "presentation" in name:
                slideshow_pdf_path = str(file)
            elif "script" in name or "outline" in name or "notes" in name:
                script_pdf_path = str(file)
            else:
                # fallback: first unmatched pdf becomes slideshow, second becomes script
                if slideshow_pdf_path is None:
                    slideshow_pdf_path = str(file)
                elif script_pdf_path is None:
                    script_pdf_path = str(file)

        elif file.suffix.lower() == ".txt" and "video" in name:
            video_link_path = str(file)

        elif file.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi"}:
            video_file_path = str(file)

    return {
        "slideshow_pdf_path": slideshow_pdf_path,
        "script_pdf_path": script_pdf_path,
        "video_link_path": video_link_path,
        "video_file_path": video_file_path,
    }


def read_video_url_from_txt(video_link_path: Optional[str]) -> Optional[str]:
    if not video_link_path:
        return None
    try:
        with open(video_link_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text or None
    except Exception:
        return None


def infer_video_access(video_url: Optional[str], video_file_path: Optional[str]) -> Optional[str]:
    """
    Prototype heuristic:
    - if URL exists -> assume embedded_ok
    - elif video file exists -> requires_download
    - else -> inaccessible
    """
    if video_url:
        return "embedded_ok"
    if video_file_path:
        return "requires_download"
    return "inaccessible"


def build_record_from_submission_folder(
    student_dir: str,
    metadata: Optional[BrightspaceStudentMetadata] = None,
) -> GradingRecord:
    """
    Build a partially populated GradingRecord from a downloaded submission folder.
    Human graders will still fill in rubric judgments and qualitative checks later.
    """
    files = detect_submission_files(student_dir)
    video_url = read_video_url_from_txt(files["video_link_path"])

    artifacts = SubmissionArtifacts(
        slideshow_pdf_path=files["slideshow_pdf_path"],
        script_pdf_path=files["script_pdf_path"],
        video_url=video_url or (metadata.video_url if metadata else None),
        video_file_path=files["video_file_path"],
        submitted_at_iso=metadata.submitted_at_iso if metadata else None,
        discussion_posted_at_iso=metadata.discussion_posted_at_iso if metadata else None,
        student_name=metadata.student_name if metadata else None,
        student_id=metadata.student_id if metadata else None,
        submission_id=metadata.submission_id if metadata else None,
    )

    slideshow_pdf_present = files["slideshow_pdf_path"] is not None
    script_pdf_present = files["script_pdf_path"] is not None
    video_present = (video_url is not None) or (files["video_file_path"] is not None)

    record = GradingRecord(
        # Human will fill these later
        intro_score=0.0,
        depth_score=0.0,
        sources_score=0.0,
        presentation_score=0.0,

        # Auto-filled fields
        slideshow_pdf_present=slideshow_pdf_present,
        script_pdf_present=script_pdf_present,
        video_present=video_present,
        video_access=infer_video_access(video_url or (metadata.video_url if metadata else None), files["video_file_path"]),
        turnitin_slide_pct=metadata.turnitin_slide_pct if metadata else None,
        turnitin_script_pct=metadata.turnitin_script_pct if metadata else None,
        resubmission=metadata.resubmission if metadata else False,
        artifacts=artifacts,
    )

    return record


def load_submission_batch(
    submissions_root: str,
    metadata_csv_path: Optional[str] = None,
) -> List[GradingRecord]:
    """
    Load multiple student submission folders into a list of GradingRecords.
    Each immediate child folder under submissions_root is treated as one student submission.
    """
    metadata_map = load_metadata_csv(metadata_csv_path) if metadata_csv_path else {}

    records: List[GradingRecord] = []
    root = Path(submissions_root)

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        key_candidates = [child.name]
        meta = None

        for key in key_candidates:
            if key in metadata_map:
                meta = metadata_map[key]
                break

        record = build_record_from_submission_folder(str(child), metadata=meta)
        records.append(record)

    return records