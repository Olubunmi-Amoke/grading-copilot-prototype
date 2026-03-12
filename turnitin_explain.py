from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any

# Optional PDF text extraction deps:
# - pip install pypdf
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# ----------------------------
# Explainability model
# ----------------------------
@dataclass
class TurnitinExplainability:
    similarity_pct: Optional[float] = None

    # Multi-label reasons triggered by deterministic heuristics
    reasons: List[str] = field(default_factory=list)

    # Human-reviewable evidence snippets (short, bounded)
    evidence: List[Dict[str, str]] = field(default_factory=list)

    # Extra metrics for transparency/audit
    metrics: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# Text extraction
# ----------------------------
def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a PDF. Best-effort.
    If PdfReader isn't installed, raises RuntimeError.
    """
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Install with: pip install pypdf")

    reader = PdfReader(pdf_path)
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]

    chunks: List[str] = []
    for p in pages:
        try:
            chunks.append(p.extract_text() or "")
        except Exception:
            chunks.append("")
    return "\n".join(chunks)


# ----------------------------
# Helpers
# ----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def first_n_words(text: str, n: int = 30) -> str:
    words = _WORD_RE.findall(text or "")
    return " ".join(words[:n])

def safe_snippet(text: str, start: int, end: int, max_len: int = 220) -> str:
    snippet = (text or "")[start:end]
    snippet = normalize_whitespace(snippet)
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rstrip() + "…"
    return snippet

def find_section_spans(text: str, headers: List[str]) -> List[Tuple[int, int, str]]:
    """
    Returns spans (start_idx, end_idx, header) for each detected header section.
    Very lightweight: finds header occurrences and uses next header as boundary.
    """
    low = text.lower()
    hits: List[Tuple[int, str]] = []
    for h in headers:
        idx = low.find(h.lower())
        if idx != -1:
            hits.append((idx, h))
    hits.sort(key=lambda x: x[0])

    spans: List[Tuple[int, int, str]] = []
    for i, (pos, h) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        spans.append((pos, end, h))
    return spans

def quote_spans(text: str) -> List[Tuple[int, int]]:
    """
    Identify quoted spans using common quote characters. Best effort.
    """
    spans: List[Tuple[int, int]] = []

    # “...” and "..."
    patterns = [
        r'“[^”]{10,}”',
        r'"[^"]{10,}"',
        r"‘[^’]{10,}’",
        r"'[^']{10,}'",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            spans.append((m.start(), m.end()))
    spans.sort()
    return spans

def span_total_len(spans: List[Tuple[int, int]]) -> int:
    return sum(max(0, b - a) for a, b in spans)

# def overlap_spans(spans: List[Tuple[int, int]], start: int, end: int) -> int:
#     total = 0
#     for a, b in spans:
#         lo = max(a, start)
#         hi = min(b, end)
#         if hi > lo:
#             total += hi - lo
#     return total


# ----------------------------
# Core analysis heuristics
# ----------------------------
def analyze_similarity_text(
    text: str,
    similarity_pct: Optional[float],
    *,
    doc_label: str = "document",
    quote_ratio_heavy: float = 0.15,
    overlap_block_min_words: int = 40,
    definition_density_threshold: float = 0.006,  # definitional triggers / word
    repeated_ngram_threshold: int = 4,
    bibliography_headers: Optional[List[str]] = None,
    max_evidence: int = 6,
) -> TurnitinExplainability:
    """
    Explainable analysis based purely on the student's text.
    Produces multi-label reasons + evidence snippets.

    IMPORTANT: This does NOT determine plagiarism. It diagnoses patterns consistent with
    common Turnitin similarity causes (heavy quotes, definitions, references, etc.).
    """
    bibliography_headers = bibliography_headers or ["references", "works cited", "bibliography"]

    raw = text or ""
    norm = normalize_whitespace(raw)
    wc = word_count(norm)

    out = TurnitinExplainability(similarity_pct=similarity_pct)
    out.metrics.update({
        "doc_label": doc_label,
        "word_count": wc,
        "similarity_pct": similarity_pct,
    })

    if wc < 120:
        # very short text; many heuristics become noisy
        out.reasons.append("text_too_short_for_reliable_diagnosis")
        out.metrics["note"] = "Document text is short; similarity drivers may be hard to infer."
        return out

    # 1) Quotes: quote ratio
    qsp = quote_spans(norm)
    quoted_chars = span_total_len(qsp)
    quote_ratio = quoted_chars / max(1, len(norm))
    out.metrics["quote_ratio_chars"] = round(quote_ratio, 4)

    if quote_ratio >= quote_ratio_heavy:
        out.reasons.append("heavy_quotes")
        # add evidence snippets from first few quote spans
        for (a, b) in qsp[:max_evidence]:
            out.evidence.append({
                "reason": "heavy_quotes",
                "snippet": safe_snippet(norm, a, b),
                "notes": "Large quoted span detected.",
            })

    # 2) Bibliography / references section size (not similarity, but likely to inflate matches)
    spans = find_section_spans(norm, bibliography_headers)
    bib_ratio = 0.0
    if spans:
        # take the earliest header as bibliography start
        start, end, header = spans[0]
        bib_len = end - start
        bib_ratio = bib_len / max(1, len(norm))
        out.metrics["bibliography_section_ratio_chars"] = round(bib_ratio, 4)
        if bib_ratio >= 0.18:
            out.reasons.append("large_references_section")
            out.evidence.append({
                "reason": "large_references_section",
                "snippet": safe_snippet(norm, start, min(end, start + 400)),
                "notes": f"Detected '{header}' section occupying a large portion of document text.",
            })
    else:
        out.metrics["bibliography_section_ratio_chars"] = 0.0

    # 3) Definition/boilerplate density (deterministic phrase triggers)
    definitional_patterns = [
        r"\bis defined as\b",
        r"\brefers to\b",
        r"\bcan be defined as\b",
        r"\bdefined by\b",
        r"\bin other words\b",
        r"\baccording to\b",
        r"\bthe term\b.+\brefers to\b",
    ]
    def_hits = 0
    for pat in definitional_patterns:
        def_hits += len(re.findall(pat, norm.lower()))
    def_density = def_hits / max(1, wc)
    out.metrics["definition_trigger_count"] = def_hits
    out.metrics["definition_trigger_density"] = round(def_density, 6)

    if def_density >= definition_density_threshold and def_hits >= 2:
        out.reasons.append("definition_heavy")
        out.evidence.append({
            "reason": "definition_heavy",
            "snippet": first_n_words(norm, 45),
            "notes": "High density of definitional/boilerplate phrasing (may elevate similarity).",
        })

    # 4) Repeated phrases within the doc (proxy for copied templates or repeated pasted chunks)
    # We look for repeated 8-grams (conservative) to reduce noise.
    tokens = _WORD_RE.findall(norm.lower())
    n = 8
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(0, len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        counts[ng] = counts.get(ng, 0) + 1
    repeated = [(ng, c) for ng, c in counts.items() if c >= repeated_ngram_threshold]
    repeated.sort(key=lambda x: x[1], reverse=True)

    out.metrics["repeated_8gram_count"] = len(repeated)
    if repeated:
        out.reasons.append("repeated_phrases_template_like")
        # evidence: show the top repeated phrase
        ng, c = repeated[0]
        phrase = " ".join(ng)
        out.evidence.append({
            "reason": "repeated_phrases_template_like",
            "snippet": phrase,
            "notes": f"Repeated phrase detected {c} times within the document (possible template/paste repetition).",
        })

    # 5) Unquoted long blocks heuristic (proxy for copy-paste)
    # Without Turnitin highlights we cannot know if it matches sources,
    # but we can flag *risk*: long spans without quotes + high similarity %.
    # We approximate "blockiness" by finding long sentences and low quote overlap.
    sentences = re.split(r"(?<=[.!?])\s+", norm)
    long_sentences = [s for s in sentences if word_count(s) >= overlap_block_min_words]
    out.metrics["long_sentence_count_ge_threshold"] = len(long_sentences)

    if similarity_pct is not None and similarity_pct >= 30 and len(long_sentences) >= 2:
        out.reasons.append("long_unquoted_blocks_risk")
        # evidence: show up to 2 long sentences
        for s in long_sentences[:2]:
            out.evidence.append({
                "reason": "long_unquoted_blocks_risk",
                "snippet": normalize_whitespace(s)[:220] + ("…" if len(s) > 220 else ""),
                "notes": f"Long sentence/block (>= {overlap_block_min_words} words). If not original, this can drive similarity.",
            })

    # 6) Contextual summary flags based on similarity %
    if similarity_pct is not None:
        if similarity_pct >= 70:
            out.reasons.append("very_high_similarity_requires_manual_review")
        elif similarity_pct >= 30:
            out.reasons.append("elevated_similarity_requires_quick_review")

    # Keep evidence bounded
    out.evidence = out.evidence[:max_evidence]

    return out


def analyze_presentation_turnitin(
    *,
    slides_text: Optional[str],
    script_text: Optional[str],
    slides_similarity_pct: Optional[float],
    script_similarity_pct: Optional[float],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns a machine-friendly dict suitable for ScoreBreakdown.turnitin_findings:
      {"slides": {...}, "script": {...}}
    """
    cfg = cfg or {}
    te = cfg.get("turnitin_explainability", {})

    kwargs = dict(
        quote_ratio_heavy=te.get("quote_ratio_heavy", 0.15),
        overlap_block_min_words=te.get("overlap_block_min_words", 40),
        definition_density_threshold=te.get("definition_density_threshold", 0.006),
        repeated_ngram_threshold=te.get("repeated_ngram_threshold", 4),
        bibliography_headers=te.get("bibliography_section_headers", ["references", "works cited", "bibliography"]),
        max_evidence=te.get("max_evidence", 6),
    )

    slides = analyze_similarity_text(
        slides_text or "",
        slides_similarity_pct,
        doc_label="slides",
        **kwargs,
    )
    script = analyze_similarity_text(
        script_text or "",
        script_similarity_pct,
        doc_label="script",
        **kwargs,
    )

    return {
        "slides": asdict(slides),
        "script": asdict(script),
    }
    
def turnitin_should_run_explain(
    similarity_pct: Optional[float],
    *,
    threshold: float = 30.0
) -> bool:
    """
    Decide whether to run explainability analysis.

    We only run when similarity is elevated (>= threshold) because extracting
    PDF text can be slow and noisy for low-similarity cases.

    Returns False if similarity_pct is None.
    """
    if similarity_pct is None:
        return False
    try:
        return float(similarity_pct) >= float(threshold)
    except (TypeError, ValueError):
        return False
    
def max_similarity(slides_pct: Optional[float], script_pct: Optional[float]) -> Optional[float]:
    vals = [v for v in [slides_pct, script_pct] if v is not None]
    return max(vals) if vals else None
    

def _pick_top_reasons(reasons: List[str], max_n: int = 4) -> List[str]:
    """
    Prefer more serious reasons first; keep deterministic ordering.
    """
    priority = {
        "very_high_similarity_requires_manual_review": 0,
        "long_unquoted_blocks_risk": 1,
        "heavy_quotes": 2,
        "definition_heavy": 3,
        "repeated_phrases_template_like": 4,
        "large_references_section": 5,
        "elevated_similarity_requires_quick_review": 6,
        "text_too_short_for_reliable_diagnosis": 7,
    }
    dedup = []
    seen = set()
    for r in reasons or []:
        if r not in seen:
            dedup.append(r)
            seen.add(r)
    dedup.sort(key=lambda r: priority.get(r, 99))
    return dedup[:max_n]


def _fmt_reason(reason: str) -> str:
    mapping = {
        "very_high_similarity_requires_manual_review": "very high similarity (manual review needed)",
        "elevated_similarity_requires_quick_review": "elevated similarity (quick review needed)",
        "long_unquoted_blocks_risk": "long unquoted blocks risk",
        "heavy_quotes": "heavy quoting",
        "definition_heavy": "definition/boilerplate-heavy phrasing",
        "repeated_phrases_template_like": "repeated template-like phrasing",
        "large_references_section": "large references section (may inflate similarity)",
        "text_too_short_for_reliable_diagnosis": "text too short for reliable diagnosis",
    }
    return mapping.get(reason, reason.replace("_", " "))


# def _safe_get(d: Dict[str, Any], path: Tuple[str, ...], default=None):
#     cur: Any = d
#     for p in path:
#         if not isinstance(cur, dict) or p not in cur:
#             return default
#         cur = cur[p]
#     return cur


def format_similarity_summary(
    turnitin_findings: Optional[Dict[str, Any]],
    *,
    include_evidence: bool = True,
    max_evidence_per_doc: int = 2,
) -> str:
    """
    Deterministic, explainable summary of Turnitin findings.

    Expected input shape:
      {"slides": {"similarity_pct":..., "reasons":[...], "metrics":{...}, "evidence":[...]},
       "script": {...}}

    Returns a human-readable summary suitable for:
      - UI display (include_evidence=False)
      - escalation notes/email (include_evidence=True)
    """
    if not turnitin_findings:
        return "Turnitin findings: none available."

    parts: List[str] = []
    evidence_lines: List[str] = []

    for doc_label in ["slides", "script"]:
        doc = turnitin_findings.get(doc_label) or {}
        sim = doc.get("similarity_pct", None)
        reasons = doc.get("reasons", []) or []
        metrics = doc.get("metrics", {}) or {}
        evidence = doc.get("evidence", []) or []

        top_reasons = _pick_top_reasons(reasons, max_n=4)
        reasons_txt = ", ".join(_fmt_reason(r) for r in top_reasons) if top_reasons else "no strong pattern detected"

        quote_ratio = metrics.get("quote_ratio_chars", None)
        bib_ratio = metrics.get("bibliography_section_ratio_chars", None)
        long_blocks = metrics.get("long_sentence_count_ge_threshold", None)
        wc = metrics.get("word_count", None)

        metric_bits: List[str] = []
        if wc is not None:
            metric_bits.append(f"words={wc}")
        if quote_ratio is not None:
            metric_bits.append(f"quote_ratio={quote_ratio:.2f}")
        if bib_ratio is not None:
            metric_bits.append(f"refs_section_ratio={bib_ratio:.2f}")
        if long_blocks is not None:
            metric_bits.append(f"long_blocks={long_blocks}")

        metric_txt = "; ".join(metric_bits) if metric_bits else "metrics unavailable"
        sim_txt = f"{sim:.0f}%" if isinstance(sim, (int, float)) else "N/A"

        parts.append(f"{doc_label.capitalize()}: Turnitin={sim_txt}. Drivers: {reasons_txt}. ({metric_txt})")

        if include_evidence and evidence:
            used = 0
            for ev in evidence:
                if used >= max_evidence_per_doc:
                    break
                snippet = (ev.get("snippet") or "").strip()
                if not snippet:
                    continue
                ev_reason = _fmt_reason(ev.get("reason", "evidence"))
                note = (ev.get("notes") or "").strip()
                line = f"- {doc_label}: {ev_reason}: “{snippet}”"
                if note:
                    line += f" — {note}"
                evidence_lines.append(line)
                used += 1

    summary = " | ".join(parts)
    if include_evidence and evidence_lines:
        summary += "\nEvidence (for human review):\n" + "\n".join(evidence_lines)

    return summary