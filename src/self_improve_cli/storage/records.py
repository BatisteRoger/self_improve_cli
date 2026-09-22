"""Analyst records — assessment and findings persistence.

Records are analyst-authored metadata stored alongside the trace
(`assessment.json`, `findings.json`). They are subjective judgments, not
computed artifacts — see domain/records.py for the types.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from self_improve_cli.domain import Assessment, Finding, OutcomeSource, OutcomeStatus
from self_improve_cli.storage.base import StoreBase, _dataclass_to_dict, write_json

logger = logging.getLogger(__name__)


class RecordsMixin(StoreBase):
    """Assessment and findings persistence alongside the trace."""

    # -- Assessment ------------------------------------------------------

    def save_assessment(self, assessment: Assessment) -> Path:
        """Persist a manual assessment as JSON alongside the trace."""
        trace_dir = self._trace_dir(assessment.trace_id)
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / "assessment.json"
        payload = _dataclass_to_dict(assessment)
        write_json(path, payload)
        logger.info("Saved assessment for trace %s to %s", assessment.trace_id, path)
        return path

    def load_assessment(self, trace_id: str) -> Assessment | None:
        """Load a saved assessment, or None if it doesn't exist."""
        path = self._trace_dir(trace_id) / "assessment.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Assessment(
            trace_id=data.get("trace_id", trace_id),
            task=data.get("task", ""),
            outcome=OutcomeStatus(data.get("outcome", "unknown")),
            outcome_source=OutcomeSource(data.get("outcome_source", "unknown")),
            notes=data.get("notes", ""),
            assessed_at=data.get("assessed_at", ""),
        )

    def clear_assessment(self, trace_id: str) -> bool:
        """Remove a saved assessment. Returns True if it existed."""
        path = self._trace_dir(trace_id) / "assessment.json"
        if path.exists():
            path.unlink()
            logger.info("Cleared assessment for trace %s", trace_id)
            return True
        return False

    # -- Findings --------------------------------------------------------

    def _findings_path(self, trace_id: str) -> Path:
        return self._trace_dir(trace_id) / "findings.json"

    def load_findings(self, trace_id: str) -> list[Finding]:
        """Load persisted findings for a trace, or [] if none exist."""
        path = self._findings_path(trace_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            Finding(
                id=f.get("id", ""),
                trace_id=f.get("trace_id", trace_id),
                title=f.get("title", ""),
                pattern=f.get("pattern", ""),
                secondary_patterns=f.get("secondary_patterns", []),
                impact=f.get("impact", "no_impact"),
                evidence_strength=f.get("evidence_strength", "observed"),
                triangle_axis=f.get("triangle_axis", "quality"),
                evidence=f.get("evidence", []),
                assessment=f.get("assessment", ""),
                fault_locus=f.get("fault_locus", ""),
                suggested_next_action=f.get("suggested_next_action", ""),
                candidate_improvement=f.get("candidate_improvement", ""),
                validation=f.get("validation", ""),
                created_at=f.get("created_at", ""),
            )
            for f in data.get("findings", [])
        ]

    def save_findings(self, trace_id: str, findings: list[Finding]) -> Path:
        """Persist the full findings list for a trace."""
        trace_dir = self._trace_dir(trace_id)
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = self._findings_path(trace_id)
        payload = {
            "trace_id": trace_id,
            "findings": [_dataclass_to_dict(f) for f in findings],
        }
        write_json(path, payload)
        logger.info("Saved %d findings for trace %s to %s", len(findings), trace_id, path)
        return path

    def next_finding_id(self, trace_id: str) -> str:
        """Return the next sequential finding id (f1, f2, ...) for a trace."""
        existing = self.load_findings(trace_id)
        max_n = 0
        for f in existing:
            if f.id.startswith("f") and f.id[1:].isdigit():
                max_n = max(max_n, int(f.id[1:]))
        return f"f{max_n + 1}"

    def add_finding(self, finding: Finding) -> Path:
        """Append a finding to the trace's findings list."""
        findings = self.load_findings(finding.trace_id)
        findings.append(finding)
        return self.save_findings(finding.trace_id, findings)

    def remove_finding(self, trace_id: str, finding_id: str) -> bool:
        """Remove one finding by id. Returns True if it existed."""
        findings = self.load_findings(trace_id)
        kept = [f for f in findings if f.id != finding_id]
        if len(kept) == len(findings):
            return False
        self.save_findings(trace_id, kept)
        logger.info("Removed finding %s for trace %s", finding_id, trace_id)
        return True

    def clear_findings(self, trace_id: str) -> bool:
        """Remove all findings for a trace. Returns True if the file existed."""
        path = self._findings_path(trace_id)
        if path.exists():
            path.unlink()
            logger.info("Cleared findings for trace %s", trace_id)
            return True
        return False
