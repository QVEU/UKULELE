"""
Field-level provenance tracking. The core anti-confabulation primitive.

Each field has:
  value       - the current value (or None)
  provenance  - one of: 'stated' | 'inferred' | 'unknown'
  source      - for 'stated': the user text it came from
                for 'inferred': the reasoning (must be shown at confirmation)
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict

STATED, INFERRED, UNKNOWN = "stated", "inferred", "unknown"

@dataclass
class Field:
    value: Any = None
    provenance: str = UNKNOWN
    source: Optional[str] = None

    def is_confident(self):
        return self.provenance == STATED

# The fields we track, grouped by how much scrutiny they need at confirmation.
# HIGH_VALUE gaps are the ones that change a null's trustworthiness — the agent
# asks about these specifically rather than interrogating everything equally.
HIGH_VALUE = [
    "evidence_layer",          # binding vs functional vs phenotypic — changes the meaning
    "positive_control",        # was assay sensitivity ever proven?
    "confidence_level",        # human should OWN this, not rubber-stamp
    "powered", "n",            # strength of the null
    "conditions_structured",   # drives comparability
]

# HUMAN_OWNED must never be auto-accepted from an inference — the agent may
# prompt, but the human has to actively set these (anchoring risk otherwise).
HUMAN_OWNED = ["confidence_level", "caveats"]

ALL_FIELDS = [
    "claim_type", "evidence_layer", "domain", "title",
    "observation", "conditions", "conditions_structured", "caveats",
    "alternatives", "system", "method",
    "confidence_level", "positive_control", "powered", "n",
    "detection_limit", "controls", "related_positive",
    "contributor", "date", "references",
]

class IntakeState:
    def __init__(self):
        self.fields: Dict[str, Field] = {k: Field() for k in ALL_FIELDS}

    def set(self, name, value, provenance, source=None):
        # HUMAN_OWNED fields cannot be set by inference — downgrade to a prompt.
        if name in HUMAN_OWNED and provenance == INFERRED:
            return False
        self.fields[name] = Field(value=value, provenance=provenance, source=source)
        return True

    def unknowns(self):
        return [k for k, f in self.fields.items() if f.provenance == UNKNOWN]

    def high_value_gaps(self):
        """Unknown OR merely-inferred high-value fields — worth a question."""
        gaps = []
        for k in HIGH_VALUE:
            f = self.fields.get(k, Field())
            if f.provenance in (UNKNOWN, INFERRED):
                gaps.append(k)
        return gaps

    def inferred(self):
        return {k: f for k, f in self.fields.items() if f.provenance == INFERRED}

    def as_dict(self):
        return {k: asdict(f) for k, f in self.fields.items()}
