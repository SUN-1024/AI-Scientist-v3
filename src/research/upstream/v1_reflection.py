"""V1's bounded idea-reflection loop adapted to a source-grounded agenda.

Derived from SakanaAI/AI-Scientist, commit
1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb, ai_scientist/generate_ideas.py,
idea_reflection_prompt, generate_ideas' iterative improvement loop, and its
idea_str_archive serialization into each subsequent first-idea prompt.
Copyright Sakana AI. AI Scientist Source Code License v1.0, December 2025.
Full license: third_party/ai_scientist_v1/LICENSE.

Modifications: request a brief critique, not hidden reasoning; use the existing
agenda schema and provider receipts; repeat the complete frozen source/code
context in each fresh call; require parsed JSON; record each proposal; accept
convergence only for an explicit stop plus an unchanged normalized agenda.
Reflection remains an author's proposal, never independent novelty evidence.
The archive renderer below carries completed host-assigned idea cards into new
portfolio prompts. It validates each card, keeps negative outcomes, and bounds
the complete archive rather than silently dropping earlier ideas.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable


def render_idea_archive(records: object) -> str:
    """Adapt V1's serialized previous-ideas archive to host portfolio cards.

    V1 appends each parsed idea JSON and joins the archive with blank lines for
    the next generation prompt. Here host-owned identities are separate from
    bounded model cards, and statuses remain observations rather than verdicts.
    """
    from src.research.whole_idea_policy import normalize_whole_idea

    if not isinstance(records, list) or len(records) > 128:
        raise ValueError("V1 idea archive must contain at most 128 records")
    archive: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("V1 idea archive entries must be objects")
        root_id = record.get("root_idea_id")
        if not isinstance(root_id, str) or not root_id or len(root_id) > 256 or root_id in seen:
            raise ValueError("V1 idea archive needs unique bounded host root IDs")
        seen.add(root_id)
        card = record.get("source_card")
        # A provider failure may have no parsed idea. Retain that record as an
        # observed failure; never manufacture a proposal or drop its outcome.
        normalized = {
            "root_idea_id": root_id,
            "source_card": normalize_whole_idea(card) if card is not None else None,
        }
        for key in ("innovation_axis", "status", "best_score", "outcome"):
            if key in record:
                normalized[key] = record[key]
        try:
            archive.append(json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("V1 idea archive must contain finite JSON") from exc
    joined = "\n\n".join(archive)
    if len(joined.encode("utf-8")) > 512 * 1024:
        raise ValueError("V1 idea archive exceeds 512 KiB; no ideas were silently removed")
    return ("\n\nV1 PRIOR-IDEA ARCHIVE (untrusted declarations and recorded outcomes):\n"
            "Use differences in questions, contributions, protocols and evidence to avoid duplicates. "
            "A failed or negative trial remains in the archive. Author confidence does not establish novelty.\n"
            + (joined or "No completed prior idea cards are available yet."))


IDEA_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
Carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Return the complete research agenda JSON using the original schema. Add one
optional _reflection object with done (boolean), changes (brief text), and
remaining_gaps (a list of brief strings). Do not return hidden reasoning.
If there is nothing to improve, repeat the previous agenda EXACTLY and set
_reflection.done to true. Set done to true ONLY IF YOU ARE MAKING NO MORE CHANGES.

The frozen sources and selected implementation remain authoritative. Do not
invent an experiment, result, dataset, citation, or new implementation to close a
gap. Keep missing evidence explicit. Preserve and develop the user's idea where
the source evidence permits a meaningful incremental contribution. A core
collision needs an explicit limit or narrower claim, not a cosmetic renaming.
This self-reflection does not establish novelty or replace independent review.
"""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass
class ReflectionResult:
    agenda: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    converged: bool = False
    num_reflections: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "upstream": "AI-Scientist-v1",
                "source_commit": "1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb",
                "status": "converged" if self.converged else "reflection_budget_exhausted",
                "author_proposal_only": True, "independent_novelty_evidence": False,
                "num_reflections": self.num_reflections, "history": self.history,
                "final_agenda_sha256": _digest(self.agenda)}


def refine_agenda(initial_agenda: dict[str, Any], *, frozen_context: str,
                  request: Callable[[str, int], dict[str, Any]],
                  normalize: Callable[[object], dict[str, Any]],
                  num_reflections: int = 3) -> ReflectionResult:
    """Run V1's bounded refine/parse/archive loop through the host provider.

    ``num_reflections`` includes the already completed initial drafting call,
    matching V1. The callback must persist its normal model-response receipt.
    Provider/validation errors propagate; they cannot become positive evidence.
    """
    if type(num_reflections) is not int or not 1 <= num_reflections <= 5:
        raise ValueError("Use one to five total agenda reflection rounds")
    if not isinstance(frozen_context, str) or not frozen_context.strip():
        raise ValueError("Reflection requires the complete frozen source and candidate context")
    agenda = normalize(copy.deepcopy(initial_agenda))
    agenda.pop("_reflection", None)
    result = ReflectionResult(agenda=agenda, num_reflections=num_reflections)
    result.history.append({"round": 1, "agenda_sha256": _digest(agenda), "agenda": copy.deepcopy(agenda)})

    for round_index in range(2, num_reflections + 1):
        prompt = (frozen_context + "\n\nAUTHOR REFLECTION:\n"
                  + IDEA_REFLECTION_PROMPT.format(current_round=round_index,
                                                  num_reflections=num_reflections)
                  + "\n\nPREVIOUS AGENDA JSON:\n"
                  + json.dumps(result.agenda, ensure_ascii=False, sort_keys=True))
        if len(prompt.encode("utf-8")) > 1_800_000:
            raise ValueError("Reflection prompt exceeds 1.8 MB; do not truncate frozen evidence")
        raw = request(prompt, round_index)
        if not isinstance(raw, dict):
            raise ValueError("Reflection response must be a JSON object")
        proposed = copy.deepcopy(raw)
        feedback = proposed.pop("_reflection", {})
        if not isinstance(feedback, dict) or type(feedback.get("done", False)) is not bool:
            raise ValueError("Reflection.done must be a boolean")
        changes, gaps = feedback.get("changes", ""), feedback.get("remaining_gaps", [])
        if not isinstance(changes, str) or not isinstance(gaps, list) or not all(isinstance(gap, str) for gap in gaps):
            raise ValueError("Reflection critique must contain text and a list of text gaps")
        agenda = normalize(proposed)
        previous_hash, agenda_hash = _digest(result.agenda), _digest(agenda)
        changed = agenda_hash != previous_hash
        result.history.append({"round": round_index, "agenda_sha256": agenda_hash,
                               "previous_agenda_sha256": previous_hash, "changed": changed,
                               "stop_requested": feedback.get("done", False),
                               "changes": changes, "remaining_gaps": gaps,
                               "agenda": copy.deepcopy(agenda)})
        result.agenda = agenda
        # The V1 text marker can occur in source text or a quoted idea. A parsed
        # boolean bound to unchanged JSON makes the stop condition unambiguous.
        if feedback.get("done") is True and not changed:
            result.converged = True
            break
    return result
