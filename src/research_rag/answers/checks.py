"""Check the argument chain, its links and the per-source checks.

Copied from github.com/AhmedKishki/agent-skills @ 2a3a28f,
`skills/white-box-synthesis/scripts/checks.py`. The copy is deliberate; carry a fix
to both copies.
"""

import re

from .ops import Rejected
from .pieces import copy_span

STATUSES = {"part", "stated", "gap", "ai"}


def check_step(step, index, sources, allowed):
    sid, to, status = step["id"], step.get("supports"), step.get("status")
    if sid == "Z":
        if to or status:
            raise Rejected("Step Z is the stated conclusion. It supports nothing.")
        return
    if not step.get("why"):
        raise Rejected(f"Step {sid}: say why it is necessary for Z.")
    if to != "Z" and to not in index:
        raise Rejected(f"Step {sid}: supports unknown step {to!r}.")
    if status not in STATUSES:
        raise Rejected(f"Step {sid}: status must be one of {sorted(STATUSES)}.")
    if status == "part" and to != "Z":
        raise Rejected(f"Step {sid}: only a direct part of the answer Z can be 'part'.")
    if status == "stated":
        lic = step.get("licence") or {}
        copy_span(
            sources, lic.get("source"), lic.get("text", ""), lic.get("occurrence", 1)
        )
        if to == "Z" and "Z" not in index:
            raise Rejected(
                f"Step {sid}: a licence for the link to Z states Z. Add Z as a step."
            )
    if status in ("gap", "ai") and "→" not in step.get("missing", ""):
        raise Rejected(f"Step {sid}: describe the missing link as 'X → Y: …'.")
    if status == "ai":
        if allowed != "all" and sid not in allowed:
            raise Rejected(
                f"Step {sid}: an AI-authored link is not allowed here. Keep the gap."
            )
        if to == "Z" and "Z" not in index:
            raise Rejected(
                f"Step {sid}: an AI-authored link may join stated steps only."
            )
        if not step.get("text") or re.search(r"\d|[\"“”]|\[S", step["text"]):
            raise Rejected(
                f"Step {sid}: an AI link adds no figure, date, quotation or reference."
            )


def check_argument(derivation, sources):
    """Return steps by id after checking the chain reaches Z without loops."""
    derivation.update(check_apparatus(derivation))
    steps = derivation.get("steps") or []
    ids = [s.get("id") for s in steps]
    if not steps or None in ids or len(set(ids)) != len(ids):
        raise Rejected("steps must list the chain with unique ids.")
    if not derivation.get("conclusion"):
        raise Rejected("State the conclusion Z the chain establishes.")
    index = {s["id"]: s for s in steps}
    allowed = derivation.get("ai_allowed", [])
    for step in steps:
        check_step(step, index, sources, allowed)
        seen, cur = set(), step["id"]
        while cur != "Z":
            if cur in seen:
                raise Rejected(f"Step {step['id']}: the chain loops.")
            seen.add(cur)
            cur = index[cur].get("supports")
    return index


def check_sources(derivation, sources):
    """Every supplied source, used or not, needs a recorded contradiction check."""
    results = {
        c.get("source"): c.get("result", "") for c in derivation.get("checks", [])
    }
    missing = [label for label in sources if not results.get(label)]
    if missing:
        raise Rejected("Record a contradiction check for: " + ", ".join(missing) + ".")
    return results


def check_apparatus(derivation):
    """Return notes and conflicts as lists of sentences. One bare sentence is a list of one."""
    out = {}
    for field in ("notes", "conflicts"):
        value = derivation.get(field) or []
        value = [value] if isinstance(value, str) else value
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise Rejected(f"{field} must be a list of sentences.")
        out[field] = value
    return out


def coverage(src, spans):
    """Count the source's sentences that a used span touches."""
    found = [
        m.span() for m in re.finditer(r"\S[^.?!]*(?:[.?!]+[”’\"')\]]*|$)", src["text"])
    ]
    used = sum(any(s < e2 and s2 < e for s2, e2 in spans) for s, e in found)
    return used, len(found)
