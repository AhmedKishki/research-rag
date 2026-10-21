"""Render a white-box derivation: Synthesis, References, Provenance, Reasoning.

Copied from github.com/AhmedKishki/agent-skills @ 7236e5a,
`skills/white-box-synthesis/scripts/render.py`. The copy is deliberate: this app
depends on no other project by its author. Carry a fix to both copies.

Here the script's command-line entry is `render`, which refuses a malformed
derivation as the script's exit 1 does, and `render_parts`, which keeps the four
parts apart for a caller that lays them out itself.
"""

import re

from .checks import check_argument, check_sources, coverage
from .ops import Rejected
from .pieces import build_sentence, load_sources

PARTS = ("synthesis", "references", "provenance", "reasoning")

# What a derivation written by hand or by a model can raise before a rule
# names its fault. The script's entry reads the same set as a rejection.
_MALFORMED = (KeyError, TypeError, IndexError, AttributeError)


def link_text(step):
    kind = {"gap": ("GAP", "missing"), "ai": ("AI-AUTHORED", "text")}.get(
        step.get("status")
    )
    return f"⟦{kind[0]}: {step[kind[1]]}⟧" if kind else None


def render(d):
    """Return the four parts as one Markdown reply, or raise `Rejected`."""

    parts = render_parts(d)
    titles = dict(zip(PARTS, ("Synthesis", "References", "Provenance", "Reasoning")))
    return "\n\n".join(f"**{titles[p]}**\n\n{parts[p]}" for p in PARTS) + "\n"


def render_parts(d):
    """Return {synthesis, references, provenance, reasoning}, or raise `Rejected`."""

    try:
        return _render(d)
    except _MALFORMED as error:
        raise Rejected(f"The derivation is malformed: {error!r}.") from error


def _render(d):
    sources = load_sources(d.get("sources"))
    steps = check_argument(d, sources)
    results = check_sources(d, sources)
    order, built, last, cursor = list(steps), [], {}, -1
    pairs = d.get("normalisations", [])
    for n, sentence in enumerate(d.get("sentences") or [], 1):
        ids = sentence.get("step")
        ids = ids if isinstance(ids, list) else [ids]
        if any(i not in steps for i in ids):
            raise Rejected(f"Sentence {n}: unknown step {ids}.")
        if (k := min(order.index(i) for i in ids)) < cursor:
            raise Rejected(
                f"Sentence {n}: sentences must follow the chain order of steps."
            )
        cursor = k
        built.append(
            (n, sentence, ids, *build_sentence(sentence, steps, sources, pairs))
        )
        last.update({i: n for i in ids})
    if missing := [i for i in steps if i not in last]:
        raise Rejected(
            f"No source sentence for {', '.join(missing)}: drop it and mark a gap."
        )
    paras, prov, refs, spans = [[]], [], [], {}
    for n, sentence, ids, text, labels, line, used in built:
        if sentence.get("paragraph") and paras[-1]:
            paras.append([])
        paras[-1].append(text)
        prov.append(f"{n}. {line}")
        refs += labels
        for label, span in used:
            spans.setdefault(label, []).append(span)
        for sid in (i for i in steps if last[i] == n and link_text(steps[i])):
            paras[-1].append(mark := link_text(steps[sid]))
            refs += [r for r in sources if re.search(rf"\b{re.escape(r)}\b", mark)]
            kind = steps[sid]["status"].upper()
            prov.append(f"- After {n}: {kind} for {sid} → {steps[sid]['supports']}")
    return {
        "synthesis": "\n\n".join(" ".join(p) for p in paras),
        "references": "\n".join(reference(sources[r], r) for r in dict.fromkeys(refs)),
        "provenance": "\n".join(prov),
        "reasoning": "\n".join(reasoning(d, steps, sources, results, spans, built)),
    }


def reference(src, label):
    parts = [
        src.get("author"),
        src.get("title") and f"*{src['title']}*",
        src.get("locator"),
    ]
    note = f" {src['note']}." if src.get("note") else ""
    return f"- [{label}] " + ", ".join(p or "not supplied" for p in parts) + "." + note


def reasoning(d, steps, sources, results, spans, built):
    lines = [f"- Z: {d['conclusion']}", "- Chain:"]
    for sid, step in steps.items():
        labels = sorted(
            {lb for _, _, ids, _, lbs, _, _ in built if sid in ids for lb in lbs}
        )
        head = f"  - {sid} [{'; '.join(labels)}] {step.get('claim', '')}".rstrip()
        if sid == "Z":
            lines.append(f"{head}. Stated conclusion.")
            continue
        lic = step.get("licence") or {}
        link = {
            "part": "none needed, part of the answer",
            "gap": "gap, marked in the synthesis",
            "stated": f"stated by {lic.get('source')} in “{lic.get('text')}”",
            "ai": f"AI-authored with permission, for {step.get('missing')}",
        }[step["status"]]
        lines.append(
            f"{head} → {step['supports']} ({step.get('relation', 'part')}),"
            f" needed because {step['why'].rstrip('.')}. Link: {link}"
        )
    n = sum(s.get("status") in ("gap", "ai") for s in steps.values())
    lines.append(
        f"- Gaps and AI links on the chain: {n}. The sources establish Z only at 0."
    )
    left = [
        f"  - {x.get('source')}: “{x.get('text')}”\n    - {x.get('reason')}"
        for x in d.get("excluded", [])
    ]
    lines += ["- Left out:"] + left
    for label, src in sources.items():
        used, total = coverage(src, spans.get(label, []))
        lines.append(f"  - {label}: {used} of {total} sentences copied.")
    lines += ["- Conflicts:"] + (
        [f"  - {c}" for c in d.get("conflicts", [])] or ["  - None."]
    )
    lines += ["- Source checks:"] + [f"  - {k}: {v}" for k, v in results.items()]
    lines += [f"- Note: {note}" for note in d.get("notes", [])]
    lines.append(
        "- Wording check: passed by the white-box renderer. Meaning checks: best effort."
    )
    return lines
