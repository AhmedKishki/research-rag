"""Write the References and the Reasoning: the agent's own wording, never a source.

Copied from github.com/AhmedKishki/agent-skills @ ea68182,
`skills/white-box-synthesis/scripts/reasoning.py`. The copy is deliberate; carry a fix
to both copies.
"""

from .checks import coverage


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
            {lb for _, _, ids, _, lbs, *_ in built if sid in ids for lb in lbs}
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
    devs = [
        f"  - Sentence {n}: {s['deviation']} "
        + (f"Waived: {' '.join(w)}" if w else "No rule needed it.")
        for n, s, *_, w in built
        if s.get("deviation")
    ]
    lines += ["- Deviations:"] + (devs or ["  - None."])
    lines += [f"- Note: {note}" for note in d.get("notes", [])]
    lines.append(
        "- Wording check: passed by the white-box renderer. Meaning checks: best effort."
    )
    return lines
