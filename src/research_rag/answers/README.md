# `answers`

The white-box answer: a derivation a model submits, rendered from exact source
wording or refused.

| Module | Holds |
|---|---|
| `render.py` | `render` and `render_parts`: the four parts of a reply, rendered from a derivation, or `Rejected` |
| `pieces.py` | COPY of an exact source span, its operations replayed, and pieces joined into a sentence |
| `ops.py` | DELETE, INFLECT and NORMALISE on their closed lists, and `Rejected` |
| `checks.py` | The argument chain, its links, and the per-source checks |

## Rules

- `render.py`, `pieces.py`, `ops.py` and `checks.py` copy the white-box-synthesis
  renderer from `AhmedKishki/agent-skills`; each docstring names the commit.
  Change their logic only to carry a fix that the skill also takes.
- Every character a reply holds outside a marker replays from a source span. A
  derivation that cannot be replayed is `Rejected`, never rendered in part.
- Nothing here reads a project, a setting or the network.
