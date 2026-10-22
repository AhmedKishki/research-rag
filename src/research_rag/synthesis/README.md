# `synthesis`

The white-box synthesis: a derivation a model submits, rendered from exact source
wording or refused.

| Module | Holds |
|---|---|
| `loop.py` | `run_synthesis`: the four tools DeepSeek answers through, the labelled passages, and the bounds on searches, attempts and time |
| `citations.py` | `source_citations`: a page, author–year, bracketed, or footnote citation a passage carried into a rendered synthesis |
| `deepseek.py` | `DeepSeek`: one connection to the chat-completions endpoint, and a refusal named with its remedy |
| `prompt.md` | The system prompt: the white-box procedure, condensed from the skill's references for these tools |
| `render.py` | `render` and `render_parts`: the four parts of a reply, each deviation and each sentence, rendered from a derivation, or `Rejected` |
| `ops.py` | Sources loaded, an exact span copied, `Rejected`, and the waiver a sentence's deviation sets for a form rule |
| `edits.py` | INFLECT and NORMALISE on their closed lists |
| `pieces.py` | A piece's operations replayed, and pieces joined into a sentence |
| `checks.py` | The argument chain, its links, and the per-source checks |
| `reasoning.py` | The References and the Reasoning, with each deviation and what it waived |

## Rules

- `render.py`, `ops.py`, `edits.py`, `pieces.py`, `checks.py` and `reasoning.py` copy the white-box-synthesis
  renderer from `AhmedKishki/agent-skills`; each docstring names the commit.
  Change their logic only to carry a fix that the skill also takes.
- Every character a reply holds outside a marker replays from a source span. A
  derivation that cannot be replayed is `Rejected`, never rendered in part.
- The renderer modules read no project, setting or network. `loop.py` reaches
  the corpus only through the callables the service lends it, and the network
  only through `deepseek.py`.
- The app builds a derivation's sources from the passages a run retrieved. Text
  the model puts there is discarded.
- A result the model wrote outside the renderer never reaches a reader, and
  every result names what was sent.
