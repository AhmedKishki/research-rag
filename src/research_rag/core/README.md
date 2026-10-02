# What this project is

The service the three surfaces call, and the answers it gives when it cannot.

| Module | Holds |
|---|---|
| `service.py` | `ResearchService`: the composition root, and the operation lock every read and write takes |
| `review.py` | `ReviewWorkflow`: what a reader decided about a source, a passage, or a metadata field |
| `review_state.py` | The four write operations over the three review files |
| `source_inventory.py` | Every source the project can name, and the catalog that remembers them |
| `status.py` | `StatusWorkflow`: the status payload, in one vocabulary with and without a generation |
| `blocked_answers.py` | The three answers a surface gives when there is no corpus to answer about |
| `tool_views.py` | What an agent and a terminal are told, projected from what the service returned |

## Rules

- A workflow class here is a set of methods under the operation lock. Its
  algorithms are module-level functions in this folder, so a caller that is not a
  service can use them without one.
- No module here imports a surface. A status reader that needed `surfaces/mcp.py`
  would be reading a front end to describe the engine.
- `source_inventory.known_sources` is allowed to rewrite `source-catalog.json`,
  and that is the price of remembering a source the project has seen and no longer
  holds. It is not an accident: the catalog is the only memory of it.
- `tool_views.py` is the one place a payload is projected for a reader. A second
  projection is a second answer to the same question.

## What lives elsewhere now

Building the generation this service reads is `../generations/`, and answering a
query against it is `../retrieval/`.