# `core`

The service the three surfaces call, and the answers it gives when it cannot.

| Module | Holds |
|---|---|
| `service.py` | `ResearchService`: the composition root, the write and search queues, the read lease, and the BM25 retriever lock |
| `admission.py` | `Admission`: a fixed number of turns handed out to waiting callers in rounds, each wait bounded |
| `review.py` | `ReviewWorkflow`: what a reader decided about a source, a passage, or a metadata field |
| `review_state.py` | The four write operations over the three review files |
| `source_inventory.py` | Every source the project can name, and the catalog that remembers them |
| `status.py` | `StatusWorkflow`: the status payload, in one vocabulary with and without a generation |
| `stats.py` | `StatsWorkflow`: a search's first five ranks counted, and the stats answer beside the generation's own facts |
| `blocked_answers.py` | The three answers a surface gives when there is no corpus to answer about |
| `tool_views.py` | What an agent and a terminal are told, projected from what the service returned |

## Rules

- A workflow class here is a set of methods under the write lock or a read lease. Its algorithms are module-level functions, so a caller that is not a service can use them.
- No module here imports a surface.
- `source_inventory.known_sources` may rewrite `source-catalog.json`, because the catalog is the only memory of a source the project has seen and no longer holds.
- `tool_views.py` is the one place a payload is projected for a reader.
