# `core`

The service the three surfaces call, and the answers it gives when it cannot.

| Module | Holds |
|---|---|
| `service.py` | `ResearchService`: the composition root, the write and search queues, the read lease, and the BM25 retriever lock |
| `operations.py` | Every operation a surface forwards: its arguments, their bounds, and the service call |
| `admission.py` | `Admission`: a fixed number of turns handed out to waiting callers in rounds, each wait bounded |
| `answers.py` | `AnswerWorkflow`: one answer, refused before anything leaves the machine when offline, keyless, unbuilt or busy, and the account's key read and written for the workspace |
| `review.py` | `ReviewWorkflow`: what a reader decided about a source, a passage, or a metadata field |
| `review_state.py` | The four write operations over the three review files |
| `source_inventory.py` | Every source the project can name, and the catalog that remembers them |
| `records.py` | The five hand-editable record schemas and the current-generation pointer |
| `health.py` | The checks one health report is made of, and what each one blocks |
| `version.py` | The version the running process started with, and the one installed now |
| `status.py` | `StatusWorkflow`: the status payload, in one vocabulary with and without a generation |
| `stats.py` | `StatsWorkflow`: a search's first five ranks counted, and the stats answer beside the generation's own facts |
| `blocked_answers.py` | The three answers a surface gives when there is no corpus to answer about |
| `tool_views.py` | What an agent and a terminal are told, projected from what the service returned |

## Rules

- A workflow class here is a set of methods under the write lock or a read lease. Its algorithms are module-level functions, so a caller that is not a service can use them.
- No module here imports a surface.
- `source_inventory.known_sources` may rewrite `source-catalog.json`, because the catalog is the only memory of a source the project has seen and no longer holds.
- `tool_views.py` is the one place a payload is projected for a reader.
- `records.py` validates and never does I/O of its own. Every read refuses a file
  written for a later version and names the field it could not read.
- A surface forwards a corpus operation through `operations.run`; it adds presentation, never a second argument check.
