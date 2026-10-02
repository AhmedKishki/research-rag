"""A research app with a browser workspace, an agent surface, and a command line.

The package is divided into folders, one per concern, and each folder carries a
`README.md` saying what it owns and what it may never do:

| Folder | Holds |
|---|---|
| `project/` | what this project is, how it is configured, the names its state is found by |
| `storage/` | durable writes and the record schemas they write |
| `core/` | the service the surfaces call, and the answers it gives when it cannot |
| `corpus/` | the corpus on disk, how it is read, and what its text says about itself |
| `generations/` | building a generation, and what is inside the one in use |
| `retrieval/` | answering a query against a built generation |
| `surfaces/` | the command line, the agent surface, and the browser workspace |
| `runtime/` | the running process and the commands that manage it |
| `gateway/` | the stdio MCP gateway to UltraRAG, and the runtime it proxies |

`AGENTS.md` holds the architecture the three surfaces share, and the gates that
keep the folders from reaching into each other.
"""
