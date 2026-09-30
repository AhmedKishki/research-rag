"""A research app with a browser workspace, an agent surface, and a command line.

One domain core with three surfaces over the same project state and one process
that owns it: the workspace (`surfaces.ui`), the agent's tools and resources
(`surfaces.mcp`), and the command line (`surfaces.cli`). The running app
(`app.py`) serves the first two on one loopback port and answers the third
through the control API (`control.py`), so one project lock and one UltraRAG
gateway serve all three.
"""
