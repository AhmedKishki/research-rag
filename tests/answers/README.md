# Answer tests

The white-box answer, without a network or a model.

- `test_render.py`: the renderer copy renders and refuses each case the skill's own test names, renders the skill's worked example word for word, and refuses a malformed derivation instead of raising.
- `fixtures/`: the skill's worked examples, `mill.json` and `coop.json`, copied unchanged.
