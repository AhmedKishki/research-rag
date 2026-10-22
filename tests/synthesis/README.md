# Synthesis tests

The white-box synthesis, without a network or a model.

- `test_render.py`: the renderer copy renders and refuses each case the skill's own test names, renders the skill's worked example word for word, and refuses a malformed derivation instead of raising.
- `test_loop.py`: a scripted model over an in-memory corpus. A derivation renders from retrieved passages only, a forged source text or an unknown label or an AI-authored link is refused back to the model, attempts and searches are bounded, prose outside a tool is discarded, an excluded neighbour is never sent, a citation a passage carried is refused until the model deletes it, and a provider failure or a timeout still says what was sent.
- `test_citations.py`: each kind of citation a passage carries is found, and a label, a gap marker, a figure, a bare year, or an editorial mark is not.
- `test_deepseek.py`: the request a turn sends through `httpx.MockTransport`, and each refusal named with its remedy and never the key.
- `fixtures/`: the skill's worked examples, `mill.json` and `coop.json`, copied unchanged.
