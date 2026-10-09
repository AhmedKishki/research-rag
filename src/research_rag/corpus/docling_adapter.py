"""Turn a Docling ``export_to_dict`` payload into extraction units.

The adapter is deliberately pure: it takes the JSON-serializable mapping Docling
produces and returns plain dictionaries, so it imports neither Docling nor
PyMuPDF and can be exercised with a recorded payload. The bounded worker calls
it; the parent never imports Docling.

Reading order follows the Document's own body tree (``body.children``, recursing
through ``groups``) whenever it is present, so a picture keeps its place beside
the prose around it. When the body tree is absent the adapter falls back to the
deterministic payload order of ``texts`` and then ``pictures``.

Physical page locators come from each item's provenance ``page_no``; every
provenance entry is preserved, with its ``charspan`` and bounding box. A
multipage item is split across its physical pages only when its charspans form
one contiguous, non-overlapping partition of the whole text; otherwise the text
stays whole and all provenance is retained, so no page is given words it never
held.

A picture's caption is kept and a picture's non-caption descendants — nested
through groups as well as listed directly — are excluded and counted, because a
recognised figure's interior text is unreadable on a scanned body. A caption that
cannot be hung on an in-range picture is emitted on its own physical page rather
than dropped. Tables, key-value regions, and forms are adapted from their cell
text with provenance; a nonempty unsupported item is counted and named rather
than silently discarded. No page is declared image-only here; a captionless
figure on a textless page yields no unit and no scan claim, which the source
analysis owns.
"""

from __future__ import annotations

from typing import Any

# Labels Docling uses for page furniture. A running head or foot is not
# evidence, and the document layer already separates the two.
FURNITURE_LABELS = frozenset(
    {"page_header", "page_footer", "page_number", "document_index"}
)

# Labels that name a list rather than prose, so a page's list items keep the
# list kind the chunker and statistics read.
LIST_LABELS = frozenset({"list_item", "list"})

# Labels a picture's caption carries. A caption is evidence; the caption's own
# label is preserved for the unit.
CAPTION_LABELS = frozenset({"caption"})

# Every collection a serialized Document can hold an item in.
_ITEM_COLLECTIONS = (
    "texts",
    "pictures",
    "tables",
    "groups",
    "key_value_items",
    "form_items",
)

# A bounded sample of unsupported items is reported beside the count, so a
# reviewer can trace an incompleteness without a per-item log.
_UNSUPPORTED_SAMPLE_LIMIT = 32


def _ref(entry: Any) -> str | None:
    if isinstance(entry, dict) and isinstance(entry.get("$ref"), str):
        return str(entry["$ref"])
    return None


def _self_ref(item: dict[str, Any]) -> str | None:
    ref = item.get("self_ref")
    return str(ref) if isinstance(ref, str) else None


def _label(item: dict[str, Any]) -> str:
    return str(item.get("label") or "text")


def _prov_entries(item: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for prov in item.get("prov") or []:
        if not isinstance(prov, dict):
            continue
        page_no = prov.get("page_no")
        if not isinstance(page_no, int):
            continue
        entry: dict[str, Any] = {
            "page_no": page_no,
            "charspan": list(prov.get("charspan") or []),
        }
        if isinstance(bbox := prov.get("bbox"), dict):
            entry["bbox"] = {
                key: bbox.get(key) for key in ("l", "t", "r", "b", "coord_origin")
            }
        entries.append(entry)
    return entries


def _pages(item: dict[str, Any]) -> list[int]:
    return [int(entry["page_no"]) for entry in _prov_entries(item)]


def _first_page(item: dict[str, Any]) -> int | None:
    for entry in _prov_entries(item):
        return int(entry["page_no"])
    return None


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _bbox_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {key: value.get(key) for key in ("l", "t", "r", "b", "coord_origin")}


def _build_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for key in _ITEM_COLLECTIONS:
        for item in payload.get(key) or []:
            if isinstance(item, dict) and (ref := _self_ref(item)) is not None:
                index[ref] = item
    for key in ("body", "furniture"):
        root = payload.get(key)
        if isinstance(root, dict) and (ref := _self_ref(root)) is not None:
            index[ref] = root
    return index


def _walk_children(
    item: dict[str, Any], index: dict[str, dict[str, Any]], seen: set[str]
):
    """Yield a node's descendant refs depth-first, in declared order."""

    for child in item.get("children") or []:
        cref = _ref(child)
        if cref is None or cref in seen:
            continue
        seen.add(cref)
        yield cref
        resolved = index.get(cref)
        if isinstance(resolved, dict):
            yield from _walk_children(resolved, index, seen)


def _declared_caption_refs(picture: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for child in picture.get("captions") or []:
        ref = _ref(child)
        if ref is not None and ref not in refs:
            refs.append(ref)
    return refs


def _caption_refs(
    picture: dict[str, Any], index: dict[str, dict[str, Any]]
) -> list[str]:
    """A picture's captions in stable order: declared refs, then labelled children."""

    refs = _declared_caption_refs(picture)
    for ref in _walk_children(picture, index, set()):
        resolved = index.get(ref)
        if (
            isinstance(resolved, dict)
            and _label(resolved) in CAPTION_LABELS
            and ref not in refs
        ):
            refs.append(ref)
    return refs


def _unambiguous_partition(entries: list[dict[str, Any]], length: int) -> bool:
    """Whether charspans already, in order, tile ``text`` exactly once."""

    if len(entries) < 2 or length <= 0:
        return False
    spans: list[tuple[int, int]] = []
    for entry in entries:
        span = entry.get("charspan")
        if not (isinstance(span, list) and len(span) == 2):
            return False
        if not all(isinstance(bound, int) for bound in span):
            return False
        spans.append((span[0], span[1]))
    if spans[0][0] != 0 or spans[-1][1] != length:
        return False
    for index, (start, end) in enumerate(spans):
        if end <= start:
            return False
        if index and start != spans[index - 1][1]:
            return False
    return True


def _content_kind(label: str) -> str:
    if label in LIST_LABELS:
        return "list"
    if label == "table":
        return "table"
    return "prose"


def _table_text_and_provenance(
    table: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """A table's row text and provenance, or ``(None, None)`` when unsupported."""

    data = table.get("data")
    if not isinstance(data, dict):
        return None, None
    cells = data.get("table_cells")
    if not isinstance(cells, list):
        return None, None
    provenance = _prov_entries(table)
    page_no = _first_page(table)
    rows: dict[int, list[str]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        text = str(cell.get("text") or "").strip()
        if not text:
            continue
        rows.setdefault(_int(cell.get("start_row_offset_idx")), []).append(text)
        entry: dict[str, Any] = {"page_no": page_no, "charspan": []}
        if (bbox := _bbox_entry(cell.get("bbox"))) is not None:
            entry["bbox"] = bbox
        provenance.append(entry)
    text = "\n".join(" | ".join(row) for _index, row in sorted(rows.items()))
    return text, provenance


def _graph_text_and_provenance(
    item: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """A key-value or form region's cell text and provenance, or unsupported."""

    graph = item.get("graph")
    if not isinstance(graph, dict):
        return None, None
    cells = graph.get("cells")
    if not isinstance(cells, list):
        return None, None
    texts: list[str] = []
    provenance = _prov_entries(item)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        text = str(cell.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        provenance.extend(_prov_entries(cell))
    return "\n".join(texts), provenance


class _DocumentAdapter:
    """One payload's adaptation, keeping the state its emission rules share."""

    def __init__(
        self, payload: dict[str, Any], page_first: int, page_last: int
    ) -> None:
        self.payload = payload
        self.page_first = page_first
        self.page_last = page_last
        self.index = _build_index(payload)
        self.texts: dict[str, dict[str, Any]] = {
            ref: item
            for item in payload.get("texts") or []
            if isinstance(item, dict) and (ref := _self_ref(item)) is not None
        }
        self.pictures = [
            item for item in payload.get("pictures") or [] if isinstance(item, dict)
        ]
        self.tables = [
            item for item in payload.get("tables") or [] if isinstance(item, dict)
        ]
        self.key_values = [
            item
            for item in payload.get("key_value_items") or []
            if isinstance(item, dict)
        ]
        self.forms = [
            item for item in payload.get("form_items") or [] if isinstance(item, dict)
        ]
        self.graph_items = self.key_values + self.forms
        self.picture_refs = self._refs(self.pictures)
        self.table_refs = self._refs(self.tables)
        self.graph_refs = self._refs(self.graph_items)
        self.form_refs = self._refs(self.forms)
        self.pages: dict[int, list[dict[str, Any]]] = {}
        self.diagnostics: dict[str, Any] = {
            "retained_item_count": 0,
            "excluded_furniture_item_count": 0,
            "excluded_picture_child_item_count": 0,
            "excluded_picture_child_character_count": 0,
            "excluded_empty_item_count": 0,
            "caption_item_count": 0,
            "orphan_caption_item_count": 0,
            "missing_caption_ref_count": 0,
            "missing_child_ref_count": 0,
            "table_item_count": 0,
            "table_without_text_item_count": 0,
            "unsupported_table_item_count": 0,
            "form_item_count": 0,
            "unsupported_form_item_count": 0,
            "key_value_item_count": 0,
            "unsupported_key_value_item_count": 0,
            "unsupported_item_count": 0,
            "unsupported_item_refs": [],
            "unordered_item_count": 0,
        }
        self.emitted: set[str] = set()
        self.handled: set[str] = set()
        self.caption_refs: set[str] = set()
        self.classified: set[str] = set()

    @staticmethod
    def _refs(items: list[dict[str, Any]]) -> set[str]:
        return {ref for item in items if (ref := _self_ref(item)) is not None}

    def run(
        self,
    ) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
        self._classify()
        for ref in self._order():
            self._dispatch(ref)
        self._emit_orphan_captions()
        self._emit_unordered_remainder()
        return self.pages, self.diagnostics

    def _in_range(self, page_no: int | None) -> bool:
        return isinstance(page_no, int) and self.page_first <= page_no <= self.page_last

    def _first_page_in_range(self, item: dict[str, Any]) -> int | None:
        for entry in _prov_entries(item):
            if self._in_range(int(entry["page_no"])):
                return int(entry["page_no"])
        return None

    def _any_page_in_range(self, item: dict[str, Any]) -> bool:
        return any(self._in_range(page) for page in _pages(item))

    def _is_furniture(self, item: dict[str, Any]) -> bool:
        layer = str(item.get("content_layer") or "body")
        return layer == "furniture" or _label(item) in FURNITURE_LABELS

    def _add(
        self, page_no: int, contents: str, kind: str, provenance: list[dict[str, Any]]
    ) -> None:
        self.pages.setdefault(page_no, []).append(
            {"contents": contents, "content_kind": kind, "provenance": provenance}
        )
        self.diagnostics["retained_item_count"] += 1

    def _record_unsupported(self, ref: str | None, kind: str) -> None:
        self.diagnostics["unsupported_item_count"] += 1
        if len(self.diagnostics["unsupported_item_refs"]) < _UNSUPPORTED_SAMPLE_LIMIT:
            self.diagnostics["unsupported_item_refs"].append({"ref": ref, "kind": kind})

    def _classify(self) -> None:
        """Map caption owners and mark furniture, picture children, and empties."""

        owners: dict[str, str | None] = {}
        for picture in self.pictures:
            picture_ref = _self_ref(picture)
            for ref in _caption_refs(picture, self.index):
                owners.setdefault(ref, picture_ref)
        self.caption_refs = set(owners)
        self.caption_refs.update(
            ref for ref, item in self.texts.items() if _label(item) in CAPTION_LABELS
        )

        # A picture's non-caption descendants are excluded structurally, nested
        # groups included, never by a guess about how long a child's words are.
        excluded: set[str] = set()
        for picture in self.pictures:
            for ref in _walk_children(picture, self.index, set()):
                if ref in self.caption_refs or ref in excluded:
                    continue
                if ref in self.texts:
                    excluded.add(ref)

        for picture in self.pictures:
            for child in picture.get("children") or []:
                ref = _ref(child)
                if ref is not None and ref not in self.index:
                    self.diagnostics["missing_child_ref_count"] += 1
            for caption in picture.get("captions") or []:
                ref = _ref(caption)
                if ref is not None and ref not in self.index:
                    self.diagnostics["missing_caption_ref_count"] += 1

        for ref, item in self.texts.items():
            if self._is_furniture(item):
                self.diagnostics["excluded_furniture_item_count"] += 1
                self.classified.add(ref)
            elif ref in excluded:
                self.diagnostics["excluded_picture_child_item_count"] += 1
                self.diagnostics["excluded_picture_child_character_count"] += len(
                    str(item.get("text") or "")
                )
                self.classified.add(ref)
            elif not str(item.get("text") or "").strip():
                self.diagnostics["excluded_empty_item_count"] += 1
                self.classified.add(ref)

    def _order(self) -> list[str]:
        body = self.payload.get("body")
        if isinstance(body, dict) and (body.get("children") or []):
            return list(_walk_children(body, self.index, set()))
        order = list(self.texts)
        order += [ref for item in self.pictures if (ref := _self_ref(item)) is not None]
        order += [ref for item in self.tables if (ref := _self_ref(item)) is not None]
        order += [
            ref for item in self.graph_items if (ref := _self_ref(item)) is not None
        ]
        return order

    def _dispatch(self, ref: str) -> None:
        if ref in self.handled:
            return
        self.handled.add(ref)
        if ref in self.texts:
            if ref not in self.classified and ref not in self.caption_refs:
                self._emit_text(ref)
        elif ref in self.picture_refs:
            self._emit_picture(ref)
        elif ref in self.table_refs:
            self._emit_table(ref)
        elif ref in self.graph_refs:
            self._emit_graph(ref, "form" if ref in self.form_refs else "key_value")

    def _emit_text(self, ref: str) -> bool:
        item = self.texts[ref]
        text = str(item.get("text") or "")
        kind = _content_kind(_label(item))
        provenance = _prov_entries(item)
        if _unambiguous_partition(provenance, len(text)):
            wrote = False
            for entry in provenance:
                start, end = entry["charspan"]
                if not self._in_range(entry["page_no"]):
                    continue
                piece = text[start:end]
                if piece.strip():
                    self._add(int(entry["page_no"]), piece, kind, [entry])
                    wrote = True
            return wrote
        page = _first_page(item)
        if not self._in_range(page):
            return False
        self._add(int(page), text, kind, provenance)
        return True

    def _emit_picture(self, ref: str) -> bool:
        picture = self.index.get(ref)
        if not isinstance(picture, dict) or not self._any_page_in_range(picture):
            return False
        caption_items = [
            (caption_ref, self.texts[caption_ref])
            for caption_ref in _caption_refs(picture, self.index)
            if caption_ref in self.texts
            and caption_ref not in self.emitted
            and str(self.texts[caption_ref].get("text") or "").strip()
        ]
        page = self._first_page_in_range(picture)
        if not caption_items or page is None:
            return False
        provenance = _prov_entries(picture)
        contents: list[str] = []
        for caption_ref, caption_item in caption_items:
            contents.append(str(caption_item.get("text")))
            provenance.extend(_prov_entries(caption_item))
            self.emitted.add(caption_ref)
            self.handled.add(caption_ref)
            self.diagnostics["caption_item_count"] += 1
        self._add(page, "\n\n".join(contents), "figure", provenance)
        self.emitted.add(ref)
        return True

    def _emit_table(self, ref: str) -> bool:
        table = self.index.get(ref)
        if not isinstance(table, dict):
            return False
        if self._is_furniture(table):
            self.diagnostics["excluded_furniture_item_count"] += 1
            return False
        if not self._any_page_in_range(table):
            return False
        text, provenance = _table_text_and_provenance(table)
        if text is None or provenance is None:
            self.diagnostics["unsupported_table_item_count"] += 1
            self._record_unsupported(ref, "table")
            return False
        if not text.strip():
            self.diagnostics["table_without_text_item_count"] += 1
            return False
        page = self._first_page_in_range(table)
        if page is None:
            return False
        self._add(page, text, "table", provenance)
        self.diagnostics["table_item_count"] += 1
        return True

    def _emit_graph(self, ref: str, kind: str) -> bool:
        item = self.index.get(ref)
        if not isinstance(item, dict):
            return False
        if self._is_furniture(item):
            self.diagnostics["excluded_furniture_item_count"] += 1
            return False
        if not self._any_page_in_range(item):
            return False
        text, provenance = _graph_text_and_provenance(item)
        if text is None or not text.strip() or provenance is None:
            self.diagnostics[f"unsupported_{kind}_item_count"] += 1
            self._record_unsupported(ref, kind)
            return False
        page = self._first_page_in_range(item)
        if page is None:
            return False
        self._add(page, text, "prose", provenance)
        self.diagnostics[f"{kind}_item_count"] += 1
        return True

    def _emit_orphan_captions(self) -> None:
        # A caption kept out of a picture by an out-of-range or unresolvable
        # owner is still evidence on its own physical page.
        for ref, item in self.texts.items():
            if ref in self.emitted or ref in self.classified:
                continue
            if ref not in self.caption_refs:
                continue
            page = self._first_page_in_range(item)
            if page is None or not str(item.get("text") or "").strip():
                continue
            self._add(page, str(item.get("text")), "figure", _prov_entries(item))
            self.emitted.add(ref)
            self.diagnostics["orphan_caption_item_count"] += 1
            self.diagnostics["caption_item_count"] += 1

    def _emit_unordered_remainder(self) -> None:
        # Anything the body tree did not place is emitted in payload order
        # rather than dropped, and the ordering is disclosed.
        for ref in self.texts:
            if ref in self.emitted or ref in self.handled:
                continue
            if ref in self.classified or ref in self.caption_refs:
                continue
            if self._emit_text(ref):
                self.diagnostics["unordered_item_count"] += 1
        for item in self.pictures:
            ref = _self_ref(item)
            if ref is None or ref in self.emitted or ref in self.handled:
                continue
            if self._emit_picture(ref):
                self.diagnostics["unordered_item_count"] += 1
        for item in self.tables:
            ref = _self_ref(item)
            if ref is None or ref in self.emitted or ref in self.handled:
                continue
            if self._emit_table(ref):
                self.diagnostics["unordered_item_count"] += 1
        for item in self.graph_items:
            ref = _self_ref(item)
            if ref is None or ref in self.emitted or ref in self.handled:
                continue
            kind = "form" if item in self.forms else "key_value"
            if self._emit_graph(ref, kind):
                self.diagnostics["unordered_item_count"] += 1


def adapt_document(
    payload: dict[str, Any],
    *,
    page_first: int,
    page_last: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Adapt one conversion result into per-page units and document diagnostics.

    ``page_first`` and ``page_last`` are 1-based physical page numbers and bound
    the pages this payload was converted from. Returns a mapping from physical
    page number to that page's units, and a diagnostics mapping. Each unit is
    ``{"contents", "content_kind", "provenance"}``.
    """

    return _DocumentAdapter(payload, page_first, page_last).run()
