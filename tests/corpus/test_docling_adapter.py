"""The pure Docling adapter: no Docling, no PDF library, recorded payloads only.

Payloads are minimal but schema-faithful: ``body``/``furniture`` roots, ``texts``,
``pictures``, ``groups``, ``tables``, and ``form_items`` carry the field names
Docling 2.135 serializes (``self_ref``, ``parent``/``$ref``, ``children``,
``captions``, ``prov`` with ``page_no``/``charspan``/``bbox``). No source text is
copied from a real document.
"""

from __future__ import annotations

import pytest

from research_rag.corpus.docling_adapter import adapt_document


def _prov(page_no: int, start: int, end: int, *, bbox: bool = False) -> dict:
    entry: dict = {"page_no": page_no, "charspan": [start, end]}
    if bbox:
        entry["bbox"] = {
            "l": 1.0,
            "t": 2.0,
            "r": 3.0,
            "b": 4.0,
            "coord_origin": "BOTTOMLEFT",
        }
    return entry


def _text(
    self_ref: str,
    text: str,
    *,
    label: str = "text",
    layer: str = "body",
    parent: str = "#/body",
    page_no: int = 1,
    charspan: tuple[int, int] | None = None,
) -> dict:
    span = list(charspan) if charspan is not None else [0, len(text)]
    return {
        "self_ref": self_ref,
        "parent": {"$ref": parent},
        "children": [],
        "content_layer": layer,
        "label": label,
        "prov": [_prov(page_no, span[0], span[1])],
        "text": text,
    }


def _picture(
    self_ref: str,
    *,
    children: list[str],
    captions: list[str],
    page_no: int = 1,
    parent: str = "#/body",
) -> dict:
    return {
        "self_ref": self_ref,
        "parent": {"$ref": parent},
        "children": [{"$ref": ref} for ref in children],
        "content_layer": "body",
        "label": "picture",
        "captions": [{"$ref": ref} for ref in captions],
        "prov": [_prov(page_no, 0, 0)],
    }


def _body(*refs: str) -> dict:
    return {
        "self_ref": "#/body",
        "children": [{"$ref": ref} for ref in refs],
        "content_layer": "body",
        "label": "unspecified",
        "name": "_root_",
    }


def _payload(**overrides: object) -> dict:
    payload: dict = {"texts": [], "pictures": [], "groups": [], "tables": []}
    payload.update(overrides)
    return payload


# --- furniture -----------------------------------------------------------------


def test_header_label_is_excluded_even_on_the_body_layer() -> None:
    payload = _payload(
        body=_body("#/texts/0", "#/texts/1"),
        texts=[
            _text("#/texts/0", "RUNNING HEAD", label="page_header", layer="body"),
            _text("#/texts/1", "Body evidence.", page_no=1),
        ],
    )
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert [unit["contents"] for unit in pages[1]] == ["Body evidence."]
    assert diagnostics["excluded_furniture_item_count"] == 1


def test_furniture_layer_is_excluded_even_with_a_body_label() -> None:
    payload = _payload(
        body=_body("#/texts/0", "#/texts/1"),
        texts=[
            _text("#/texts/0", "Marginal note.", label="text", layer="furniture"),
            _text("#/texts/1", "Body evidence.", page_no=1),
        ],
    )
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert [unit["contents"] for unit in pages[1]] == ["Body evidence."]
    assert diagnostics["excluded_furniture_item_count"] == 1


# --- reading order and pictures ------------------------------------------------


def test_picture_keeps_its_body_position_and_chart_child_is_excluded() -> None:
    payload = _payload(
        body=_body("#/texts/0", "#/pictures/0", "#/texts/2"),
        texts=[
            _text("#/texts/0", "AAA"),
            _text(
                "#/texts/1",
                "Figure 1. A chart.",
                label="caption",
                parent="#/pictures/0",
            ),
            _text("#/texts/2", "BBB"),
            _text("#/texts/3", "garbled chart interior", parent="#/pictures/0"),
        ],
        pictures=[
            _picture(
                "#/pictures/0",
                children=["#/texts/1", "#/texts/3"],
                captions=["#/texts/1"],
            )
        ],
    )
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert [(unit["content_kind"], unit["contents"]) for unit in pages[1]] == [
        ("prose", "AAA"),
        ("figure", "Figure 1. A chart."),
        ("prose", "BBB"),
    ]
    assert diagnostics["excluded_picture_child_item_count"] == 1
    assert diagnostics["caption_item_count"] == 1
    figure = next(unit for unit in pages[1] if unit["content_kind"] == "figure")
    assert figure["provenance"][0]["page_no"] == 1
    assert figure["provenance"][-1]["page_no"] == 1


def test_caption_labelled_child_survives_without_a_captions_ref() -> None:
    payload = _payload(
        body=_body("#/pictures/0"),
        texts=[
            _text(
                "#/texts/0",
                "Figure 2. Survives.",
                label="caption",
                parent="#/pictures/0",
            ),
        ],
        pictures=[_picture("#/pictures/0", children=["#/texts/0"], captions=[])],
    )
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert [(unit["content_kind"], unit["contents"]) for unit in pages[1]] == [
        ("figure", "Figure 2. Survives."),
    ]
    assert diagnostics["caption_item_count"] == 1
    assert diagnostics["excluded_picture_child_item_count"] == 0


def test_nested_group_chart_descendants_are_excluded() -> None:
    payload = _payload(
        body=_body("#/pictures/0"),
        groups=[
            {
                "self_ref": "#/groups/0",
                "parent": {"$ref": "#/pictures/0"},
                "children": [{"$ref": "#/texts/2"}],
                "content_layer": "body",
                "label": "group",
            }
        ],
        texts=[
            _text(
                "#/texts/1",
                "Figure 3. Caption.",
                label="caption",
                parent="#/pictures/0",
            ),
            _text("#/texts/2", "axis junk", parent="#/groups/0"),
        ],
        pictures=[
            _picture(
                "#/pictures/0",
                children=["#/groups/0", "#/texts/1"],
                captions=["#/texts/1"],
            )
        ],
    )
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert [unit["contents"] for unit in pages[1]] == ["Figure 3. Caption."]
    assert diagnostics["excluded_picture_child_item_count"] == 1


def test_caption_order_is_stable_in_declared_order() -> None:
    payload = _payload(
        body=_body("#/pictures/0"),
        texts=[
            _text("#/texts/0", "ONE", label="caption", parent="#/pictures/0"),
            _text("#/texts/1", "TWO", label="caption", parent="#/pictures/0"),
        ],
        pictures=[
            _picture(
                "#/pictures/0",
                children=["#/texts/0", "#/texts/1"],
                captions=["#/texts/1", "#/texts/0"],
            )
        ],
    )
    pages, _diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert [unit["contents"] for unit in pages[1]] == ["TWO\n\nONE"]


# --- dropped captions and stray refs -------------------------------------------


def test_out_of_range_picture_caption_is_emitted_on_its_own_page() -> None:
    payload = _payload(
        body=_body("#/pictures/0"),
        texts=[
            _text(
                "#/texts/0",
                "Figure 9. Orphan caption.",
                label="caption",
                parent="#/pictures/0",
                page_no=5,
            ),
        ],
        pictures=[
            _picture(
                "#/pictures/0",
                children=["#/texts/0"],
                captions=["#/texts/0"],
                page_no=4,
            )
        ],
    )
    pages, diagnostics = adapt_document(payload, page_first=5, page_last=5)
    assert [(unit["content_kind"], unit["contents"]) for unit in pages[5]] == [
        ("figure", "Figure 9. Orphan caption.")
    ]
    assert pages[5][0]["provenance"][0]["page_no"] == 5
    assert diagnostics["orphan_caption_item_count"] == 1
    assert diagnostics["caption_item_count"] == 1


def test_stray_message_refs_are_counted_not_silent() -> None:
    payload = _payload(
        body=_body("#/pictures/0"),
        texts=[],
        pictures=[
            {
                "self_ref": "#/pictures/0",
                "parent": {"$ref": "#/body"},
                "children": [{"$ref": "#/texts/99"}],
                "captions": [{"$ref": "#/texts/98"}],
                "content_layer": "body",
                "label": "picture",
                "prov": [_prov(1, 0, 0)],
            }
        ],
    )
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert pages == {}
    assert diagnostics["missing_child_ref_count"] == 1
    assert diagnostics["missing_caption_ref_count"] == 1


# --- multipage provenance ------------------------------------------------------


def test_multipage_unambiguous_charspan_splits_by_physical_page() -> None:
    item = {
        "self_ref": "#/texts/0",
        "content_layer": "body",
        "label": "text",
        "prov": [_prov(2, 0, 5), _prov(3, 5, 10)],
        "text": "abcdeFGHIJ",
    }
    payload = _payload(body=_body("#/texts/0"), texts=[item])
    pages, _diagnostics = adapt_document(payload, page_first=2, page_last=3)
    assert set(pages) == {2, 3}
    assert pages[2][0]["contents"] == "abcde"
    assert pages[2][0]["provenance"] == [{"page_no": 2, "charspan": [0, 5]}]
    assert pages[3][0]["contents"] == "FGHIJ"
    assert pages[3][0]["provenance"] == [{"page_no": 3, "charspan": [5, 10]}]


def test_multipage_ambiguous_charspan_retains_all_and_fabricates_no_page() -> None:
    item = {
        "self_ref": "#/texts/0",
        "content_layer": "body",
        "label": "text",
        "prov": [_prov(2, 0, 5), _prov(3, 5, 10)],
        # Spans stop at 10 but the text is 15 characters, so the partition is
        # not a full tiling and must not be trusted to split.
        "text": "spans two pages",
    }
    payload = _payload(body=_body("#/texts/0"), texts=[item])
    pages, _diagnostics = adapt_document(payload, page_first=2, page_last=3)
    assert set(pages) == {2}
    unit = pages[2][0]
    assert unit["contents"] == "spans two pages"
    assert [entry["page_no"] for entry in unit["provenance"]] == [2, 3]
    assert [entry["charspan"] for entry in unit["provenance"]] == [[0, 5], [5, 10]]


def test_pages_outside_the_batch_range_are_ignored() -> None:
    payload = _payload(
        body=_body("#/texts/0"),
        texts=[_text("#/texts/0", "elsewhere", page_no=9)],
    )
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=2)
    assert pages == {}
    assert diagnostics["retained_item_count"] == 0


# --- tables, forms, key-value --------------------------------------------------


def test_nonempty_table_is_adapted_with_provenance() -> None:
    table = {
        "self_ref": "#/tables/0",
        "parent": {"$ref": "#/body"},
        "children": [],
        "content_layer": "body",
        "label": "table",
        "prov": [_prov(1, 0, 0)],
        "data": {
            "num_rows": 1,
            "num_cols": 2,
            "table_cells": [
                {
                    "text": "Header",
                    "start_row_offset_idx": 0,
                    "start_col_offset_idx": 0,
                },
                {
                    "text": "value",
                    "start_row_offset_idx": 0,
                    "start_col_offset_idx": 1,
                    "bbox": {
                        "l": 1.0,
                        "t": 2.0,
                        "r": 3.0,
                        "b": 4.0,
                        "coord_origin": "BOTTOMLEFT",
                    },
                },
            ],
        },
    }
    payload = _payload(body=_body("#/tables/0"), tables=[table])
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert [(unit["content_kind"], unit["contents"]) for unit in pages[1]] == [
        ("table", "Header | value")
    ]
    assert pages[1][0]["provenance"][0]["page_no"] == 1
    assert any(
        entry.get("bbox", {}).get("l") == 1.0 for entry in pages[1][0]["provenance"]
    )
    assert diagnostics["table_item_count"] == 1


def test_empty_table_is_counted_not_emitted() -> None:
    table = {
        "self_ref": "#/tables/0",
        "content_layer": "body",
        "label": "table",
        "prov": [_prov(1, 0, 0)],
        "data": {"num_rows": 1, "num_cols": 1, "table_cells": []},
    }
    payload = _payload(body=_body("#/tables/0"), tables=[table])
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert pages == {}
    assert diagnostics["table_without_text_item_count"] == 1


def test_unsupported_nonempty_form_is_surfaced_not_dropped_silently() -> None:
    form = {
        "self_ref": "#/form_items/0",
        "content_layer": "body",
        "label": "form",
        "prov": [_prov(1, 0, 0)],
        "graph": {},
    }
    payload = _payload(body=_body("#/form_items/0"), form_items=[form])
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert pages == {}
    assert diagnostics["unsupported_form_item_count"] == 1
    assert diagnostics["unsupported_item_count"] == 1
    assert diagnostics["unsupported_item_refs"] == [
        {"ref": "#/form_items/0", "kind": "form"}
    ]


def test_key_value_graph_cells_are_adapted() -> None:
    item = {
        "self_ref": "#/key_value_items/0",
        "content_layer": "body",
        "label": "key_value_region",
        "prov": [_prov(1, 0, 0)],
        "graph": {
            "cells": [
                {"cell_id": 0, "label": "key", "text": "Term", "prov": _prov(1, 0, 4)},
                {
                    "cell_id": 1,
                    "label": "value",
                    "text": "Definition",
                    "prov": _prov(1, 5, 15),
                },
            ]
        },
    }
    payload = _payload(body=_body("#/key_value_items/0"), key_value_items=[item])
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert [(unit["content_kind"], unit["contents"]) for unit in pages[1]] == [
        ("prose", "Term\nDefinition")
    ]
    assert diagnostics["key_value_item_count"] == 1


# --- content kinds, fallback order, image-only ---------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [("list_item", "list"), ("table", "table"), ("text", "prose")],
)
def test_content_kind_follows_the_label(label: str, expected: str) -> None:
    payload = _payload(
        body=_body("#/texts/0"),
        texts=[_text("#/texts/0", "kind", label=label)],
    )
    pages, _diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert pages[1][0]["content_kind"] == expected


def test_fallback_order_is_payload_texts_then_pictures_without_a_body() -> None:
    payload = _payload(
        texts=[_text("#/texts/0", "first"), _text("#/texts/1", "second")],
        pictures=[
            _picture("#/pictures/0", children=["#/texts/2"], captions=["#/texts/2"]),
        ],
    )
    payload["texts"].append(
        _text("#/texts/2", "caption", label="caption", parent="#/pictures/0")
    )
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert [(unit["content_kind"], unit["contents"]) for unit in pages[1]] == [
        ("prose", "first"),
        ("prose", "second"),
        ("figure", "caption"),
    ]
    assert diagnostics["unordered_item_count"] == 0


def test_captionless_picture_on_a_textless_page_makes_no_scan_claim() -> None:
    payload = _payload(
        body=_body("#/pictures/0"),
        pictures=[_picture("#/pictures/0", children=[], captions=[])],
    )
    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)
    assert pages == {}
    assert diagnostics["retained_item_count"] == 0
    assert "image_only_pages" not in diagnostics


def test_unsupported_item_is_counted_and_named() -> None:
    payload = _payload(
        tables=[
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "data": "opaque",
                "prov": [_prov(1, 0, 0)],
            }
        ]
    )

    _pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)

    assert diagnostics["unsupported_item_count"] == 1
    assert diagnostics["unsupported_table_item_count"] == 1
    assert diagnostics["unsupported_item_refs"] == [
        {"ref": "#/tables/0", "kind": "table"}
    ]


def test_unresolved_reference_is_counted() -> None:
    payload = {
        "texts": [
            {
                "self_ref": "#/texts/0",
                "content_layer": "body",
                "label": "text",
                "prov": [{"page_no": 1, "charspan": [0, 4]}],
                "text": "body",
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "children": [{"$ref": "#/texts/9"}],
                "captions": [{"$ref": "#/texts/8"}],
                "prov": [{"page_no": 1, "charspan": [0, 0]}],
            }
        ],
    }

    _pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)

    assert diagnostics["missing_child_ref_count"] == 1
    assert diagnostics["missing_caption_ref_count"] == 1


def test_a_complete_payload_keeps_clean_diagnostics() -> None:
    payload = _payload(
        body=_body("#/texts/0"),
        texts=[_text("#/texts/0", "Body evidence.")],
    )

    pages, diagnostics = adapt_document(payload, page_first=1, page_last=1)

    assert pages
    assert diagnostics["unsupported_item_count"] == 0
    assert diagnostics["missing_child_ref_count"] == 0
    assert diagnostics["missing_caption_ref_count"] == 0
    assert diagnostics["retained_item_count"] > 0
