"""The apparatus verdict: what a search may leave out, and what it must keep.

`apparatus_kind` reads one passage's own text. A reference list, citation-only
notes, a back-of-book index, and a copyright page are named by structure once the
passage has almost no argument grammar. Argument that cites heavily, a conclusion
running into its references, a table of measurements, and prose in a quoted
language keep their passage. Every case is a string, so no document is involved.
"""

from __future__ import annotations

import pytest

from research_rag.corpus.text_quality import (
    CHUNK_FLAG_APPARATUS,
    apparatus_kind,
    chunk_health_flags,
)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (
            (
                "Gidwani, Vinay. 2013. “Six Theses on Waste, Value, and Commons.” "
                "Social and Cultural Geography 14 (7): 773–83. Harvey, David. 2003. The "
                "New Imperialism. Oxford: Oxford University Press. Liboiron, Max, and "
                "Josh Lepawsky. 2022. Discard Studies: Wasting, Systems, and Power. "
                "Cambridge, MA: MIT Press. Moore, Jason W. 2015. Capitalism in the Web "
                "of Life. London: Verso."
            ),
            "bibliography",
        ),
        (
            (
                "12 Ibid., p. 45. 13 Marx, Capital, vol. 1, pp. 74–75. 14 See Harvey, "
                "Limits to Capital, pp. 190–95; Postone, Time, Labor, and Social "
                "Domination, p. 132. 15 Ibid., pp. 140–41. 16 Gidwani, “Six Theses,” "
                "p. 776."
            ),
            "bibliography",
        ),
        (
            (
                "capital, 12, 45–47, 89; accumulation of, 101–3, 140; fictitious, 77, "
                "212 commodity, 3, 15–18, 44; fetishism of, 19–22, 61, 233 labour power, "
                "27, 31–35, 90 value, 5, 8–11, 56–58, 140"
            ),
            "index",
        ),
        (
            (
                "First published by Verso. © David Harvey. All rights reserved. The "
                "moral rights of the author have been asserted. Typeset in Bembo. "
                "Printed in the United States by Maple Press."
            ),
            "front_matter",
        ),
    ],
)
def test_apparatus_is_named_by_its_structure(text: str, kind: str) -> None:
    assert apparatus_kind(text) == kind
    assert chunk_health_flags(text) & CHUNK_FLAG_APPARATUS


@pytest.mark.parametrize(
    "text",
    [
        # Argument cites in brackets, but its grammar is a sixth of its words.
        (
            "Waste is not a residue that capital leaves behind but a relation it "
            "produces, as Gidwani (2013) and Moore (2015) argue, because valorisation "
            "depends on what it can discard (Harvey 2003, 137). We therefore read the "
            "dump as a site where value is decided rather than lost (Liboiron and "
            "Lepawsky 2022, 4)."
        ),
        # Twenty words of argument sentences keep a passage that ends in references.
        (
            "This may lead us away from our currently distorted model of growth and "
            "accumulation, predicated on unsustainable throughputs of matter and "
            "energy and unevenly distributed burdens of entropy and toxicity. "
            "References Harvey, David. 2003. The New Imperialism. Oxford: Oxford "
            "University Press. Moore, Jason W. 2015. Capitalism in the Web of Life. "
            "London: Verso."
        ),
        # A table of measurements is data, whatever its years and labels.
        (
            "Year Exports 1848 Exports 1851 Exports 1860 Cotton yarn 135,831,162 "
            "143,966,106 197,343,655 Calico 1,091,373,930 1,543,161,789 2,776,218,427 "
            "Share 12.5% 14.8% 16.2% Change −1.6 −2.4 3.1"
        ),
        # Prose in a quoted language keeps its grammar words in the count.
        (
            "Die Ware ist zunächst ein äußerer Gegenstand, ein Ding, das durch seine "
            "Eigenschaften menschliche Bedürfnisse irgendeiner Art befriedigt (Marx "
            "1867, S. 49). Die Natur dieser Bedürfnisse, ob sie zum Beispiel dem Magen "
            "oder der Phantasie entspringen, ändert nichts an der Sache (Heinrich 2004, "
            "S. 37)."
        ),
    ],
)
def test_evidence_keeps_its_passage(text: str) -> None:
    assert apparatus_kind(text) is None
    assert not chunk_health_flags(text) & CHUNK_FLAG_APPARATUS


@pytest.mark.parametrize(
    ("text", "context", "kind"),
    [
        ("emergence 97–8", {"section_kind": "index"}, "index"),
        ("emergence 17,142,144", {"section_kind": "index"}, "index"),
        ("emergence 97–8", {}, None),
        ("emergence 17,142,144", {}, None),
        (
            "Emergence",
            {"docling_label": "section_header", "whole_unit": True},
            "heading",
        ),
        (
            "§2 Emergence",
            {"docling_label": "section_header", "whole_unit": True},
            "heading",
        ),
        (
            "Stratification and Emergence",
            {"docling_label": "section_header", "whole_unit": True},
            "heading",
        ),
        ("Emergence", {}, None),
        ("Emergence", {"docling_label": "section_header", "whole_unit": False}, None),
        (
            "Emergence is irreducible",
            {"docling_label": "section_header", "whole_unit": True},
            None,
        ),
        (
            "Markets Destroy Communities",
            {"docling_label": "section_header", "whole_unit": True},
            None,
        ),
        (
            "Emergence entails novelty",
            {"docling_label": "section_header", "whole_unit": True},
            None,
        ),
        (
            "Unknown words",
            {"docling_label": "section_header", "whole_unit": True},
            None,
        ),
        (
            "emergence 17,142,144",
            {"section_kind": "index", "docling_label": "table"},
            None,
        ),
        (
            "emergence 17,142,144",
            {"section_kind": "index", "content_kind": "table"},
            None,
        ),
        ("E = 17,142,144", {"section_kind": "index"}, None),
        ("Emergence increases at 97–8 degrees", {"section_kind": "index"}, None),
    ],
)
def test_contextual_fragments_need_structural_proof(
    text: str, context: dict, kind: str | None
) -> None:
    assert apparatus_kind(text, context=context) == kind
    assert bool(
        chunk_health_flags(text, apparatus_context=context) & CHUNK_FLAG_APPARATUS
    ) == (kind is not None)


@pytest.mark.parametrize(
    "label", ["caption", "formula", "code", "table", "unknown_future_label"]
)
def test_structural_data_is_not_apparatus(label: str) -> None:
    assert (
        apparatus_kind(
            "Growth, 17, 142, 144",
            context={"docling_label": label, "section_kind": "index"},
        )
        is None
    )


def test_split_author_and_year_is_a_retrieval_verdict_not_text_cleanup() -> None:
    text = (
        "Bhaskar, Roy\n\n"
        "1975. A Realist Theory of Science: Arguments on Ontology, Causality, "
        "Explanation, Naturalism, Philosophy, Scientific Practice, Methodology, "
        "Transcendental Realism, Experiments, Structures, Mechanisms, Dispositions, "
        "Powers, Tendencies, Systems, Stratification, Emergence, Knowledge, "
        "Epistemology, and Social Relations. London: University Press."
    )
    assert apparatus_kind(text) == "bibliography"
    assert (
        apparatus_kind(text.replace("University Press", "Routledge")) == "bibliography"
    )
    # Short explanatory prose and a claim-bearing note do not need twenty words.
    for prose in (
        "Emergence is not reducible to its components.",
        "1 Bhaskar (1975) argues that emergence is irreducible.",
        "Bhaskar, Roy\n\n1975. This claim is not supported by the evidence. University Press.",
    ):
        assert apparatus_kind(prose) is None
