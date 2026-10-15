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
