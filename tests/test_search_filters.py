"""Search-level source selection and category partitions."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from conftest import write_pdf, write_reviewed_metadata
from test_service import FakeDenseBackend, FakeUltraRAG

from research_rag.config import resolve_config
from research_rag.service import ResearchError, ResearchService


async def _build_project(project: Path) -> tuple[ResearchService, dict[str, str]]:
    write_pdf(
        project / "sources" / "cobalt.pdf",
        ["Cobalt extraction labour evidence from the mine."],
    )
    write_pdf(
        project / "sources" / "waste.pdf",
        ["Electronic waste labour evidence from the landfill."],
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)
    listed = await service.list_sources()
    source_ids = {
        item["source_relative_path"]: item["source_id"] for item in listed["sources"]
    }
    assert set(source_ids) == {"cobalt.pdf", "waste.pdf"}
    return service, source_ids


def test_search_selects_and_excludes_sources_by_stable_id(project: Path) -> None:
    async def exercise() -> None:
        service, source_ids = await _build_project(project)
        cobalt = source_ids["cobalt.pdf"]
        waste = source_ids["waste.pdf"]

        everything = await service.search("labour evidence", top_k=10, rerank=False)
        assert {hit["source_path"] for hit in everything["hits"]} == {
            "sources/cobalt.pdf",
            "sources/waste.pdf",
        }
        assert everything["filters"]["source_ids"] == []
        assert everything["filters"]["exclude_source_ids"] == []
        assert everything["filters"]["active_document_count"] == 2

        included = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            source_ids=[cobalt],
        )
        assert {hit["source_path"] for hit in included["hits"]} == {
            "sources/cobalt.pdf"
        }
        assert included["filters"]["source_ids"] == [cobalt]
        assert included["filters"]["unknown_source_ids"] == []
        assert included["filters"]["active_document_count"] == 1

        excluded = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            exclude_source_ids=[waste],
        )
        assert {hit["source_path"] for hit in excluded["hits"]} == {
            "sources/cobalt.pdf"
        }
        assert excluded["filters"]["exclude_source_ids"] == [waste]
        assert excluded["filters"]["active_document_count"] == 1

    asyncio.run(exercise())


def test_search_reports_unknown_source_ids_and_refuses_an_empty_include(
    project: Path,
) -> None:
    async def exercise() -> None:
        service, source_ids = await _build_project(project)
        cobalt = source_ids["cobalt.pdf"]

        with pytest.raises(ResearchError, match="matched no document"):
            await service.search(
                "labour evidence",
                top_k=5,
                rerank=False,
                source_ids=["src_missing"],
            )

        mixed = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            source_ids=[cobalt, "src_missing"],
        )
        assert {hit["source_path"] for hit in mixed["hits"]} == {"sources/cobalt.pdf"}
        assert mixed["filters"]["unknown_source_ids"] == ["src_missing"]

        unknown_exclusion = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            exclude_source_ids=["src_missing"],
        )
        assert unknown_exclusion["filters"]["unknown_exclude_source_ids"] == [
            "src_missing"
        ]
        assert unknown_exclusion["filters"]["unknown_source_ids"] == []
        assert unknown_exclusion["filters"]["active_document_count"] == 2

    asyncio.run(exercise())


def test_reviewed_exclusion_wins_over_a_search_include(project: Path) -> None:
    async def exercise() -> None:
        service, source_ids = await _build_project(project)
        waste = source_ids["waste.pdf"]
        await service.set_source_inclusion(
            included=False,
            source_id=waste,
            reason="Reviewed duplicate for this test.",
        )
        assert (await service.status())["excluded_source_count"] == 1

        result = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            source_ids=[waste],
        )
        assert result["hits"] == []
        assert result["result_count"] == 0
        assert result["filters"]["active_document_count"] == 0
        assert result["filters"]["source_ids"] == [waste]

    asyncio.run(exercise())


def test_categories_partition_the_corpus_for_search_and_listing(project: Path) -> None:
    async def exercise() -> None:
        service, _source_ids = await _build_project(project)
        assert (await service.status())["categories"] == []

        write_reviewed_metadata(
            service.config,
            "cobalt.pdf",
            {"categories": ["Cobalt corpus"]},
        )
        write_reviewed_metadata(
            service.config,
            "waste.pdf",
            {"categories": ["Waste corpus"]},
        )

        assert (await service.status())["categories"] == [
            {"category": "Cobalt corpus", "searchable_source_count": 1},
            {"category": "Waste corpus", "searchable_source_count": 1},
        ]

        one_partition = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            categories_any=["cobalt corpus"],
        )
        assert {hit["source_path"] for hit in one_partition["hits"]} == {
            "sources/cobalt.pdf"
        }
        assert one_partition["filters"]["categories_any"] == ["cobalt corpus"]

        union = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            categories_any=["cobalt corpus", "waste corpus"],
        )
        assert {hit["source_path"] for hit in union["hits"]} == {
            "sources/cobalt.pdf",
            "sources/waste.pdf",
        }
        assert union["filters"]["categories_any"] == ["cobalt corpus", "waste corpus"]

        no_such_partition = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            categories_any=["no such corpus"],
        )
        assert no_such_partition["hits"] == []

        # The source inventory is the whole corpus; it takes no filters.
        listed = await service.list_sources()
        assert [item["source_path"] for item in listed["sources"]] == [
            "sources/cobalt.pdf",
            "sources/waste.pdf",
        ]

    asyncio.run(exercise())


def test_source_selection_combines_with_categories(project: Path) -> None:
    async def exercise() -> None:
        service, source_ids = await _build_project(project)
        write_reviewed_metadata(
            service.config,
            "cobalt.pdf",
            {"categories": ["Shared corpus"]},
        )
        write_reviewed_metadata(
            service.config,
            "waste.pdf",
            {"categories": ["Shared corpus"]},
        )

        selected = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            categories_any=["shared corpus"],
            source_ids=[source_ids["waste.pdf"]],
        )
        assert {hit["source_path"] for hit in selected["hits"]} == {"sources/waste.pdf"}
        assert selected["filters"]["active_document_count"] == 1
        assert selected["filters"]["categories_any"] == ["shared corpus"]
        assert selected["filters"]["source_ids"] == [source_ids["waste.pdf"]]

    asyncio.run(exercise())


def test_project_layer_selects_and_reports(project: Path) -> None:
    async def exercise() -> None:
        service, _source_ids = await _build_project(project)
        assert (await service.status())["projects"] == []

        write_reviewed_metadata(
            service.config,
            "cobalt.pdf",
            {"project": ["ai-and-fetishism"]},
        )
        write_reviewed_metadata(
            service.config,
            "waste.pdf",
            {"project": ["other-project"]},
        )

        assert (await service.status())["projects"] == [
            {"project": "ai-and-fetishism", "searchable_source_count": 1},
            {"project": "other-project", "searchable_source_count": 1},
        ]

        mine = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            projects_any=["ai-and-fetishism"],
        )
        assert {hit["source_path"] for hit in mine["hits"]} == {"sources/cobalt.pdf"}
        assert mine["filters"]["projects_any"] == ["ai-and-fetishism"]
        # The layer is returned with every hit and reference group, so a caller
        # can see why a passage was selected without a second lookup.
        assert [hit["project"] for hit in mine["hits"]] == [["ai-and-fetishism"]]

        union = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            projects_any=["ai-and-fetishism", "other-project"],
        )
        assert {hit["source_path"] for hit in union["hits"]} == {
            "sources/cobalt.pdf",
            "sources/waste.pdf",
        }

        untagged = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            projects_any=["unassigned"],
        )
        assert untagged["hits"] == []

    asyncio.run(exercise())


def test_three_metadata_layers_combine(project: Path) -> None:
    async def exercise() -> None:
        service, _source_ids = await _build_project(project)
        write_reviewed_metadata(
            service.config,
            "cobalt.pdf",
            {
                "project": ["ai-and-fetishism"],
                "categories": ["marxism"],
                "keywords": ["fetishism", "use value"],
            },
        )
        write_reviewed_metadata(
            service.config,
            "waste.pdf",
            {
                "project": ["ai-and-fetishism"],
                "categories": ["political ecology"],
                "keywords": ["waste"],
            },
        )

        matched = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            projects_any=["ai-and-fetishism"],
            categories_any=["marxism"],
            keywords=["fetishism"],
        )
        assert {hit["source_path"] for hit in matched["hits"]} == {"sources/cobalt.pdf"}
        filters = matched["filters"]
        assert filters["projects_any"] == ["ai-and-fetishism"]
        assert filters["categories_any"] == ["marxism"]
        assert filters["keywords_all"] == ["fetishism"]

        # A branch filter from the other layer still resolves independently.
        other = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            projects_any=["ai-and-fetishism"],
            categories_any=["marxism", "political ecology"],
            keywords=["waste"],
        )
        assert {hit["source_path"] for hit in other["hits"]} == {"sources/waste.pdf"}

    asyncio.run(exercise())


def test_authors_and_titles_filter_the_corpus_by_name(project: Path) -> None:
    """An author and a title narrow a corpus by the names they are, not as tags.

    Both are phrases, so they match as case-insensitive substrings of the values
    the query path already holds: a surname finds its author without the
    bibliography's punctuation, and a remembered title fragment finds the work
    without its subtitle. Reviewed values win over extracted ones, which makes
    the filter useful on a corpus whose extraction is wrong.
    """

    async def exercise() -> None:
        service, _source_ids = await _build_project(project)

        # Both test sources extract the same title and author, so an extracted
        # name cannot separate them and this passes on both.
        both = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            authors_any=["TEST AUTHOR"],
        )
        assert {hit["source_path"] for hit in both["hits"]} == {
            "sources/cobalt.pdf",
            "sources/waste.pdf",
        }
        assert both["filters"]["authors_any"] == ["test author"]
        assert both["filters"]["active_document_count"] == 2

        write_reviewed_metadata(
            service.config,
            "cobalt.pdf",
            {"title": "Cobalt extraction labour review", "authors": ["Dana Cobalt"]},
        )

        # A reviewed author replaces the extracted one rather than joining it.
        reviewed_author = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            authors_any=["dana"],
        )
        assert {hit["source_path"] for hit in reviewed_author["hits"]} == {
            "sources/cobalt.pdf"
        }
        assert reviewed_author["filters"]["active_document_count"] == 1

        # A name no source carries is a filter that matched nothing, not a corpus
        # that holds nothing.
        no_such_author = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            authors_any=["Nobody At All"],
        )
        assert no_such_author["hits"] == []
        assert no_such_author["filters"]["active_document_count"] == 0
        assert no_such_author["filters"]["authors_any"] == ["nobody at all"]

        # A title fragment is enough, and it is matched case-insensitively.
        fragment = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            titles_any=["EXTRACTION LABOUR"],
        )
        assert {hit["source_path"] for hit in fragment["hits"]} == {
            "sources/cobalt.pdf"
        }
        assert fragment["filters"]["titles_any"] == ["extraction labour"]

        # The extracted title is still what the other source carries.
        extracted_title = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            titles_any=["test pdf"],
        )
        assert {hit["source_path"] for hit in extracted_title["hits"]} == {
            "sources/waste.pdf"
        }

    asyncio.run(exercise())


def test_author_and_title_filters_combine_with_each_other(project: Path) -> None:
    """Every filter layer has to pass, and an author list is any-of."""

    async def exercise() -> None:
        service, _source_ids = await _build_project(project)
        write_reviewed_metadata(
            service.config,
            "cobalt.pdf",
            {"title": "Cobalt extraction", "authors": ["Dana Cobalt"]},
        )
        write_reviewed_metadata(
            service.config,
            "waste.pdf",
            {"title": "Waste frontiers", "authors": ["Wendy Waste"]},
        )

        either_author = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            authors_any=["cobalt", "waste"],
        )
        assert {hit["source_path"] for hit in either_author["hits"]} == {
            "sources/cobalt.pdf",
            "sources/waste.pdf",
        }

        one_work = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            authors_any=["cobalt", "waste"],
            titles_any=["frontiers"],
        )
        assert {hit["source_path"] for hit in one_work["hits"]} == {"sources/waste.pdf"}
        assert one_work["filters"]["authors_any"] == ["cobalt", "waste"]
        assert one_work["filters"]["titles_any"] == ["frontiers"]

        # An author that excludes the only title match returns nothing, which is
        # the two layers disagreeing rather than either one failing.
        disagreeing = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            authors_any=["dana"],
            titles_any=["frontiers"],
        )
        assert disagreeing["hits"] == []
        assert disagreeing["filters"]["active_document_count"] == 0

    asyncio.run(exercise())
