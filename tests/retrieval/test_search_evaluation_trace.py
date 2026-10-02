"""The evaluation trace a search can be asked for, and the counts it makes checkable.

Every other test in this folder reads an answer. These read the pipeline behind
it: the depth the branch asked for, the reranked window, the cosine gate's own
arithmetic, and the per-stage identifier lists a measurement needs to say where a
judged passage went. The trace is opt-in, bounded, and carries no passage text,
and the tests here hold all three.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from research_rag.core.service import ResearchService
from research_rag.core.tool_views import present_tool_response
from research_rag.project.config import resolve_config
from research_rag.project.settings import LEAN_TOOL_DETAIL
from research_rag.retrieval.search import EVALUATION_TRACE_CANDIDATE_BUDGET
from research_rag.storage.records import read_jsonl, write_jsonl
from tests.conftest import write_pdf
from tests.core.test_service import (
    FakeDenseBackend,
    FakeUltraRAG,
    UnavailableRerankerDenseBackend,
)

#: The stages a trace names, in the order the pipeline reaches them.
STAGES = (
    "dense_before_filters",
    "dense_eligible",
    "dense_admitted",
    "bm25_after_gates",
    "fused_pre_rerank",
    "reranked",
    "post_collapse",
    "final",
)


class ManyPassageUltraRAG(FakeUltraRAG):
    """A chunker that emits one passage per index, so a depth can be observed.

    Each passage differs in words, not only in a digit: the engine collapses
    passages whose words are the same, and a corpus of passages that differ only
    by their index collapses to one of them before any stage can be read.
    """

    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        assert chunk_size > chunk_overlap
        self.chunk_calls += 1
        unit = read_jsonl(input_path)[0]
        write_jsonl(
            output_path,
            (
                {
                    "id": index,
                    "doc_id": unit["id"],
                    "title": unit["title"],
                    "contents": (
                        f"Cobalt evidence passage {index} about subject{index} "
                        f"and theme{index} of the archive."
                    ),
                }
                for index in range(self.count)
            ),
        )


async def _corpus(
    project: Path,
    count: int,
    *,
    settings_overrides: list[str] | None = None,
    dense: FakeDenseBackend | None = None,
    ultra: FakeUltraRAG | None = None,
) -> ResearchService:
    write_pdf(
        project / "sources" / "cobalt.pdf",
        ["A cobalt corpus page."],
        title="Cobalt",
    )
    config = resolve_config(
        project,
        vanilla_executable=sys.executable,
        settings_overrides=settings_overrides or [],
    )
    service = ResearchService(  # type: ignore[arg-type]
        config,
        ultra if ultra is not None else ManyPassageUltraRAG(count),
        dense=dense if dense is not None else FakeDenseBackend(),
    )
    await service.ingest(chunk_size=100, chunk_overlap=10)
    return service


def _generation_chunks(service: ResearchService) -> list[dict]:
    generation_root, manifest = service._load_current_optional()  # type: ignore[attr-defined]
    return read_jsonl(generation_root / str(manifest["files"]["chunks"]))


def _counts(answer: dict) -> dict[str, int | None]:
    return {
        name: None if stage is None else stage["count"]
        for name, stage in answer["evaluation_trace"]["stages"].items()
    }


def test_the_trace_is_absent_unless_a_measurement_asks_for_it(project: Path) -> None:
    async def exercise() -> None:
        service = await _corpus(project, 12)
        ordinary = await service.search(
            "cobalt evidence",
            top_k=5,
            retrieval_method="bm25",
            rerank=False,
        )
        traced = await service.search(
            "cobalt evidence",
            top_k=5,
            retrieval_method="bm25",
            rerank=False,
            evaluation_trace=True,
        )
        assert "evaluation_trace" not in ordinary
        assert traced["evaluation_trace"]["retrieval_method"] == "bm25"

    asyncio.run(exercise())


def test_the_trace_names_each_stage_with_its_own_count(project: Path) -> None:
    async def exercise() -> None:
        service = await _corpus(project, 12)
        answer = await service.search(
            "cobalt evidence",
            top_k=5,
            retrieval_method="bm25",
            rerank=False,
            evaluation_trace=True,
        )
        trace = answer["evaluation_trace"]
        assert tuple(trace["stages"]) == STAGES
        for name, stage in trace["stages"].items():
            # Two stages are null here, for two different reasons: the dense
            # half this method never opens, and the reranked window it never
            # asks for. Neither is an empty stage.
            if stage is None:
                assert name in {
                    "dense_before_filters",
                    "dense_eligible",
                    "dense_admitted",
                    "reranked",
                }
                continue
            # The count is the stage's own size and the list is a bounded copy
            # of it, so a cut stage still says what it held and says it was cut.
            assert len(stage["chunk_ids"]) == min(
                stage["count"], EVALUATION_TRACE_CANDIDATE_BUDGET
            ), name
            assert stage["truncated"] == (
                stage["count"] > EVALUATION_TRACE_CANDIDATE_BUDGET
            ), name
        # Twelve chunks, a twelve-candidate window, and five answers selected
        # from it: a stage before the selection holds candidates, not answers.
        assert answer["candidate_depth"] == 12
        assert trace["candidate_budget"] == EVALUATION_TRACE_CANDIDATE_BUDGET
        assert trace["stages"]["bm25_after_gates"]["count"] == 12
        assert trace["stages"]["fused_pre_rerank"]["count"] == 12
        assert trace["stages"]["post_collapse"]["count"] == 12
        assert trace["stages"]["final"]["chunk_ids"] == [
            hit["chunk_id"] for hit in answer["hits"]
        ]

    asyncio.run(exercise())


def test_the_trace_carries_identifiers_and_no_passage_text(project: Path) -> None:
    async def exercise() -> None:
        service = await _corpus(project, 12)
        answer = await service.search(
            "cobalt evidence",
            top_k=5,
            rerank=False,
            evaluation_trace=True,
        )
        rendered = str(answer["evaluation_trace"])
        # A passage's text belongs in the passage. A trace carrying it would be a
        # second copy of the result list, sized by the depth rather than by k.
        for chunk in _generation_chunks(service):
            assert str(chunk.get("contents")) not in rendered

    asyncio.run(exercise())


def test_the_cosine_gate_reports_a_denominator_and_conserves_its_candidates(
    project: Path,
) -> None:
    async def exercise() -> None:
        service = await _corpus(project, 30)
        answer = await service.search(
            "cobalt evidence",
            top_k=5,
            retrieval_method="hybrid",
            rerank=False,
            evaluation_trace=True,
        )
        gate = answer["dense_gate"]
        assert gate["eligible_total"] is not None
        assert gate["conserved"] is True
        assert gate["eligible_total"] == (
            gate["admitted_above_floor"]
            + gate["admitted_below_floor"]
            + gate["rejected_below_floor"]
        )
        # What the gate saw, what it admitted, and what it removed are three
        # separate numbers, and the share removed is a division on the first.
        dense = answer["evaluation_trace"]["dense"]
        assert dense["eligible_total"] == gate["eligible_total"]
        assert dense["conserved"] is True
        assert (
            dense["excluded_before_gate"]["returned_by_index"]
            >= (gate["eligible_total"])
        )
        # The gate's own rejections are counted here and in the per-reason block,
        # so the two cannot disagree about what the gate removed.
        assert (
            answer["rejected_candidates"]["dense_below_threshold"]
            == (gate["rejected_below_floor"])
        )

    asyncio.run(exercise())


def test_a_bm25_search_reports_no_dense_gate_counts_rather_than_zeros(
    project: Path,
) -> None:
    async def exercise() -> None:
        service = await _corpus(project, 30)
        answer = await service.search(
            "cobalt evidence",
            top_k=5,
            retrieval_method="bm25",
            rerank=False,
        )
        gate = answer["dense_gate"]
        # The block is present for every method, so its presence says nothing.
        # What is absent is every count: a gate that never opened did not admit
        # zero candidates, it admitted none, and those are different statements.
        assert gate["eligible_total"] is None
        assert gate["admitted_above_floor"] is None
        assert gate["admitted_below_floor"] is None
        assert gate["rejected_below_floor"] is None
        assert gate["conserved"] is None
        assert gate["excluded_before_gate"] is None
        assert answer["rejected_candidates"]["dense_below_threshold"] == 0
        assert (
            answer["evaluation_trace" if False else "dense_gate"]["conserved"] is None
        )

    asyncio.run(exercise())


def test_the_branch_depth_follows_the_candidate_formula(project: Path) -> None:
    """min(active chunks, maximum candidates, max(minimum candidates, top_k * 4))."""

    async def exercise() -> None:
        service = await _corpus(project, 200)
        for top_k, expected in ((10, 40), (20, 80), (40, 160)):
            answer = await service.search(
                "cobalt evidence",
                top_k=top_k,
                retrieval_method="hybrid",
                rerank=False,
                evaluation_trace=True,
            )
            assert answer["candidate_depth"] == expected, top_k
            trace = answer["evaluation_trace"]
            assert trace["candidate_depth"] == expected, top_k
            assert trace["stages"]["dense_before_filters"]["count"] == expected, top_k
            assert trace["stages"]["dense_admitted"]["count"] == expected, top_k

    asyncio.run(exercise())


def test_the_reranked_window_follows_the_window_formula(project: Path) -> None:
    """min(candidates ranked, rerank_max_candidates, max(top_k * multiple, floor))."""

    async def exercise() -> None:
        service = await _corpus(project, 200)
        for top_k, expected in ((10, 20), (15, 30), (25, 50)):
            answer = await service.search(
                "cobalt evidence",
                top_k=top_k,
                retrieval_method="hybrid",
                rerank=True,
                evaluation_trace=True,
            )
            assert answer["rerank_window"] == expected, top_k
            rerank = answer["evaluation_trace"]["rerank"]
            assert rerank["requested"] is True
            assert rerank["applied"] is True
            assert rerank["window"] == expected, top_k
            assert rerank["scored_count"] == expected, top_k
            assert rerank["scored_truncated"] is False
            assert len(rerank["scored_ids"]) == expected, top_k
            assert answer["evaluation_trace"]["stages"]["reranked"]["count"] == expected
            # The window is a prefix of the fused order that entered it, so a
            # target outside it never reached the cross-encoder.
            fused = answer["evaluation_trace"]["stages"]["fused_pre_rerank"][
                "chunk_ids"
            ]
            assert (
                answer["evaluation_trace"]["stages"]["reranked"]["chunk_ids"]
                == fused[:expected]
            )

    asyncio.run(exercise())


def test_a_reranker_that_cannot_load_is_traced_as_not_applied(project: Path) -> None:
    async def exercise() -> None:
        service = await _corpus(
            project,
            10,
            dense=UnavailableRerankerDenseBackend(),
        )
        answer = await service.search(
            "cobalt evidence",
            top_k=3,
            retrieval_method="hybrid",
            rerank=True,
            evaluation_trace=True,
        )

        assert answer["rerank_requested"] is True
        assert answer["reranked"] is False
        assert answer["rerank_fallback"]["reason"] == "reranker_model_unavailable"
        rerank = answer["evaluation_trace"]["rerank"]
        assert rerank["requested"] is True
        assert rerank["applied"] is False
        assert rerank["fallback"] == answer["rerank_fallback"]
        assert rerank["window"] == answer["rerank_window"]
        # The window is named and nothing is: a candidate no model read is not a
        # scored one, and a trace saying otherwise would let a measurement treat
        # an unranked fusion as something the cross-encoder had judged.
        assert rerank["scored_ids"] == []
        assert rerank["scored_count"] == 0
        assert rerank["scored_truncated"] is False
        # The window was chosen and the candidates were in it; only the scores
        # never arrived, so the order is the one that entered it.
        trace = answer["evaluation_trace"]
        window = answer["rerank_window"]
        assert (
            trace["stages"]["reranked"]["chunk_ids"]
            == trace["stages"]["fused_pre_rerank"]["chunk_ids"][:window]
        )

    asyncio.run(exercise())


def test_an_ordinary_search_never_gains_a_trace_key_in_a_lean_answer(
    project: Path,
) -> None:
    async def exercise() -> None:
        service = await _corpus(project, 12)
        answer = await service.search(
            "cobalt evidence",
            top_k=5,
            rerank=False,
            evaluation_trace=True,
        )
        lean = present_tool_response("search", answer, detail=LEAN_TOOL_DETAIL)
        assert "evaluation_trace" in answer
        assert "evaluation_trace" not in lean
        # Nothing else about the answer changes: the trace is added beside it.
        assert [hit["chunk_id"] for hit in lean["hits"]] == [
            hit["chunk_id"] for hit in answer["hits"]
        ]

    asyncio.run(exercise())


def test_a_measured_stage_list_is_bounded_and_says_it_was_cut(project: Path) -> None:
    async def exercise() -> None:
        service = await _corpus(
            project,
            400,
            settings_overrides=[
                "retrieval.minimum_candidates=400",
                "retrieval.maximum_candidates=400",
            ],
        )
        answer = await service.search(
            "cobalt evidence",
            top_k=10,
            retrieval_method="bm25",
            rerank=False,
            evaluation_trace=True,
        )
        stage = answer["evaluation_trace"]["stages"]["post_collapse"]
        assert stage["count"] == 400
        assert len(stage["chunk_ids"]) == EVALUATION_TRACE_CANDIDATE_BUDGET
        assert stage["truncated"] is True
        final = answer["evaluation_trace"]["stages"]["final"]
        assert final["count"] == 10
        assert final["truncated"] is False

    asyncio.run(exercise())


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_every_method_traces_the_stages_its_own_branch_opens(
    project: Path, method: str
) -> None:
    async def exercise() -> None:
        service = await _corpus(project, 20)
        answer = await service.search(
            "cobalt evidence",
            top_k=5,
            retrieval_method=method,
            rerank=False,
            evaluation_trace=True,
        )
        counts = _counts(answer)
        assert (counts["bm25_after_gates"] is not None) == (method != "dense")
        assert (counts["dense_admitted"] is not None) == (method != "bm25")
        # Each branch considered the same twenty chunks, and each admitted all
        # of them: the trace says where a candidate was, before and after fusion.
        assert counts["fused_pre_rerank"] == counts["post_collapse"] == 20
        assert counts["final"] == 5

    asyncio.run(exercise())


def test_a_plain_chunker_still_produces_a_complete_trace(project: Path) -> None:
    """The trace does not depend on the corpus shape the other tests arrange."""

    async def exercise() -> None:
        write_pdf(
            project / "sources" / "article.pdf",
            [
                "Cobalt evidence about labour and artificial intelligence.",
                "Quartz material unrelated to the primary question.",
            ],
            title="Research Article",
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        service = ResearchService(  # type: ignore[arg-type]
            config,
            FakeUltraRAG(),
            dense=FakeDenseBackend(),
        )
        await service.ingest(chunk_size=50, chunk_overlap=10)
        answer = await service.search(
            "cobalt labour",
            top_k=3,
            retrieval_method="hybrid",
            rerank=True,
            evaluation_trace=True,
        )
        counts = _counts(answer)
        # Both passages reached the dense half and only one cleared its floor:
        # the passage without a query term scores zero, and a BM25 window keeps
        # one for the same reason. The counts are separate, and both agree here.
        assert counts["dense_before_filters"] == 2
        assert counts["dense_admitted"] == 1
        assert counts["bm25_after_gates"] == 1
        assert counts["final"] == answer["result_count"] == 1
        assert answer["evaluation_trace"]["rerank"]["applied"] is True
        # One passage matched the query and one did not, and the trace names which
        # side of the lexical gate each candidate fell on.
        bm25 = answer["evaluation_trace"]["bm25"]
        assert bm25["rejected"]["no_query_token_overlap"] == 1

    asyncio.run(exercise())


def test_the_trace_changes_nothing_else_about_the_answer(project: Path) -> None:
    """Ranking and admission must be the same answer either way, key for key.

    A trace is a measurement of the pipeline, so the pipeline cannot notice it was
    asked for. Two searches over the same corpus, the same query, and a
    deterministic backend differ by the trace and by nothing else.
    """

    async def exercise() -> None:
        service = await _corpus(project, 40)
        for method in ("bm25", "dense", "hybrid"):
            plain = await service.search(
                "cobalt evidence",
                top_k=5,
                retrieval_method=method,
                rerank=False,
            )
            traced = await service.search(
                "cobalt evidence",
                top_k=5,
                retrieval_method=method,
                rerank=False,
                evaluation_trace=True,
            )
            without = {
                key: value for key, value in traced.items() if key != "evaluation_trace"
            }
            assert without == plain, method

    asyncio.run(exercise())
