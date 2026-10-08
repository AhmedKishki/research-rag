"""Group positioned chunks into lines and detect page structure.

The middle of text extraction: `text.content` produces positioned runs, and
this module decides which of them are lines, in what order they are read, and
which of the small ones are markers riding on a neighbour rather than text of
their own.

    grouped = group_lines(chunks)              # runs sharing a baseline
    ordered = order_page_lines(grouped, width)  # reading order, columns split
    text = merge_line_marked(grouped[0].chunks)

Positions are measured in PDF points from `core.geometry`, which is also where
`chunk_advance` and `chunk_bounds` are defined; both are re-exported here
because a caller working with lines almost always needs them too.

Adapted from the user's own AhmedKishki/pdf-tools project; the code is reused
here with the owner's permission.
"""

from bisect import bisect_left
from collections import Counter
from itertools import pairwise

from .geometry import chunk_advance, chunk_bounds

__all__ = [
    "SUP_MARK",
    "Line",
    "body_size",
    "chunk_advance",
    "chunk_bounds",
    "detect_column_gutter",
    "detect_column_x",
    "fragments",
    "group_lines",
    "mark_inline",
    "merge_line_marked",
    "merge_line_text",
    "order_page_lines",
    "split_at_x",
    "split_column_chunks",
]


def _within_line_key(c):
    """Reading order inside one baseline.

    Stream order wins over x. Kerning can place a chunk slightly left of the
    one before it -- `[(V)111(erstehen)]` is a V followed by 'erstehen' with a
    -111/1000 em kern -- and a right-aligned folio can be drawn before the
    entry it belongs to, so sorting by x transposes both and corrupts the text.
    """
    return (getattr(c, "order", 0), c.x)


def _gap(prev, c):
    """Gap from the drawn end of `prev` to the start of `c`, in points.

    Measured along `prev`'s baseline (`ux`, `uy`) -- the direction its own
    advance was measured in -- so the answer stays meaningful for a run set
    vertically, where an x-only difference collapses to the left edge.
    """
    return (
        (c.x - prev.x) * getattr(prev, "ux", 1.0)
        + (c.y - prev.y) * getattr(prev, "uy", 0.0)
        - chunk_advance(prev)
    )


class Line:
    __slots__ = ("chunks", "fonts", "main", "page", "size", "text", "x", "x1", "y")

    def __init__(self, chunks, page, main=None):
        self.chunks = chunks
        self.page = page
        self.main = main if main is not None else list(chunks)
        self.x = min(chunk_bounds(c)[0] for c in chunks)
        self.x1 = max(chunk_bounds(c)[2] for c in chunks)
        self.y = min(c.y for c in chunks)
        sizes = [c.size for c in self.main if c.size > 0 and c.text.strip()]
        self.size = max(sizes) if sizes else 0.0
        self.fonts = [c.font for c in self.main]
        self.text = ""


SUPERSCRIPT_MAX = 7.6
"""Largest rendered size that may still be read as an inline run.

An absolute ceiling, not a body-text threshold. Which runs are raised or
lowered is decided relatively (see `mark_inline`); this only stops the test
from promoting something plainly set at display size. 7.6 pt cannot serve as a
floor: it sits inside the ordinary range of a fine-set journal, whose running
text measures 5.2 pt, and a fixed threshold there reads half the page as
footnote markers.
"""

SUPERSCRIPT_MIN = 3.0
"""Below this a run is too small to be inline, whatever it sits next to."""

SUPERSCRIPT_RATIO = 0.8
"""An inline run is at most this share of the size of the run it rides on."""

SUPERSCRIPT_RISE = 0.45
"""How far, as a share of that run's size, its baseline may be offset.

The same distance separates a superscript from its text and the second row of a
wrapped heading from the first, so the offset alone decides nothing; it is the
size ratio that tells the two apart, and the sign that says which way.
"""

SPACE_EM = 0.20
"""Gap between two runs, in em, above which a word boundary is assumed.

Measured on both samples. In `book.pdf` a word break is a literal space in the
text and the geometry either side of it is ~0.00 em, so this never fires; in
`paper.pdf` there are no space characters at all and the producer sets words
with a positive `TJ` adjustment, which lands at +0.42 em against +0.00 em for
the kern-only boundaries inside a word. The margin is wide in both directions,
so the exact value is not delicate; 0.20 em is roughly where a space starts.
"""

BASELINE_SLACK = 9.0
FRAGMENT_GAP = 11.0  # em-ish gap that separates fragments on one baseline
ATTACH_SLACK = 2.0  # how far outside a line's extent an inline run may sit


def body_size(chunks):
    """The size most of a page's glyphs are set in.

    Weighted by glyph count, so a two-word 24 pt heading cannot outweigh a page
    of running text. Ties go to the larger size.
    """
    tally = Counter()
    for c in chunks:
        if c.size > 0:
            tally[round(c.size, 2)] += len(c.text.strip()) or 1
    if not tally:
        return 0.0
    top = max(tally.values())
    return max(s for s, n in tally.items() if n >= top * 0.5)


def mark_inline(chunks):
    """Flag the runs that ride on another run's baseline rather than their own.

    A superscript is not "small", it is small *next to the text it sits in*: a
    run whose size is a fraction of a nearby larger run, set within about a
    third of an em of that run's baseline, and horizontally adjacent to it.
    Every part of that is load-bearing. Size alone reads the whole of a
    fine-set page as markers -- a 5.2 pt affiliation line below a 9 pt abstract
    is small, but it stands on its own baseline and is ordinary text.
    Baseline alone reads the second row of a wrapped title as a marker, because
    a line break drops the baseline by about a third of an em, which is exactly
    how a superscript is raised; only the size test tells the two apart.
    Horizontal adjacency is what keeps a small run in the line whose text it
    annotates rather than in the line above it.

    Sets `Chunk.sup` for a raised run and `Chunk.sub` for a lowered one, and
    returns the pairs. `group_lines` calls this once per page and the merge
    helpers read the flags, so a run is classified once.

    Both directions matter. Note references are set above the baseline
    (`the latter already exist.1`), but symbol indices are set below it
    (`conditions C1 and C2`); treating the second as a footnote reference would
    link a table note to a sentence that has nothing to do with it.
    """
    hosts = sorted(
        (c for c in chunks if c.size > SUPERSCRIPT_MIN and c.text.strip()),
        key=lambda c: c.y,
    )
    ys = [c.y for c in hosts]
    tallest = max((c.size for c in hosts), default=0.0)
    # A generous window is safe because the test inside it is not: `rise` is
    # measured against the particular host, so a display heading far above
    # cannot lend its scale to a body-sized run below it.
    reach = SUPERSCRIPT_RISE * tallest
    out = []
    for c in chunks:
        c.sup = c.sub = False
        if not c.text.strip():
            continue
        if not (SUPERSCRIPT_MIN <= c.size <= SUPERSCRIPT_MAX):
            continue
        lo = bisect_left(ys, c.y - reach)
        hi = bisect_left(ys, c.y + reach)
        for h in hosts[lo:hi]:
            rise = c.y - h.y
            if abs(rise) > SUPERSCRIPT_RISE * h.size or rise == 0:
                continue
            if c.size > SUPERSCRIPT_RATIO * h.size:
                continue
            if h.x - 0.5 * h.size <= c.x <= h.x + chunk_advance(h) + h.size:
                c.sup, c.sub = rise > 0, rise < 0
                out.append((c, c.sup))
                break
    return out


def fragments(chunks, gap=None):
    """Split a line's chunks into spatially separate runs.

    One baseline often carries more than one thing -- a journal's bottom line
    mixes repeating furniture ('Vol', 'ISSN', the publisher) with values that
    change every page (the DOI). Fingerprinting whole lines misses the
    furniture, so candidates are judged per fragment.
    """
    if not chunks:
        return []
    if gap is None:
        gap = FRAGMENT_GAP * (max(c.size for c in chunks) or 10.0) / 10.0
    out, cur = [], [chunks[0]]
    for prev, c in pairwise(chunks):
        if _gap(prev, c) > gap:
            out.append(cur)
            cur = [c]
        else:
            cur.append(c)
    out.append(cur)
    return out


def detect_column_x(lines, page_w=432.0, min_frac=0.42, max_frac=0.80, min_share=0.10):
    """Second-column start x for a two-column page, else None.

    Uses chunk start positions: a merged baseline still contains a chunk that
    begins exactly at the column boundary.

    The deciding test is how many *lines* begin there. Chunk starts cluster
    somewhere on most pages -- at a figure's left edge, at a stray label -- and a
    cluster is not a column. Measured on the two samples: the sample book's
    index pages put 29-33% of their lines at the boundary, while every page that
    only looked two-column put 0-2% there, among them the paper's page 11, where a
    chart's labels sit at x=366 and the detector cut the copyright line that runs
    the full width of the page in half.
    """
    if len(lines) < 8:
        return None
    xs = []
    for L in lines:
        for c in L.chunks:
            if page_w * min_frac <= c.x <= page_w * max_frac:
                xs.append(round(c.x))
    if not xs:
        return None
    counts = Counter(xs)
    x, n = counts.most_common(1)[0]
    if n < max(5, len(lines) * min_share):
        return None
    x = column_origin(x, counts)
    # Is there a first column, and does its text stop short of the boundary?
    # Measured on the part of each line that lies left of the boundary, which is
    # the whole line when it is already inside the first column. Counting whole
    # lines only is what loses the index's first page: its two columns share their
    # baselines exactly, so every line carries both, no line is wholly inside the
    # first column, and the page is rejected as single-column -- leaving thirty-odd
    # index entries with a headword from each column welded to the other.
    first = [L for L in lines if L.x < x - 1 and left_part_ends(L, x)]
    if len(first) < max(4, len(lines) * 0.30):
        return None
    ends = [chunk_bounds(c)[2] for L in first for c in L.chunks if c.x < x - 1]
    if sum(ends) / len(ends) > x - 8:
        return None
    right = sum(1 for L in lines if L.x >= x - 1)
    if right < max(4, len(lines) * 0.12):
        return None
    return x


COLUMN_CLUSTER = 24.0
"""How far right of a column's origin its hanging indent may sit.

A column begins where its flush lines begin; the continuations of those lines hang
a few points to the right of it. Where an entry is mostly locators the hanging
indent is the more frequent start -- the sample's index sets entries flush at 222
and their locators at 234 -- and taking that for the origin would put every flush
entry on the wrong side of the split. Only a start of comparable standing can be
the origin, so one stray label in the same neighbourhood cannot move it.
"""


def column_origin(x, counts, share=0.4):
    """Left edge of the column whose most frequent start is `x`."""
    group = [v for v in counts if x - COLUMN_CLUSTER <= v <= x]
    candidates = [v for v in group if counts[v] >= counts[x] * share]
    return min(candidates) if candidates else x


def left_part_ends(line, bx, slack=8.0):
    """Does the text of `line` that lies left of `bx` stop short of it?

    On a page of justified prose the answer is no: every line runs the full width
    of the measure, so whatever part of it falls left of a proposed boundary
    reaches right up to that boundary. On the left half of a two-column page the
    answer is yes, and that is the whole test.
    """
    ends = [chunk_bounds(c)[2] for c in line.chunks if c.x < bx - 1]
    return bool(ends) and sum(ends) / len(ends) <= bx - slack


def split_at_x(line, bx):
    cs = sorted(line.chunks, key=_within_line_key)
    left = [c for c in cs if c.x < bx - 1]
    right = [c for c in cs if c.x >= bx - 1]
    if not left or not right:
        return [line]
    out = []
    for grp in (left, right):
        nl = Line(grp, line.page, [c for c in grp if not c.sup] or grp)
        nl.text = merge_line_text(nl.chunks)
        out.append(nl)
    return out


def order_page_lines(lines, page_w=432.0):
    """Column-major reading order when the page has two columns."""
    bx = detect_column_x(lines, page_w)
    if bx is None:
        return sorted(lines, key=lambda L: (-round(L.y, 2), L.x))
    # column split first, then reading order inside each column
    expanded = []
    for L in lines:
        expanded.extend(split_at_x(L, bx))
    left = [L for L in expanded if L.x < bx - 1]
    right = [L for L in expanded if L.x >= bx - 1]
    left.sort(key=lambda L: (-round(L.y, 2), L.x))
    right.sort(key=lambda L: (-round(L.y, 2), L.x))
    return left + right


COLUMN_GUTTER = 12.0
"""Smallest empty vertical band, in points, that separates two columns.

An ordinary inter-word space is a few points; a column gutter is a band no run
crosses. The floor keeps a word gap from reading as a column boundary.
"""

COLUMN_SPAN = 0.75
"""A run at least this share of the page width spans both columns.

A full-width heading or rule legitimately crosses the gutter. Ignoring runs that
wide lets a spanning element sit above the columns without hiding the gutter
between the body lines below it.
"""


def detect_column_gutter(chunks, page_w=432.0, min_share=0.25):
    """Second-column boundary from an empty vertical band between runs, else None.

    Two columns that share their baselines are welded into one line by
    `group_lines` before any line-start cluster can see them, so `detect_column_x`
    refuses them. The gutter between the columns is still a band no run crosses,
    and this finds it: runs are sorted by left edge and the widest vertical gap
    that leaves a share of the runs on each side is the boundary. Full-width runs
    are left out of the search so a spanning heading does not close the gutter.
    """

    if not chunks or page_w <= 0:
        return None
    spanning = page_w * COLUMN_SPAN
    body = sorted(
        (
            bounds
            for chunk in chunks
            if chunk.text.strip()
            for bounds in (chunk_bounds(chunk),)
            if 0.0 < bounds[2] - bounds[0] < spanning
        ),
        key=lambda bounds: bounds[0],
    )
    if len(body) < 4:
        return None
    best_boundary = None
    best_gap = COLUMN_GUTTER
    for index in range(2, len(body) - 1):
        left_end = max(bounds[2] for bounds in body[:index])
        right_start = min(bounds[0] for bounds in body[index:])
        gap = right_start - left_end
        if gap < best_gap:
            continue
        left_share = index / len(body)
        if min(left_share, 1.0 - left_share) < min_share:
            continue
        boundary = (left_end + right_start) / 2.0
        if not page_w * 0.25 <= boundary <= page_w * 0.75:
            continue
        best_gap = gap
        best_boundary = boundary
    return best_boundary


def split_column_chunks(chunks, boundary):
    """Partition runs into the columns left and right of a boundary."""

    left = [chunk for chunk in chunks if chunk_bounds(chunk)[0] < boundary - 1]
    right = [chunk for chunk in chunks if chunk_bounds(chunk)[0] >= boundary - 1]
    return left, right


def group_lines(chunks, ytol=2.6, page_w=None):
    """Group chunks sharing a baseline, reattaching raised and lowered runs."""
    if not chunks:
        return []
    vertical = [c for c in chunks if abs(getattr(c, "uy", 0.0)) > 0.7]
    if vertical:
        horizontal = [c for c in chunks if abs(getattr(c, "uy", 0.0)) <= 0.7]
        lines = group_lines(horizontal, ytol, page_w)
        groups = []
        for c in sorted(vertical, key=_within_line_key):
            c.sup = c.sub = False
            target = next(
                (g for g in groups if abs(c.x - g[0].x) <= ytol and c.uy * g[0].uy > 0),
                None,
            )
            if target is None:
                groups.append([c])
            else:
                target.append(c)
        lines.extend(Line(g, g[0].page) for g in groups)
        return sorted(lines, key=lambda L: (-L.y, L.x))
    inline = mark_inline(chunks)
    main = [c for c in chunks if not (c.sup or c.sub)]
    sup = [c for c, _raised in inline]

    lines = []
    if main:
        cur, cur_y = [], None
        # walk baselines top-down on the raw y; rounding here would split a
        # single baseline whose chunks differ in the last decimal place
        for c in sorted(main, key=lambda c: -c.y):
            if cur and abs(c.y - cur_y) <= ytol:
                cur.append(c)
                cur_y = sum(x.y for x in cur) / len(cur)
            else:
                if cur:
                    lines.append(Line(sorted(cur, key=_within_line_key), cur[0].page))
                cur, cur_y = [c], c.y
        if cur:
            lines.append(Line(sorted(cur, key=_within_line_key), cur[0].page))

    for s in sup:
        # A raised run annotates the line *below* it and a lowered one the line
        # above; taking whichever line happens to be near, without the
        # direction, attaches every note reference to the preceding line of
        # text -- which reads plausibly and is wrong, because the sentence it
        # belongs to ends on the far side of the line break.
        target, best = None, None
        for L in lines:
            if not L.x - ATTACH_SLACK <= s.x <= L.x1 + ATTACH_SLACK:
                continue
            d = (s.y - L.y) if s.sup else (L.y - s.y)
            if 0 <= d <= BASELINE_SLACK and (best is None or d < best):
                target, best = L, d
        if target is not None:
            target.chunks.append(s)
            target.chunks.sort(key=_within_line_key)
            target.x = min(target.x, s.x)
            if not any(s.font is f for f in target.fonts):
                target.fonts.append(s.font)
        else:
            lines.append(Line([s], s.page))
    if sup:
        lines.sort(key=lambda L: (-round(L.y, 2), L.x))
    return lines


def _is_joinable(a, b):
    if not a or not b:
        return True
    if a == " " or b == " ":
        return True
    return b in ",.;:!?)%" or a in "([{/"


SUP_MARK = "\ue000"
"""Placeholder written where a superscript run was removed from the line.

Footnote markers are set in a smaller font on a raised baseline, so they are
re-attached to their line rather than read as separate text. Merging them
silently glues the marker to the word it follows ('problems28'); marking the
position instead lets the EPUB builder emit a real <sup>.
"""


def _word_boundary(prev, c):
    """Does a space belong between these two runs?

    Two independent checks, and both are needed. A run that already carries a
    space decides its own boundary, and a boundary with punctuation against it
    is a join however far apart the two boxes are. What is left is decided
    geometrically: the gap between the drawn right edge of `prev` and the left
    edge of `c`, as a fraction of the text size.

    Neither kind of inline run is preceded by a space. `merge_line_marked`
    turns a raised one into SUP_MARK and a lowered one into the symbol's own
    subscript (`C1`), and both belong inside the token they annotate.
    """
    a = prev.text[-1:] if prev.text else ""
    b = c.text[:1] if c.text else ""
    if c.sup or c.sub or _is_joinable(a, b):
        return False
    if not (prev.text.strip() and c.text.strip()):
        return False
    ref = max(prev.size, c.size)
    if ref <= 0:
        return False
    return _gap(prev, c) > SPACE_EM * ref


def _merge(chunks, marked):
    out = []
    prev = None
    for c in chunks:
        if prev is not None:
            if marked and c.sup:
                out.append(SUP_MARK)
            elif _word_boundary(prev, c):
                out.append(" ")
        out.append(c.text)
        prev = c
    return "".join(out)


def merge_line_marked(chunks):
    """As merge_line_text, but flags superscript runs with SUP_MARK."""
    return _merge(chunks, True)


def merge_line_text(chunks):
    """Join chunk texts, inserting a space only where the geometry implies one.

    The gap is measured against the drawn extent of the previous run rather than
    against its nominal advance, because a producer that splits a word across
    `TJ` items places the second part by a kern that the advance already
    accounts for. Advances themselves are as reliable as the font's `Widths`;
    the *difference* between what a run is supposed to occupy and where the next
    one starts is what survives a subset font with a sparse width array.
    """
    return _merge(chunks, False)
