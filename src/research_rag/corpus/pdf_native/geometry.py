"""Page geometry: matrices, positioned-run extents and page sizes.

Everything here works on *positions*, not on text, and nothing here imports the
extractor. That is what lets the cleaner ask how wide a run is drawn without
depending on how the run was produced, and it keeps the dependency arrows
pointing one way: `core` knows about coordinates, `text` knows about words.

A positioned run is duck-typed. Anything with `x`, `y`, `size`, `width`,
`scale` and optionally `ux`/`uy` works, so a caller may pass the extractor's
`Chunk` or a stand-in of its own.

Adapted from the user's own AhmedKishki/pdf-tools project; the code is reused
here with the owner's permission.
"""

DEFAULT_PAGE_SIZE = (612.0, 792.0)
"""US Letter, PDF points. Used when a page carries no usable /MediaBox."""


def mat_mul(a, b):
    """Concatenate two affine matrices given as `(a, b, c, d, e, f)` tuples.

    `mat_mul(a, b)` is the matrix that applies `b` and then `a`, which is the
    order PDF 32000-1 8.3.3 defines for the `cm` operator: a new matrix
    multiplies on the *left* of the CTM.
    """
    return (
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    )


def chunk_advance(chunk):
    """Drawn width of a positioned run, in page units.

    `Chunk.width` is the run's advance *along its own baseline* in text-space
    units and `Chunk.scale` carries those units into page space, so the drawn
    length is their product. The floor keeps a zero-width run from collapsing
    to nothing; a run with no measurable extent has no extent to speak of, and
    this is what the callers already assumed.
    """
    return chunk.width * max(chunk.scale, 1e-6)


def chunk_bounds(chunk):
    """Font-size rectangle projected along the run's baseline, in PDF points.

    The rectangle is the run's own box rotated onto its baseline direction
    (`ux`, `uy`), not an axis-aligned guess from its length, so a run set a
    quarter turn round -- a margin sidebar, a rotated drop cap -- reports the
    box it actually occupies.
    """
    ux, uy = getattr(chunk, "ux", 1.0), getattr(chunk, "uy", 0.0)
    dx, dy = chunk_advance(chunk) * ux, chunk_advance(chunk) * uy
    points = [
        (chunk.x, chunk.y),
        (chunk.x + dx, chunk.y + dy),
        (chunk.x - uy * chunk.size, chunk.y + ux * chunk.size),
        (chunk.x + dx - uy * chunk.size, chunk.y + dy + ux * chunk.size),
    ]
    return (
        min(x for x, y in points),
        min(y for x, y in points),
        max(x for x, y in points),
        max(y for x, y in points),
    )


def chunk_centre(chunk):
    """Centre of `chunk_bounds`, in PDF points."""
    x0, y0, x1, y1 = chunk_bounds(chunk)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def point_in_rect(x, y, rect):
    """Is `(x, y)` inside `rect` `(x0, y0, x1, y1)`? Edges are inclusive.

    Coordinates are PDF user space: origin bottom left, y increasing upward.
    """
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def chunk_in_rect(chunk, rect):
    """Does a positioned run's centre fall inside `rect`?

    Centre, not overlap: a rectangle given on the command line is meant to
    select a region of the page -- a sidebar column, a chart's plot area -- and
    a long word that merely crosses its edge is not part of it. Testing the
    projected bounds (`chunk_bounds`) rather than the run's x position is what
    makes this work for rotated text, whose x-advance is close to zero.
    """
    x, y = chunk_centre(chunk)
    return point_in_rect(x, y, rect)


def page_size(doc, page, default=DEFAULT_PAGE_SIZE):
    """MediaBox of one page in points, as `(width, height)`.

    Normalised so the answer is a size and not a box: a reversed or
    unparsable /MediaBox yields the same width and height rather than a
    negative one, and a page with no /MediaBox inherits the caller's default
    instead of silently becoming Letter.
    """
    box = doc.resolve(page.get("MediaBox"))
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return default
    try:
        x0, y0, x1, y1 = (float(doc.resolve(v)) for v in box[:4])
    except (TypeError, ValueError):
        return default
    return abs(x1 - x0), abs(y1 - y0)
