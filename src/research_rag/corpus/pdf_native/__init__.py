"""Bundled pure-python PDF text reader, adapted from AhmedKishki/pdf-tools.

Text only: no image decoding, rendering, OCR, or external process. `objects`
parses the file, `content` reads positioned runs, `lines` groups them, and
`extract` opens a document and reads its pages.

Adapted from the user's own AhmedKishki/pdf-tools project; the code is reused
here with the owner's permission.
"""
