"""Moves the inline <script> blocks of the built documentation into external files.

The site that serves the docs (fannar.fernandocsdm.com) sends ``script-src 'self'``, which blocks
the few inline scripts Material for MkDocs writes in every page. Each distinct block becomes
``assets/inline/<hash>.js``, loaded by a plain (synchronous) <script src>, so the order and the
behaviour are unchanged. JSON blocks (``type="application/json"``) are data and stay inline.

    python scripts/externalize_inline_scripts.py build/docs
"""

import hashlib
import os
import re
import sys
from pathlib import Path

INLINE = re.compile(r"<script(?P<attrs>(?![^>]*\bsrc=)[^>]*)>(?P<body>.*?)</script>", re.S)


def main(site: Path) -> None:
    out = site / "assets" / "inline"
    out.mkdir(parents=True, exist_ok=True)
    written, pages = set(), 0
    for page in site.rglob("*.html"):
        html = page.read_text(encoding="utf-8")

        def replace(m: re.Match) -> str:
            attrs, body = m.group("attrs"), m.group("body")
            if "json" in attrs or not body.strip():
                return m.group(0)
            name = hashlib.sha256(body.encode()).hexdigest()[:16] + ".js"
            if name not in written:
                (out / name).write_text(body, encoding="utf-8")
                written.add(name)
            rel = os.path.relpath(out / name, page.parent).replace(os.sep, "/")
            return f'<script{attrs} src="{rel}"></script>'

        new = INLINE.sub(replace, html)
        if new != html:
            page.write_text(new, encoding="utf-8")
            pages += 1
    print(f"externalized inline scripts in {pages} pages into {len(written)} files")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "build/docs"))
