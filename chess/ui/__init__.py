"""UI package providing frontend assets, templates, and loaders."""

import importlib.resources
from functools import cache


@cache
def get_html() -> str:
    """Load and return the self-contained HTML document for the web UI."""
    ref = importlib.resources.files("chess.ui")
    html = ref.joinpath("index.html").read_text(encoding="utf-8")
    css = ref.joinpath("style.css").read_text(encoding="utf-8")
    js = ref.joinpath("app.js").read_text(encoding="utf-8")

    html = html.replace("/* __INJECT_CSS__ */", css)
    return html.replace("// __INJECT_JS__", js)


__all__ = ["get_html"]
