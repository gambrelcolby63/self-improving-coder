"""Extract code blocks from model output."""
from __future__ import annotations

import re

FENCE = re.compile(r"```([a-zA-Z0-9_+\-]*)[ \t]*\n(.*?)```", re.S)
LANG_ALIASES = {"python": {"python", "py", "python3", ""},
                "cpp": {"cpp", "c++", "cc", "cxx", "c", ""},
                "verilog": {"verilog", "v", "systemverilog", "sv", ""}}


def extract_code(text: str, lang: str = "python") -> str:
    """Return the largest fenced block in `lang` (or untagged); fall back to raw text."""
    blocks = FENCE.findall(text)
    if not blocks:
        # unterminated fence
        m = re.search(r"```[a-zA-Z0-9_+\-]*[ \t]*\n(.*)", text, re.S)
        return (m.group(1) if m else text).strip() + "\n"
    ok = [b for tag, b in blocks if tag.lower() in LANG_ALIASES.get(lang, {""})]
    cand = ok or [b for _, b in blocks]
    return max(cand, key=len).strip() + "\n"


def strip_main_guard(code: str) -> str:
    """Drop `if __name__ == '__main__':` blocks so test harnesses don't run demo code."""
    lines = code.split("\n")
    out, skip = [], False
    for ln in lines:
        if re.match(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", ln):
            skip = True
            continue
        if skip and (ln.startswith((" ", "\t")) or not ln.strip()):
            continue
        skip = False
        out.append(ln)
    return "\n".join(out)
