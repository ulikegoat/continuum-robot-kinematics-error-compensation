"""Check local Markdown links without launching the GUI or running experiments."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", *(ROOT / "docs").glob("*.md"), ROOT / "figures" / "gui" / "README.md"]
LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")


def main() -> int:
    broken = []
    count = 0
    for document in FILES:
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            local = unquote(target.split("#", 1)[0])
            if not local:
                continue
            count += 1
            if not (document.parent / local).exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    if broken:
        print("Missing local links:")
        print("\n".join(broken))
        return 1
    print(f"Checked {count} local documentation links; all targets exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
