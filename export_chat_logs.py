"""CLI: export all chat logs to Excel (admin)."""

from __future__ import annotations

import argparse
import sys

from chat_log import export_xlsx
from config import CHAT_LOG_EXPORT_PATH


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Export RAG chat logs to xlsx")
    from pathlib import Path

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=CHAT_LOG_EXPORT_PATH,
    )
    args = parser.parse_args()
    count = export_xlsx(args.output)
    print(f"Exported {count} records -> {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
