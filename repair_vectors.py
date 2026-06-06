"""Rebuild missing Chroma embedding vectors from sqlite chunk text."""

from __future__ import annotations

import sys

from chroma_utils import repair_vectors_from_sqlite, vector_index_healthy, ensure_collection_readable


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client, collection = ensure_collection_readable()
    if vector_index_healthy(collection):
        print(f"Vector index OK ({collection.count()} chunks). Nothing to do.")
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        return 0

    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass

    count = repair_vectors_from_sqlite()
    print(f"Rebuilt vectors for {count} chunks.")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
