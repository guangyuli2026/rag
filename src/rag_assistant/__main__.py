"""Allow ``python -m rag_assistant`` to run the same demo."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
