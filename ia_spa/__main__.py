"""Allow ``python -m ia_spa`` to run the placement CLI."""

from ia_spa.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
