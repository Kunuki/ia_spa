"""
scripts/place_towers.py
-----------------------
Thin wrapper so the placement CLI is reachable in the same style as the other
scripts in this folder::

    python -m scripts.place_towers --radio-maps maps.npy --locations sites.csv -n 15

It is exactly the ``ia-spa`` console script; see ``ia_spa/cli.py``.
"""

from ia_spa.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
