import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1] / "images_tt"
def load(split="tune", kinds=None, overhead=None):
    """Manifest rows. split='locked' is refused until analysis/FROZEN exists."""
    if split != "tune" and not (pathlib.Path(__file__).parent / "FROZEN").exists():
        raise PermissionError("locked split is sealed until analysis/FROZEN exists")
    rows = [r for r in json.loads((ROOT / "manifest.json").read_text()) if not r["dup"] and (split == "all" or r["split"] == split)]
    return [r for r in rows if (kinds is None or r["kind"] in kinds) and (overhead is None or r["tag_overhead"] == overhead)]
