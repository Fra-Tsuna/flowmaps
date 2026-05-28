import json
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Any, Iterable, List, Sequence, Union


@dataclass(frozen=True)
class LookupTables:
    env_name: str
    pickupable2id: Dict[str, int]
    id2pickupable: Dict[int, str]
    receptacle2id: Dict[str, int]
    id2receptacle: Dict[int, str]
    pickupable_to_receptacle: Dict[str, Any]
    receptacle_colors: Dict[int, Tuple[float, float, float]]
    pickupable_colors: Dict[int, Tuple[float, float, float]]


_ACTIVE_LOOKUP: Optional[LookupTables] = None


def _load_json_preserve_order(path: Path):
    # Preserves key order for JSON objects exactly as in the file
    with open(path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def build_bev_palettes(id2receptacle, id2pickupable):
    """
    Returns:
        background_color: (3,) tuple in [0,1]
        receptacle_colors: dict[int, (3,)]
        pickupable_colors: dict[int, (3,)]
    """
    from colorsys import hsv_to_rgb

    receptacle_ids = sorted(id2receptacle.keys())
    pickupable_ids = sorted(id2pickupable.keys())

    def make_palette(ids, hue_offset: float, value: float):
        n = max(len(ids), 1)
        colors = {}
        for i, k in enumerate(ids):
            # Evenly spaced hues, shifted by hue_offset
            h = (hue_offset + i / n) % 1.0
            s = 0.9
            v = value
            r, g, b = hsv_to_rgb(h, s, v)
            colors[k] = (float(r), float(g), float(b))   # in [0, 1]
        return colors

    # Receptacles: bright colors
    receptacle_colors = make_palette(receptacle_ids, hue_offset=0.0,  value=0.9)

    # Objects: shifted hue + slightly darker value so they are visually distinct
    pickupable_colors = make_palette(pickupable_ids, hue_offset=0.25, value=0.7)

    return receptacle_colors, pickupable_colors


def _normalize_env_names(env_name: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(env_name, (str, Path)):
        return [str(env_name)]
    if isinstance(env_name, Iterable):
        return [str(name) for name in env_name]
    return [str(env_name)]


def _all_env_names(data_root: Path) -> List[str]:
    return sorted(p.name for p in data_root.iterdir() if p.is_dir() and p.name.startswith("env"))


def _resolve_env_names(env_name: Optional[Union[str, Sequence[str]]], data_root: Path) -> List[str]:
    if env_name is None:
        matches = _all_env_names(data_root)
        if not matches:
            raise ValueError(f"No env directories found under {data_root}.")
        return matches

    names = _normalize_env_names(env_name)
    resolved: List[str] = []
    seen = set()

    for name in names:
        direct_path = data_root / name
        if direct_path.exists() and direct_path.is_dir():
            if name not in seen:
                resolved.append(name)
                seen.add(name)
            continue

        matches = _all_env_names(data_root)
        if not matches:
            raise ValueError(f"No env directories found under {data_root}.")
        for match in matches:
            if match not in seen:
                resolved.append(match)
                seen.add(match)

    return resolved


def resolve_env_names(
    env_name: Optional[Union[str, Sequence[str]]],
    data_root: Optional[Path] = None,
) -> List[str]:
    if data_root is None:
        data_root = Path(__file__).resolve().parents[2] / "data"
    else:
        data_root = Path(data_root)
    return _resolve_env_names(env_name, data_root)


def load_lookup(env_name: Optional[Union[str, Sequence[str]]] = None, data_root: Optional[Path] = None) -> LookupTables:
    if data_root is None:
        data_root = Path(__file__).resolve().parents[2] / "data"
    else:
        data_root = Path(data_root)

    env_names = _resolve_env_names(env_name, data_root)

    pickupable_names: List[str] = []
    receptacle_names: List[str] = []
    pickupable_seen = set()
    receptacle_seen = set()

    # Geometry is env-specific; the global lookup only stores IDs and labels.
    pickupable_to_receptacle: "OrderedDict[str, OrderedDict[str, None]]" = OrderedDict()

    for env in env_names:
        data_dir = data_root / env
        pickupables_json = data_dir / "pickupable_names.json"
        receptacles_json = data_dir / "receptacle_names.json"
        pickupable_to_receptacle_json = data_dir / "pickupable_to_receptacle.json"

        env_pickupable_names = _load_json_preserve_order(pickupables_json)
        env_receptacle_names = _load_json_preserve_order(receptacles_json)
        env_pickupable_to_receptacle = _load_json_preserve_order(pickupable_to_receptacle_json)

        for name in env_pickupable_names:
            category = name.split("|")[0]
            if category not in pickupable_seen:
                pickupable_seen.add(category)
                pickupable_names.append(category)

        for name in env_receptacle_names:
            category = name.split("|")[0]
            if category not in receptacle_seen:
                receptacle_seen.add(category)
                receptacle_names.append(category)

        for pickupable, receptacles in env_pickupable_to_receptacle.items():
            pickupable_cat = pickupable.split("|")[0]
            if pickupable_cat not in pickupable_seen:
                pickupable_seen.add(pickupable_cat)
                pickupable_names.append(pickupable_cat)
            bucket = pickupable_to_receptacle.setdefault(pickupable_cat, OrderedDict())
            for receptacle in receptacles:
                receptacle_cat = receptacle.split("|")[0]
                if receptacle_cat not in receptacle_seen:
                    receptacle_seen.add(receptacle_cat)
                    receptacle_names.append(receptacle_cat)
                if receptacle_cat not in bucket:
                    bucket[receptacle_cat] = None

    pickupable_to_receptacle = OrderedDict(
        (k, list(v.keys())) for k, v in pickupable_to_receptacle.items()
    )

    pickupable2id = {name: i for i, name in enumerate(pickupable_names)}
    id2pickupable = {i: name for name, i in pickupable2id.items()}

    receptacle2id = {name: i for i, name in enumerate(receptacle_names)}
    id2receptacle = {i: name for name, i in receptacle2id.items()}

    receptacle_colors, pickupable_colors = build_bev_palettes(
        id2receptacle,
        id2pickupable,
    )

    display_name = "all" if env_name is None else "+".join(_normalize_env_names(env_name))
    merged_env_name = env_names[0] if len(env_names) == 1 else display_name

    return LookupTables(
        env_name=merged_env_name,
        pickupable2id=pickupable2id,
        id2pickupable=id2pickupable,
        receptacle2id=receptacle2id,
        id2receptacle=id2receptacle,
        pickupable_to_receptacle=pickupable_to_receptacle,
        receptacle_colors=receptacle_colors,
        pickupable_colors=pickupable_colors,
    )


def set_active_lookup(lookup: LookupTables) -> None:
    global _ACTIVE_LOOKUP
    _ACTIVE_LOOKUP = lookup


def get_active_lookup() -> LookupTables:
    if _ACTIVE_LOOKUP is None:
        raise RuntimeError(
            "LookupTables are not initialized. Call load_lookup() and "
            "set_active_lookup() early (e.g., in run.py) before using lookup data."
        )
    return _ACTIVE_LOOKUP


__all__ = [
    "LookupTables",
    "build_bev_palettes",
    "load_lookup",
    "resolve_env_names",
    "set_active_lookup",
    "get_active_lookup",
]
