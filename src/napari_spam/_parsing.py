from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class TimeKey:
    kind: str
    a: int
    b: int | None = None


@dataclass(frozen=True)
class ParsedTif:
    path: Path
    suffix_tokens: tuple[str, ...]
    time_key: TimeKey
    field: str = ""
    common_mid: str = ""


@dataclass(frozen=True)
class ParsedTsv:
    path: Path
    time_key: TimeKey


_ACTION_TIFS = "Load tifs"
_ACTION_TSVS = "Load TSV files"
_ACTION_VTKS = "Load VTK files"


_DIGIT_RE = re.compile(r"(\d+)")


def _natural_sort_key(text: str) -> list[object]:
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in _DIGIT_RE.split(text)
    ]


def _is_int_token(token: str) -> bool:
    return token.isdigit()


def _parse_time_key_from_tokens(tokens: list[str]) -> tuple[TimeKey | None, int]:
    if not tokens:
        return None, 0
    if _is_int_token(tokens[0]):
        if len(tokens) > 1 and _is_int_token(tokens[1]):
            return TimeKey(kind="pair", a=int(tokens[0]), b=int(tokens[1])), 2
        return TimeKey(kind="single", a=int(tokens[0])), 1
    return None, 0


def _scan_folder(folder: str) -> dict:
    root = Path(folder)
    tif_paths = sorted(root.glob("*.tif"), key=lambda p: _natural_sort_key(p.name))
    tsv_paths = sorted(root.glob("*.tsv"), key=lambda p: _natural_sort_key(p.name))
    vtk_paths = sorted(root.glob("*.vtk"), key=lambda p: _natural_sort_key(p.name))

    parsed_tifs = _parse_tif_paths(tif_paths)
    parsed_tsvs = _parse_tsv_paths(tsv_paths)

    actions: list[str] = []
    if parsed_tifs:
        actions.append(_ACTION_TIFS)
    if parsed_tsvs:
        actions.append(_ACTION_TSVS)
    if vtk_paths:
        actions.append(_ACTION_VTKS)

    tsv_columns = []
    if parsed_tsvs:
        tsv_columns = _read_tsv_header(parsed_tsvs[0].path)

    return {
        "actions": actions,
        "tifs": parsed_tifs,
        "tif_fields": sorted({tif.field for tif in parsed_tifs}, key=_natural_sort_key),
        "tif_has_pairs": any(tif.time_key.kind == "pair" for tif in parsed_tifs),
        "tsvs": parsed_tsvs,
        "tsv_columns": tsv_columns,
        "tsv_has_pairs": any(tsv.time_key.kind == "pair" for tsv in parsed_tsvs),
        "vtks": vtk_paths,
    }


def _parse_tif_paths(paths: Iterable[Path]) -> list[ParsedTif]:
    staged: list[tuple[str, ParsedTif]] = []
    for path in paths:
        tokens = path.stem.split("-")
        time_key, consumed = _parse_time_key_from_tokens(tokens)
        if time_key is None:
            continue
        first_part = "-".join(tokens[:consumed])
        suffix_tokens = tuple(tokens[consumed:])
        staged.append(
            (
                first_part,
                ParsedTif(
                    path=path,
                    suffix_tokens=suffix_tokens,
                    time_key=time_key,
                ),
            )
        )

    grouped: dict[str, list[ParsedTif]] = {}
    for first_part, entry in staged:
        grouped.setdefault(first_part, []).append(entry)

    parsed: list[ParsedTif] = []
    for entries in grouped.values():
        suffix_lists = [list(item.suffix_tokens) for item in entries]
        common_prefix = _common_prefix(suffix_lists)
        for item in entries:
            field_tokens = item.suffix_tokens[len(common_prefix) :]
            field = "-".join(field_tokens) if field_tokens else "image"
            common_mid = "-".join(common_prefix)
            parsed.append(
                ParsedTif(
                    path=item.path,
                    suffix_tokens=item.suffix_tokens,
                    time_key=item.time_key,
                    field=field,
                    common_mid=common_mid,
                )
            )

    return sorted(parsed, key=lambda t: _natural_sort_key(t.path.name))


def _parse_tsv_paths(paths: Iterable[Path]) -> list[ParsedTsv]:
    parsed: list[ParsedTsv] = []
    for path in paths:
        tokens = path.stem.split("-")
        time_key, consumed = _parse_time_key_from_tokens(tokens)
        if time_key is None:
            continue
        parsed.append(ParsedTsv(path=path, time_key=time_key))

    return sorted(parsed, key=lambda t: _natural_sort_key(t.path.name))


def _common_prefix(token_lists: list[list[str]]) -> list[str]:
    if not token_lists:
        return []
    shortest = min(len(tokens) for tokens in token_lists)
    prefix: list[str] = []
    for idx in range(shortest):
        token = token_lists[0][idx]
        if all(tokens[idx] == token for tokens in token_lists[1:]):
            prefix.append(token)
        else:
            break
    return prefix


def _read_tsv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
    if not header:
        return []
    return header.split()
