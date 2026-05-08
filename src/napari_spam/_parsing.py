from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class ParsedTif:
    path: Path
    field: str = ""
    common_mid: str = ""


@dataclass(frozen=True)
class ParsedTsv:
    path: Path


_ACTION_TIFS = "Load tifs"
_ACTION_TSVS = "Load TSV files"
_ACTION_VTKS = "Load VTK files"


TOKEN_SEPARATORS = ["-", "_"]

_DIGIT_RE = re.compile(r"(\d+)")
_TOKEN_SPLIT_RE = re.compile("[" + re.escape("".join(TOKEN_SEPARATORS)) + "]")


def _natural_sort_key(text: str) -> list[object]:
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in _DIGIT_RE.split(text)
    ]


def _split_tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_SPLIT_RE.split(text) if token]


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
        "tsvs": parsed_tsvs,
        "tsv_columns": tsv_columns,
        "vtks": vtk_paths,
    }


def _parse_tif_paths(paths: Iterable[Path]) -> list[ParsedTif]:
    entries: list[tuple[Path, list[str]]] = []
    for path in paths:
        tokens = _split_tokens(path.stem)
        entries.append((path, tokens))

    if not entries:
        return []

    token_lists = [tokens for _, tokens in entries]
    common_prefix = _common_prefix(token_lists)

    parsed: list[ParsedTif] = []
    for path, tokens in entries:
        field_tokens = tokens[len(common_prefix) :]
        field = "-".join(field_tokens) if field_tokens else "image"
        common_mid = "-".join(common_prefix)
        parsed.append(
            ParsedTif(
                path=path,
                field=field,
                common_mid=common_mid,
            )
        )

    return sorted(parsed, key=lambda t: _natural_sort_key(t.path.name))


def _parse_tsv_paths(paths: Iterable[Path]) -> list[ParsedTsv]:
    parsed: list[ParsedTsv] = []
    for path in paths:
        parsed.append(ParsedTsv(path=path))

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
