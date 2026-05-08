from pathlib import Path

from napari_spam._parsing import _scan_folder


def _write_tsv(path: Path) -> None:
    """Write a minimal TSV with required coordinate and value columns."""
    # Minimal grid with coordinate + value columns.
    content = "Zpos Ypos Xpos Xdisp\n0 0 0 1\n0 0 1 2\n0 1 0 3\n0 1 1 4\n"
    path.write_text(content, encoding="utf-8")


def test_scan_folder_collects_actions_and_fields(tmp_path: Path) -> None:
    """Scan a folder and collect actions, fields, and TSV columns.

    Checks that tif/tsv discovery enables the correct actions, extracts the
    expected field names from tif filenames, and reads TSV header columns.
    """
    # Create tif variants that share a prefix but differ in field suffixes.
    (tmp_path / "17-18-ldic-filtered-Xdisp.tif").write_text("", encoding="utf-8")
    (tmp_path / "17-18-ldic-filtered-Ydisp.tif").write_text("", encoding="utf-8")
    (tmp_path / "17-18-ldic-filtered-strain-dev.tif").write_text("", encoding="utf-8")
    # Add one TSV so the TSV action and columns are present.
    _write_tsv(tmp_path / "18-19-ldic-filtered-strain-Q8.tsv")

    scan = _scan_folder(str(tmp_path))

    # Both actions should be available.
    assert "Load tifs" in scan["actions"]
    assert "Load TSV files" in scan["actions"]
    # Field names are parsed from the tif suffix tokens.
    assert set(scan["tif_fields"]) == {"Xdisp", "Ydisp", "strain-dev"}
    # TSV columns come from the header line.
    assert scan["tsv_columns"] == ["Zpos", "Ypos", "Xpos", "Xdisp"]


def test_scan_folder_reports_pairs(tmp_path: Path) -> None:
    """Ensure scan collects actions even with mixed filename patterns.

    Confirms the presence of tif/tsv files still yields the expected actions.
    """
    # Single-index tif should not prevent action discovery.
    (tmp_path / "01.tif").write_text("", encoding="utf-8")
    # Mixed naming patterns should still be recognized.
    (tmp_path / "02-03-ldic-filtered-Xdisp.tif").write_text("", encoding="utf-8")
    _write_tsv(tmp_path / "02-03-ldic-filtered-strain-Q8.tsv")

    scan = _scan_folder(str(tmp_path))

    assert "Load tifs" in scan["actions"]
    assert "Load TSV files" in scan["actions"]
