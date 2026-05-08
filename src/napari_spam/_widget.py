from importlib import resources

import numpy as np
import tifffile
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from pathlib import Path


from napari.viewer import Viewer

from napari_spam._parsing import (
    _ACTION_TIFS,
    _ACTION_TSVS,
    _ACTION_VTKS,
    _scan_folder,
)

SPAM_DOC_URL = "https://www.spam-project.dev/docs/"
SPAM_PAPER_URL = "https://joss.theoj.org/papers/10.21105/joss.02286"


def _read_tsv_data(path: Path) -> np.ndarray:
    data = np.genfromtxt(
        path,
        delimiter=None,
        names=True,
        dtype=float,
        invalid_raise=False,
    )
    return np.atleast_1d(data)


def _field_dims_from_coords(data: np.ndarray) -> np.ndarray:
    z_vals = np.unique(data["Zpos"])
    y_vals = np.unique(data["Ypos"])
    x_vals = np.unique(data["Xpos"])
    return np.array([len(z_vals), len(y_vals), len(x_vals)], dtype=int)


def _deduce_spatial_scale_and_translate(
    data: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    z_vals = np.unique(data["Zpos"])
    y_vals = np.unique(data["Ypos"])
    x_vals = np.unique(data["Xpos"])

    step = _mean_step(z_vals) or _mean_step(y_vals) or _mean_step(x_vals) or 1.0
    translate = (float(z_vals.min()), float(y_vals.min()), float(x_vals.min()))
    return (step, step, step), translate


def _mean_step(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    diffs = np.diff(np.sort(values))
    if diffs.size == 0:
        return None
    return float(np.mean(diffs))


class SpamLoaderWidget(QWidget):
    def __init__(self, viewer: Viewer):  # "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._scan: dict | None = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        self._folder_line = QLineEdit()
        self._folder_button = QPushButton("Browse")

        header_row = self._build_header()

        folder_row = QHBoxLayout()
        folder_row.addWidget(self._folder_line)
        folder_row.addWidget(self._folder_button)

        self._action_combo = QComboBox()
        self._action_combo.addItem("Select a folder first")

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_tif_panel())
        self._stack.addWidget(self._build_tsv_panel())
        self._stack.addWidget(self._build_vtk_panel())

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)

        self._time_base = QSpinBox()
        self._time_base.setRange(0, 10_000)
        self._time_base.setValue(0)
        self._load_button = QPushButton("Load")

        self._options_panel = QWidget()
        options_layout = QVBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.addWidget(self._action_combo)
        options_layout.addWidget(self._stack)

        shared_form = QFormLayout()
        shared_form.addRow("First timepoint", self._time_base)
        options_layout.addLayout(shared_form)
        options_layout.addWidget(self._load_button)
        options_layout.addWidget(self._status_label)
        self._options_panel.setLayout(options_layout)
        self._options_panel.setVisible(False)

        layout = QVBoxLayout()
        layout.addLayout(header_row)
        layout.addLayout(folder_row)
        layout.addWidget(self._options_panel)
        layout.addStretch(1)
        self.setLayout(layout)

    def _build_header(self) -> QHBoxLayout:
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        logo_ref = resources.files("napari_spam").joinpath("assets/napari-spam.png")
        with resources.as_file(logo_ref) as logo_path:
            if logo_path.exists():
                pixmap = QPixmap(str(logo_path))
                if not pixmap.isNull():
                    logo_label.setPixmap(
                        pixmap.scaledToHeight(48, Qt.SmoothTransformation)
                    )

        links_layout = QVBoxLayout()
        links_layout.setContentsMargins(0, 0, 0, 0)

        doc_label = QLabel(f'<a href="{SPAM_DOC_URL}">Spam doc</a>')
        doc_label.setTextFormat(Qt.RichText)
        doc_label.setOpenExternalLinks(True)

        spacer_label = QLabel(" ")

        paper_label = QLabel(f'<a href="{SPAM_PAPER_URL}">Spam paper</a>')
        paper_label.setTextFormat(Qt.RichText)
        paper_label.setOpenExternalLinks(True)

        links_layout.addWidget(spacer_label)
        links_layout.addWidget(doc_label)
        links_layout.addWidget(paper_label)
        links_layout.addWidget(spacer_label)

        header_row.addWidget(logo_label)
        header_row.addLayout(links_layout)
        header_row.addStretch(1)
        return header_row

    def _build_tif_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout()

        self._tif_field_combo = QComboBox()
        self._tif_scale = QDoubleSpinBox()
        self._tif_scale.setRange(0.001, 1000.0)
        self._tif_scale.setDecimals(4)
        self._tif_scale.setValue(1.0)
        form.addRow("Field", self._tif_field_combo)
        tif_scale_label = QLabel("Scale (pix<sup>-1</sup>)")
        tif_scale_label.setTextFormat(Qt.RichText)
        form.addRow(tif_scale_label, self._tif_scale)

        box = QGroupBox("TIF options")
        box.setLayout(form)

        layout = QVBoxLayout()
        layout.addWidget(box)
        panel.setLayout(layout)
        return panel

    def _build_tsv_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout()

        self._tsv_column_combo = QComboBox()
        self._tsv_deduce_scale = QCheckBox("Deduce scale from file")
        self._tsv_scale = QDoubleSpinBox()
        self._tsv_scale.setRange(0.001, 1000.0)
        self._tsv_scale.setDecimals(4)
        self._tsv_scale.setValue(1.0)
        form.addRow("Column", self._tsv_column_combo)
        form.addRow(self._tsv_deduce_scale)
        tsv_scale_label = QLabel("Scale (pix<sup>-1</sup>)")
        tsv_scale_label.setTextFormat(Qt.RichText)
        form.addRow(tsv_scale_label, self._tsv_scale)

        box = QGroupBox("TSV options")
        box.setLayout(form)

        layout = QVBoxLayout()
        layout.addWidget(box)
        panel.setLayout(layout)
        return panel

    def _build_vtk_panel(self) -> QWidget:
        panel = QWidget()
        label = QLabel("VTK loading is not implemented yet.")
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout = QVBoxLayout()
        layout.addWidget(label)
        panel.setLayout(layout)
        return panel

    def _connect_signals(self) -> None:
        self._folder_button.clicked.connect(self._browse_folder)
        self._folder_line.editingFinished.connect(self._on_folder_text)
        self._action_combo.currentIndexChanged.connect(self._on_action_change)
        self._load_button.clicked.connect(self._load_current)
        self._tsv_deduce_scale.stateChanged.connect(self._on_tsv_scale_mode)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            self._folder_line.setText(folder)
            self._scan_folder(folder)

    def _on_folder_text(self) -> None:
        folder = self._folder_line.text().strip()
        if folder:
            self._scan_folder(folder)

    def _scan_folder(self, folder: str) -> None:
        try:
            self._scan = _scan_folder(folder)
        except Exception as exc:  # noqa: BLE001
            self._scan = None
            self._status_label.setText(f"Failed to scan folder: {exc}")
            self._options_panel.setVisible(False)
            return

        self._status_label.setText("")
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        self._action_combo.blockSignals(True)
        self._action_combo.clear()
        if self._scan is None or not self._scan["actions"]:
            self._action_combo.addItem("No supported files")
            self._action_combo.setEnabled(False)
            self._action_combo.blockSignals(False)
            self._options_panel.setVisible(False)
            return

        for action in self._scan["actions"]:
            self._action_combo.addItem(action)
        self._action_combo.setEnabled(True)
        self._action_combo.blockSignals(False)
        self._options_panel.setVisible(True)
        self._on_action_change(self._action_combo.currentIndex())

    def _on_action_change(self, index: int) -> None:
        if self._scan is None:
            return
        action = self._action_combo.currentText()
        if action == _ACTION_TIFS:
            self._stack.setCurrentIndex(0)
            self._refresh_tif_fields()
        elif action == _ACTION_TSVS:
            self._stack.setCurrentIndex(1)
            self._refresh_tsv_columns()
        elif action == _ACTION_VTKS:
            self._stack.setCurrentIndex(2)
        else:
            self._stack.setCurrentIndex(0)

    def _refresh_tif_fields(self) -> None:
        if self._scan is None:
            return
        fields = self._scan["tif_fields"]
        self._tif_field_combo.clear()
        self._tif_field_combo.addItems(fields)

    def _refresh_tsv_columns(self) -> None:
        if self._scan is None:
            return
        columns = self._scan["tsv_columns"]
        self._tsv_column_combo.clear()
        self._tsv_column_combo.addItems(columns)

    def _load_current(self) -> None:
        action = self._action_combo.currentText()
        if action == _ACTION_TIFS:
            self._load_tifs()
        elif action == _ACTION_TSVS:
            self._load_tsvs()
        elif action == _ACTION_VTKS:
            self._status_label.setText("VTK loading is not implemented yet.")

    def _on_tsv_scale_mode(self) -> None:
        self._tsv_scale.setEnabled(not self._tsv_deduce_scale.isChecked())

    def _load_tifs(self) -> None:
        if self._scan is None:
            return
        field = self._tif_field_combo.currentText()
        if not field:
            return

        items = [tif for tif in self._scan["tifs"] if tif.field == field]
        if not items:
            self._status_label.setText("No tif files found for this field.")
            return

        scale_factor = float(self._tif_scale.value())
        time_base = int(self._time_base.value())

        try:
            stack = np.stack([tifffile.imread(item.path) for item in items], axis=0)
        except Exception as exc:  # noqa: BLE001
            self._status_label.setText(f"Failed to load tifs: {exc}")
            return

        spatial_scale = (scale_factor,) * (stack.ndim - 1)
        layer_name = "tifs" if field == "image" else f"tifs_{field}"
        self._add_layer(
            stack,
            layer_name,
            time_base,
            spatial_scale,
            spatial_translate=None,
        )

    def _load_tsvs(self) -> None:
        if self._scan is None:
            return
        column = self._tsv_column_combo.currentText()
        if not column:
            return

        items = list(self._scan["tsvs"])
        if not items:
            self._status_label.setText("No TSV files found.")
            return

        deduce_scale = self._tsv_deduce_scale.isChecked()
        fallback_scale = float(self._tsv_scale.value())
        time_base = int(self._time_base.value())

        arrays: list[np.ndarray] = []
        spatial_scale = (fallback_scale, fallback_scale, fallback_scale)
        spatial_translate = (0.0, 0.0, 0.0)

        try:
            for idx, item in enumerate(items):
                data = _read_tsv_data(item.path)
                if column not in data.dtype.names:
                    raise ValueError(
                        f"Column '{column}' not found in {item.path.name}."
                    )
                field_dims = _field_dims_from_coords(data)
                values = data[column].reshape(field_dims)
                arrays.append(values)

                if idx == 0 and deduce_scale:
                    spatial_scale, spatial_translate = (
                        _deduce_spatial_scale_and_translate(data)
                    )
        except Exception as exc:  # noqa: BLE001
            self._status_label.setText(f"Failed to load TSV files: {exc}")
            return

        stack = np.stack(arrays, axis=0)
        layer_name = f"tsv_{column}"
        self._add_layer(
            stack,
            layer_name,
            time_base,
            spatial_scale,
            spatial_translate,
        )

    def _add_layer(
        self,
        data: np.ndarray,
        name: str,
        time_base: int,
        spatial_scale: tuple[float, ...],
        spatial_translate: tuple[float, ...] | None,
    ) -> None:
        ndim = data.ndim
        scale = [1.0] + list(spatial_scale)
        translate = [-time_base] + [0.0] * (ndim - 1)

        if spatial_translate is not None:
            for idx, value in enumerate(spatial_translate, start=1):
                if idx < ndim:
                    translate[idx] = float(value)

        self._viewer.add_image(data, name=name, scale=scale, translate=translate)
