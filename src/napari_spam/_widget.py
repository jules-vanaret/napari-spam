from importlib import resources

import numpy as np
import tifffile
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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
    _natural_sort_key,
    _scan_folder,
    _split_tokens,
    TOKEN_SEPARATORS,
)

SPAM_DOC_URL = "https://www.spam-project.dev/docs/"
SPAM_PAPER_URL = "https://joss.theoj.org/papers/10.21105/joss.02286"
SHOW_FILE_LIST_DEBUG = False


def _grouped_sort_paths(paths: list[Path]) -> list[Path]:
    if len(paths) < 2:
        return list(paths)

    token_lists = [_split_tokens(path.stem) for path in paths]
    token_set = {tuple(tokens) for tokens in token_lists}

    keyed: list[tuple[Path, list[str], str]] = []
    for path, tokens in zip(paths, token_lists, strict=False):
        base_tokens = tokens
        variant_token = ""
        if len(tokens) >= 2 and tuple(tokens[:-1]) in token_set:
            base_tokens = tokens[:-1]
            variant_token = tokens[-1]
        elif len(tokens) >= 2:
            without_penultimate = tokens[:-2] + tokens[-1:]
            if tuple(without_penultimate) in token_set:
                base_tokens = without_penultimate
                variant_token = tokens[-2]

        keyed.append((path, base_tokens, variant_token))

    def sort_key(
        entry: tuple[Path, list[str], str],
    ) -> tuple[list[object], list[object], int, list[object]]:
        _, base_tokens, variant_token = entry
        if base_tokens:
            field_key = _natural_sort_key(base_tokens[-1])
            prefix_key = _natural_sort_key("-".join(base_tokens[:-1]))
        else:
            field_key = []
            prefix_key = []
        variant_key = _natural_sort_key(variant_token)
        variant_flag = 1 if variant_token else 0
        return field_key, variant_flag, prefix_key, variant_key

    return [entry[0] for entry in sorted(keyed, key=sort_key)]


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
        self._selection_order: list[str] = []

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
        self._status_label.setStyleSheet("color: #b00020;")

        self._group_mode_combo = QComboBox()
        self._group_mode_combo.addItems(
            ["Stack by selection order", "Stack by sorted names"]
        )
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        row_height = self._file_list.fontMetrics().height() + 6
        self._file_list.setFixedHeight(
            row_height * 4 + 2 * self._file_list.frameWidth()
        )
        self._file_list.selectionModel().selectionChanged.connect(
            self._on_file_selection_changed
        )
        self._debug_list_button = QPushButton("Print list order")
        self._debug_list_button.clicked.connect(self._print_file_list)
        self._debug_list_button.setVisible(SHOW_FILE_LIST_DEBUG)

        self._time_base = QSpinBox()
        self._time_base.setRange(0, 10_000)
        self._time_base.setValue(0)
        self._load_button = QPushButton("Load")

        self._crop_enabled = QCheckBox("Enable crop")
        self._crop_enabled.stateChanged.connect(self._set_crop_enabled)
        self._crop_spins: list[tuple[QDoubleSpinBox, QDoubleSpinBox]] = []
        crop_group = QGroupBox("Crop (world units)")
        crop_layout = QVBoxLayout()
        crop_layout.addWidget(self._crop_enabled)
        crop_form = QFormLayout()
        for axis_label in ["Z", "Y", "X"]:
            min_spin = QDoubleSpinBox()
            max_spin = QDoubleSpinBox()
            for spin in (min_spin, max_spin):
                spin.setRange(-1e9, 1e9)
                spin.setDecimals(4)
                spin.setSingleStep(0.1)
                spin.setValue(0.0)
                spin.setEnabled(False)
            row = QHBoxLayout()
            row.addWidget(QLabel("Min"))
            row.addWidget(min_spin)
            row.addWidget(QLabel("Max"))
            row.addWidget(max_spin)
            crop_form.addRow(axis_label, row)
            self._crop_spins.append((min_spin, max_spin))
        crop_layout.addLayout(crop_form)
        crop_group.setLayout(crop_layout)

        self._options_panel = QWidget()
        options_layout = QVBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.addWidget(self._action_combo)
        options_layout.addWidget(self._stack)

        files_group = QGroupBox("Files")
        files_layout = QVBoxLayout()
        files_layout.addWidget(self._file_list)
        files_layout.addWidget(self._group_mode_combo)
        files_layout.addWidget(self._debug_list_button)
        files_group.setLayout(files_layout)
        options_layout.addWidget(files_group)

        options_layout.addWidget(crop_group)

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
        logo_ref = resources.files("napari_spam").joinpath("assets/napari-spam2.png")
        with resources.as_file(logo_ref) as logo_path:
            if logo_path.exists():
                pixmap = QPixmap(str(logo_path))
                if not pixmap.isNull():
                    logo_label.setPixmap(
                        pixmap.scaledToHeight(30, Qt.SmoothTransformation)
                    )

        links_layout = QVBoxLayout()
        links_layout.setContentsMargins(0, 0, 0, 0)

        links_label = QLabel(
            f'[<a href="{SPAM_DOC_URL}">doc</a>, '
            f'<a href="{SPAM_PAPER_URL}">paper</a>]'
        )
        links_label.setTextFormat(Qt.RichText)
        links_label.setOpenExternalLinks(True)
        links_layout.addWidget(links_label)

        header_row.addWidget(logo_label)
        header_row.addLayout(links_layout)
        header_row.addStretch(1)
        return header_row

    def _build_tif_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout()

        self._tif_scale = QLineEdit()
        self._tif_scale.setText("1.0")
        self._tif_scale.setValidator(QDoubleValidator(0.0, 1e12, 8))
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
        self._tsv_scale = QLineEdit()
        self._tsv_scale.setText("1.0")
        self._tsv_scale.setValidator(QDoubleValidator(0.0, 1e12, 8))
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
        elif action == _ACTION_TSVS:
            self._stack.setCurrentIndex(1)
            self._refresh_tsv_columns()
        elif action == _ACTION_VTKS:
            self._stack.setCurrentIndex(2)
        else:
            self._stack.setCurrentIndex(0)
        self._populate_file_list()

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

    def _read_positive_scale(self, line_edit: QLineEdit, label: str) -> float | None:
        text = "".join(line_edit.text().split())
        if not text:
            line_edit.setText("1")
            return 1.0
        if text.count(",") > 1 or ("," in text and "." in text):
            self._status_label.setText(f"{label} scale must be a positive number.")
            return None
        if text.count(",") == 1:
            text = text.replace(",", ".")
        try:
            value = float(text)
        except ValueError:
            self._status_label.setText(f"{label} scale must be a positive number.")
            return None
        if value <= 0.0:
            self._status_label.setText(f"{label} scale must be a positive number.")
            return None
        return value

    def _set_crop_enabled(self) -> None:
        enabled = self._crop_enabled.isChecked()
        for min_spin, max_spin in self._crop_spins:
            min_spin.setEnabled(enabled)
            max_spin.setEnabled(enabled)

    def _populate_file_list(self) -> None:
        self._file_list.clear()
        self._selection_order.clear()
        if self._scan is None:
            return

        action = self._action_combo.currentText()
        if action == _ACTION_TIFS:
            items = [tif.path for tif in self._scan["tifs"]]
        elif action == _ACTION_TSVS:
            items = [tsv.path for tsv in self._scan["tsvs"]]
        elif action == _ACTION_VTKS:
            items = list(self._scan["vtks"])
        else:
            items = []

        for path in _grouped_sort_paths(items):
            list_item = QListWidgetItem(path.name)
            list_item.setData(Qt.UserRole, str(path))
            self._file_list.addItem(list_item)

    def _on_file_selection_changed(self, selected, deselected) -> None:
        for index in deselected.indexes():
            item = self._file_list.itemFromIndex(index)
            if item is None:
                continue
            key = item.data(Qt.UserRole)
            if key in self._selection_order:
                self._selection_order.remove(key)

        for index in selected.indexes():
            item = self._file_list.itemFromIndex(index)
            if item is None:
                continue
            key = item.data(Qt.UserRole)
            if key not in self._selection_order:
                self._selection_order.append(key)

    def _selected_paths(self) -> list[Path]:
        items = self._file_list.selectedItems()
        if not items:
            return []

        selected = [Path(item.data(Qt.UserRole)) for item in items]
        if self._group_mode_combo.currentText() == "Group by sorted names":
            return _grouped_sort_paths(selected)

        selected_keys = {str(path) for path in selected}
        ordered_paths = [
            Path(key) for key in self._selection_order if key in selected_keys
        ]
        if len(ordered_paths) == len(selected):
            return ordered_paths

        ordered_names = {str(path) for path in ordered_paths}
        remainder = [path for path in selected if str(path) not in ordered_names]
        return ordered_paths + remainder

    def _print_file_list(self) -> None:
        entries = []
        for idx in range(self._file_list.count()):
            item = self._file_list.item(idx)
            if item is not None:
                entries.append(item.text())
        print("File list order:")
        for name in entries:
            print(f"- {name}")

    def _load_tifs(self) -> None:
        if self._scan is None:
            return
        selected_paths = self._selected_paths()
        if not selected_paths:
            self._status_label.setText("Select one or more tif files to load.")
            return

        tif_by_path = {tif.path: tif for tif in self._scan["tifs"]}
        items = [tif_by_path[path] for path in selected_paths if path in tif_by_path]
        if not items:
            self._status_label.setText("Selected files are not tif images.")
            return

        scale_factor = self._read_positive_scale(self._tif_scale, "TIF")
        if scale_factor is None:
            return
        time_base = int(self._time_base.value())

        try:
            stack = np.stack([tifffile.imread(item.path) for item in items], axis=0)
        except Exception as exc:  # noqa: BLE001
            self._status_label.setText(f"Failed to load tifs: {exc}")
            return

        spatial_scale = (scale_factor,) * (stack.ndim - 1)
        spatial_translate = (0.0,) * len(spatial_scale)
        stack, spatial_translate = self._apply_crop(
            stack, spatial_scale, spatial_translate
        )
        layer_name = "tifs"
        self._add_layer(
            stack,
            layer_name,
            time_base,
            spatial_scale,
            spatial_translate=spatial_translate,
        )

    def _load_tsvs(self) -> None:
        if self._scan is None:
            return
        column = self._tsv_column_combo.currentText()
        if not column:
            return

        selected_paths = self._selected_paths()
        if not selected_paths:
            self._status_label.setText("Select one or more TSV files to load.")
            return

        tsv_by_path = {tsv.path: tsv for tsv in self._scan["tsvs"]}
        items = [tsv_by_path[path] for path in selected_paths if path in tsv_by_path]
        if not items:
            self._status_label.setText("Selected files are not TSV files.")
            return

        deduce_scale = self._tsv_deduce_scale.isChecked()
        fallback_scale = None
        if not deduce_scale:
            fallback_scale = self._read_positive_scale(self._tsv_scale, "TSV")
            if fallback_scale is None:
                return
        time_base = int(self._time_base.value())

        arrays: list[np.ndarray] = []
        if fallback_scale is None:
            fallback_scale = 1.0
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
        stack, spatial_translate = self._apply_crop(
            stack, spatial_scale, spatial_translate
        )
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
        translate = [time_base] + [0.0] * (ndim - 1)

        if spatial_translate is not None:
            for idx, value in enumerate(spatial_translate, start=1):
                if idx < ndim:
                    translate[idx] = float(value)

        self._viewer.add_image(data, name=name, scale=scale, translate=translate)

    def _apply_crop(
        self,
        data: np.ndarray,
        spatial_scale: tuple[float, ...],
        spatial_translate: tuple[float, ...],
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        if not self._crop_enabled.isChecked():
            return data, spatial_translate

        spatial_dims = len(spatial_scale)
        crop_spins = self._crop_spins[-spatial_dims:]
        slices: list[slice] = [slice(None)]
        new_translate = list(spatial_translate)

        for idx, ((min_spin, max_spin), scale, translate, size) in enumerate(
            zip(
                crop_spins,
                spatial_scale,
                spatial_translate,
                data.shape[-spatial_dims:],
                strict=False,
            )
        ):
            min_val = float(min_spin.value())
            max_val = float(max_spin.value())
            if min_val > max_val:
                min_val, max_val = max_val, min_val

            i_min = int(np.floor((min_val - translate) / scale))
            i_max = int(np.ceil((max_val - translate) / scale))
            i_min = max(i_min, 0)
            i_max = min(i_max, size - 1)
            if i_min > i_max:
                return data, spatial_translate

            slices.append(slice(i_min, i_max + 1))
            new_translate[idx] = translate + i_min * scale

        cropped = data[tuple(slices)]
        return cropped, tuple(new_translate)
