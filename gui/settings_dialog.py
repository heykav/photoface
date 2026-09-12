from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QLabel,
)

ORG, APP = "photoface", "photoface"


def get_settings() -> QSettings:
    return QSettings(ORG, APP)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        settings = get_settings()

        form = QFormLayout(self)

        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(["3:4", "4:3"])
        self.aspect_combo.setCurrentText(settings.value("aspect", "3:4"))
        form.addRow("Preview aspect", self.aspect_combo)

        self.face_boxes_check = QCheckBox("Show face boxes on thumbnails")
        self.face_boxes_check.setChecked(settings.value("show_face_boxes", True, type=bool))
        form.addRow(self.face_boxes_check)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["filename", "date"])
        self.sort_combo.setCurrentText(settings.value("sort_order", "filename"))
        form.addRow("Sort by", self.sort_combo)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.05, 0.95)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(float(settings.value("clustering_threshold", 0.363)))
        form.addRow("Clustering similarity threshold", self.threshold_spin)
        form.addRow(QLabel("Higher = stricter matching (fewer, purer person groups)."))

        self.log_combo = QComboBox()
        self.log_combo.addItems(["quiet", "normal", "verbose"])
        self.log_combo.setCurrentText(settings.value("log_verbosity", "normal"))
        form.addRow("Log verbosity", self.log_combo)

        self.watch_folder_check = QCheckBox(
            "Watch the analyzed folder and re-analyze automatically on changes"
        )
        self.watch_folder_check.setChecked(settings.value("watch_folder", False, type=bool))
        form.addRow(self.watch_folder_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def save(self) -> None:
        settings = get_settings()
        settings.setValue("aspect", self.aspect_combo.currentText())
        settings.setValue("show_face_boxes", self.face_boxes_check.isChecked())
        settings.setValue("sort_order", self.sort_combo.currentText())
        settings.setValue("clustering_threshold", self.threshold_spin.value())
        settings.setValue("log_verbosity", self.log_combo.currentText())
        settings.setValue("watch_folder", self.watch_folder_check.isChecked())
