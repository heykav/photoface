from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QFileSystemWatcher, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QVBoxLayout,
    QWidget,
)

from analyzer import Analyzer, FaceEngine
from database import Database
from gui.analyze_worker import AnalyzeWorker
from gui.gallery import Gallery
from gui.lightbox import LightboxDialog
from gui.settings_dialog import SettingsDialog, get_settings
from gui.sidebar import Sidebar
from paths import db_path


def _tag_color(name: str) -> str:
    """A stable color per tag name, so different tags are visually
    distinguishable as dots on a card without needing a `color` column."""
    import colorsys
    import hashlib
    hue = (int(hashlib.sha1(name.encode()).hexdigest(), 16) % 360) / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("photoface")
        self.resize(1300, 850)

        self.db = Database(db_path())
        self._engine: Optional[FaceEngine] = None
        self._worker: Optional[AnalyzeWorker] = None

        self._person_ids_filter: list[int] = []
        self._tag_ids_filter: list[int] = []
        self._geotagged_only = False
        self._duplicates_only = False

        self._watcher = QFileSystemWatcher()
        self._watcher.directoryChanged.connect(self._on_folder_changed)
        self._watch_debounce = QTimer(singleShot=True)
        self._watch_debounce.setInterval(2000)
        self._watch_debounce.timeout.connect(self._start_analysis)

        self._build_ui()
        self._load_settings()
        self._reload_all()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(10, 10, 10, 10)
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Pick a folder of photos to analyze…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_folder)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setObjectName("primary")
        self.analyze_btn.clicked.connect(self._start_analysis)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedWidth(220)
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._open_settings)

        top_layout.addWidget(QLabel("Folder:"))
        top_layout.addWidget(self.folder_edit, stretch=1)
        top_layout.addWidget(browse_btn)
        top_layout.addWidget(self.analyze_btn)
        top_layout.addWidget(self.progress)
        top_layout.addWidget(settings_btn)
        outer.addWidget(top)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.gallery = Gallery()
        self.gallery.photo_clicked.connect(self._open_lightbox)
        self.gallery.photo_right_clicked.connect(self._show_tag_menu)
        self.gallery.escape_pressed.connect(self._clear_person_filter)
        body_layout.addWidget(self.gallery, stretch=1)

        self.sidebar = Sidebar(self.db)
        self.sidebar.persons_filter_changed.connect(self._on_persons_filter)
        self.sidebar.tags_filter_changed.connect(self._on_tags_filter)
        self.sidebar.geotag_toggled.connect(self._on_geotag_toggle)
        self.sidebar.duplicates_toggled.connect(self._on_duplicates_toggle)
        self.sidebar.person_renamed.connect(self._on_person_renamed)
        self.sidebar.persons_merged.connect(self._on_persons_merged)
        self.sidebar.person_deleted.connect(self._on_person_deleted)
        self.sidebar.tag_deleted.connect(self._on_tag_deleted)
        self.sidebar.recluster_requested.connect(self._recluster_now)
        body_layout.addWidget(self.sidebar)

        outer.addWidget(body, stretch=1)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #9a9ca1; padding: 4px 10px;")
        outer.addWidget(self.status)

    def _load_settings(self) -> None:
        settings = get_settings()
        last_folder = settings.value("last_folder", "")
        if last_folder:
            self.folder_edit.setText(last_folder)
        aspect = settings.value("aspect", "3:4")
        show_boxes = settings.value("show_face_boxes", True, type=bool)
        self.gallery.set_view_options(aspect, show_boxes)
        self._sort_order = settings.value("sort_order", "filename")
        self._threshold = float(settings.value("clustering_threshold", 0.363))
        self._watch_folder = settings.value("watch_folder", False, type=bool)
        self._rearm_watcher()

    # ------------------------------------------------------------------ #
    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose photo folder")
        if folder:
            self.folder_edit.setText(folder)
            get_settings().setValue("last_folder", folder)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec():
            dialog.save()
            self._load_settings()
            self._reload_all()

    # ------------------------------------------------------------------ #
    def _start_analysis(self) -> None:
        folder = self.folder_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "photoface", "Pick a valid folder first.")
            return
        get_settings().setValue("last_folder", folder)

        if self._engine is None:
            try:
                self._engine = FaceEngine()
            except FileNotFoundError as e:
                QMessageBox.critical(self, "photoface", str(e))
                return

        analyzer = Analyzer(self.db, self._engine, threshold=self._threshold)
        self._worker = AnalyzeWorker(analyzer, Path(folder))
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_analysis_finished)
        self._worker.failed.connect(self._on_analysis_failed)

        self.analyze_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._worker.start()

    def _on_progress(self, done: int, total: int, path: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.status.setText(f"Analyzing {done}/{total}: {Path(path).name}")

    def _on_analysis_finished(self) -> None:
        self.analyze_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status.setText("Analysis complete.")
        self._reload_all()
        self._rearm_watcher()

    def _on_analysis_failed(self, message: str) -> None:
        self.analyze_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "photoface", f"Analysis failed:\n{message}")

    # ------------------------------------------------------------------ #
    def _rearm_watcher(self) -> None:
        """(Re)point the QFileSystemWatcher at the current folder and every
        subdirectory - it does not watch recursively on its own, and doesn't
        pick up newly-created subdirectories until re-armed here, which
        happens after every analysis run."""
        existing = self._watcher.directories()
        if existing:
            self._watcher.removePaths(existing)
        if not getattr(self, "_watch_folder", False):
            return
        folder = self.folder_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            return
        paths = [folder] + [str(p) for p in Path(folder).rglob("*") if p.is_dir()]
        if paths:
            self._watcher.addPaths(paths)

    def _on_folder_changed(self, _path: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            return  # our own analysis isn't what changed the folder anyway
        # debounce: a burst of filesystem events (e.g. copying many files)
        # collapses into a single re-analysis, 2s after things go quiet.
        self._watch_debounce.start()

    def _recluster_now(self) -> None:
        if self._engine is None:
            try:
                self._engine = FaceEngine()
            except FileNotFoundError as e:
                QMessageBox.critical(self, "photoface", str(e))
                return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            Analyzer(self.db, self._engine, threshold=self._threshold).recluster_all()
        finally:
            QApplication.restoreOverrideCursor()
        self._reload_all()

    # ------------------------------------------------------------------ #
    def _on_persons_filter(self, ids: list[int]) -> None:
        self._person_ids_filter = ids
        self._reload_gallery()

    def _on_tags_filter(self, ids: list[int]) -> None:
        self._tag_ids_filter = ids
        self._reload_gallery()

    def _on_geotag_toggle(self, checked: bool) -> None:
        self._geotagged_only = checked
        self._reload_gallery()

    def _on_duplicates_toggle(self, checked: bool) -> None:
        self._duplicates_only = checked
        self._reload_gallery()

    def _on_person_renamed(self, person_id: int, name: str) -> None:
        self.db.rename_person(person_id, name)
        self._reload_all()

    def _on_persons_merged(self, src: int, dst: int) -> None:
        self.db.merge_persons(src, dst)
        self._reload_all()

    def _on_person_deleted(self, person_id: int) -> None:
        self.db.delete_person(person_id)
        self._reload_all()

    def _on_tag_deleted(self, tag_id: int) -> None:
        self.db.conn.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
        self.db.conn.commit()
        self._reload_all()

    def _clear_person_filter(self) -> None:
        self.sidebar.people_list.clearSelection()

    def _show_tag_menu(self, photo_id: int) -> None:
        current = self.db.tags_for_photo(photo_id)
        current_names = {t["name"] for t in current}
        menu = QMenu(self)
        add_action = menu.addAction("Add tag…")
        remove_actions = {}
        if current:
            menu.addSeparator()
            for tag in current:
                act = menu.addAction(f"Remove #{tag['name']}")
                remove_actions[act] = tag["id"]

        chosen = menu.exec(QCursor.pos())
        if chosen == add_action:
            name, ok = QInputDialog.getText(self, "Add tag", "Tag name:")
            if ok and name.strip():
                self.db.tag_photo(photo_id, name.strip())
                self._reload_all()
        elif chosen in remove_actions:
            self.db.untag_photo(photo_id, remove_actions[chosen])
            self._reload_all()

    # ------------------------------------------------------------------ #
    def _reload_all(self) -> None:
        self.sidebar.refresh()
        self._reload_gallery()

    def _current_photos(self):
        photos = self.db.filtered_photos(
            person_ids=self._person_ids_filter or None,
            tag_ids=self._tag_ids_filter or None,
            geotagged_only=self._geotagged_only,
            order_by=self._sort_order,
        )
        if self._duplicates_only:
            dup_ids = {row["id"] for group in self.db.duplicate_groups() for row in group}
            photos = [p for p in photos if p["id"] in dup_ids]
        return photos

    def _faces_with_display(self, photo_id: int) -> list[dict]:
        persons = {p["id"]: p for p in self.db.all_persons()}
        out = []
        for face in self.db.faces_for_photo(photo_id):
            person = persons.get(face["person_id"])
            out.append({
                "id": face["id"], "x": face["x"], "y": face["y"],
                "w": face["w"], "h": face["h"], "person_id": face["person_id"],
                "name": person["name"] if person else "",
                "color": person["color"] if person else "#888888",
            })
        return out

    def _reload_gallery(self) -> None:
        photos = self._current_photos()
        packed = []
        for photo in photos:
            faces = self._faces_with_display(photo["id"])
            face_tuples = [(f["x"], f["y"], f["w"], f["h"], f["color"], f["name"])
                          for f in faces]
            tag_colors = [_tag_color(t["name"]) for t in self.db.tags_for_photo(photo["id"])]
            packed.append((photo, face_tuples, tag_colors))
        self.gallery.set_photos(packed)
        self.status.setText(f"{len(photos)} photo(s)")

    def _open_lightbox(self, photo_id: int) -> None:
        photos = self._current_photos()
        items = []
        start_index = 0
        for i, photo in enumerate(photos):
            if photo["id"] == photo_id:
                start_index = i
            items.append({"photo_row": photo, "faces": self._faces_with_display(photo["id"])})
        if not items:
            return
        dialog = LightboxDialog(self.db, items, start_index, self)
        dialog.exec()
        self._reload_all()  # face edits made in the lightbox may have changed persons

    def closeEvent(self, event) -> None:  # noqa: N802
        self.db.close()
        super().closeEvent(event)
