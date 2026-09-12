from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMenu,
    QPushButton, QVBoxLayout, QWidget,
)

from database import Database


def _swatch_icon(color: str, size: int = 14) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    return QIcon(pix)


class Sidebar(QWidget):
    persons_filter_changed = Signal(list)   # list[int]
    tags_filter_changed = Signal(list)       # list[int]
    geotag_toggled = Signal(bool)
    duplicates_toggled = Signal(bool)
    person_renamed = Signal(int, str)
    persons_merged = Signal(int, int)        # src, dst
    person_deleted = Signal(int)
    tag_deleted = Signal(int)
    recluster_requested = Signal()

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setObjectName("sidebarPanel")
        self.setFixedWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 14)
        layout.setSpacing(4)

        layout.addWidget(self._header("People"))
        self.people_list = QListWidget()
        self.people_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.people_list.itemSelectionChanged.connect(self._on_people_selection)
        self.people_list.itemDoubleClicked.connect(self._on_person_double_click)
        self.people_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.people_list.customContextMenuRequested.connect(self._person_context_menu)
        layout.addWidget(self.people_list, stretch=2)

        recluster_btn = QPushButton("Recluster now")
        recluster_btn.clicked.connect(self.recluster_requested.emit)
        layout.addWidget(recluster_btn)

        layout.addWidget(self._header("Tags"))
        self.tags_list = QListWidget()
        self.tags_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.tags_list.itemSelectionChanged.connect(self._on_tags_selection)
        self.tags_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tags_list.customContextMenuRequested.connect(self._tag_context_menu)
        layout.addWidget(self.tags_list, stretch=1)

        layout.addWidget(self._header("Filters"))
        self.geotag_check = QCheckBox("Geotagged only")
        self.geotag_check.toggled.connect(self.geotag_toggled.emit)
        layout.addWidget(self.geotag_check)

        self.duplicates_check = QCheckBox("Possible duplicates only")
        self.duplicates_check.setToolTip(
            "Photos whose perceptual hash is within a small distance of "
            "another photo's - likely re-exports, light edits, or re-saves "
            "of the same shot."
        )
        self.duplicates_check.toggled.connect(self.duplicates_toggled.emit)
        layout.addWidget(self.duplicates_check)

    @staticmethod
    def _header(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionHeader")
        return label

    def refresh(self) -> None:
        self._reload_people()
        self._reload_tags()

    def _reload_people(self) -> None:
        selected = {self.people_list.item(i).data(Qt.UserRole)
                   for i in range(self.people_list.count())
                   if self.people_list.item(i).isSelected()}
        self.people_list.clear()
        for person in self.db.all_persons():
            item = QListWidgetItem(f"{person['name']}  ({person['face_count']})")
            item.setData(Qt.UserRole, person["id"])
            item.setIcon(_swatch_icon(person["color"]))
            self.people_list.addItem(item)
            if person["id"] in selected:
                item.setSelected(True)

    def _reload_tags(self) -> None:
        selected = {self.tags_list.item(i).data(Qt.UserRole)
                   for i in range(self.tags_list.count())
                   if self.tags_list.item(i).isSelected()}
        self.tags_list.clear()
        for tag in self.db.all_tags():
            item = QListWidgetItem(f"#{tag['name']}  ({tag['photo_count']})")
            item.setData(Qt.UserRole, tag["id"])
            self.tags_list.addItem(item)
            if tag["id"] in selected:
                item.setSelected(True)

    def _on_people_selection(self) -> None:
        ids = [item.data(Qt.UserRole) for item in self.people_list.selectedItems()]
        self.persons_filter_changed.emit(ids)

    def _on_tags_selection(self) -> None:
        ids = [item.data(Qt.UserRole) for item in self.tags_list.selectedItems()]
        self.tags_filter_changed.emit(ids)

    def _on_person_double_click(self, item: QListWidgetItem) -> None:
        person_id = item.data(Qt.UserRole)
        current = next((p["name"] for p in self.db.all_persons() if p["id"] == person_id), "")
        name, ok = QInputDialog.getText(self, "Rename person", "Name:", text=current)
        if ok and name.strip():
            self.person_renamed.emit(person_id, name.strip())

    def _person_context_menu(self, pos) -> None:
        item = self.people_list.itemAt(pos)
        if item is None:
            return
        person_id = item.data(Qt.UserRole)
        menu = QMenu(self)
        rename_action = menu.addAction("Rename…")
        merge_menu = menu.addMenu("Merge into…")
        others = [p for p in self.db.all_persons() if p["id"] != person_id]
        merge_actions = {}
        for other in others:
            act = merge_menu.addAction(other["name"])
            merge_actions[act] = other["id"]
        delete_action = menu.addAction("Delete person")

        chosen = menu.exec(self.people_list.mapToGlobal(pos))
        if chosen == rename_action:
            self._on_person_double_click(item)
        elif chosen == delete_action:
            self.person_deleted.emit(person_id)
        elif chosen in merge_actions:
            self.persons_merged.emit(person_id, merge_actions[chosen])

    def _tag_context_menu(self, pos) -> None:
        item = self.tags_list.itemAt(pos)
        if item is None:
            return
        tag_id = item.data(Qt.UserRole)
        menu = QMenu(self)
        delete_action = menu.addAction("Delete tag")
        if menu.exec(self.tags_list.mapToGlobal(pos)) == delete_action:
            self.tag_deleted.emit(tag_id)
