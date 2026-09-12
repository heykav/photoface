from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QScrollArea, QSizePolicy, QWidget

from gui.photo_card import PhotoCard
from gui.theme import TEXT_MUTED

CARD_W = 190
CARD_SPACING = 16


class Gallery(QScrollArea):
    photo_clicked = Signal(int)
    photo_right_clicked = Signal(int)
    escape_pressed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._content = QWidget()
        self._grid = QGridLayout(self._content)
        self._grid.setSpacing(CARD_SPACING)
        self._grid.setContentsMargins(20, 20, 20, 20)
        self.setWidget(self._content)

        self._cards: list[PhotoCard] = []
        self._selected_id: Optional[int] = None
        self._aspect = "3:4"
        self._show_face_boxes = True

        self._empty_label = QLabel(
            "No photos here yet.\n\nPick a folder above and press Analyze,"
            " or clear the current filters."
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 14px; padding: 60px;"
        )

    def set_view_options(self, aspect: str, show_face_boxes: bool) -> None:
        self._aspect = aspect
        self._show_face_boxes = show_face_boxes

    def set_photos(self, photos_with_faces: list[tuple]) -> None:
        """`photos_with_faces` is a list of (photo_row, faces, tag_colors)
        where faces is [(x, y, w, h, color, name), ...]."""
        previously_selected = self._selected_id
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards = []
        self._grid.removeWidget(self._empty_label)
        self._empty_label.setParent(None)

        if not photos_with_faces:
            self._grid.addWidget(self._empty_label, 0, 0)
            self._empty_label.setParent(self._content)
            self._empty_label.show()
            self._selected_id = None
            return

        for photo_row, faces, tag_colors in photos_with_faces:
            card = PhotoCard(photo_row, faces, tag_colors,
                             aspect=self._aspect, show_face_boxes=self._show_face_boxes)
            card.clicked.connect(self.photo_clicked.emit)
            card.clicked.connect(self._on_card_clicked)
            card.right_clicked.connect(self.photo_right_clicked.emit)
            self._cards.append(card)
        self._relayout()

        # keep the selection if that photo is still in the (possibly
        # re-filtered) list, otherwise select the first photo so keyboard
        # navigation has somewhere to start from.
        if previously_selected is not None and any(
            c.photo_id == previously_selected for c in self._cards
        ):
            self.select_photo(previously_selected)
        elif self._cards:
            self.select_photo(self._cards[0].photo_id)

    def select_photo(self, photo_id: Optional[int]) -> None:
        self._selected_id = photo_id
        for card in self._cards:
            card.set_selected(card.photo_id == photo_id)

    def _on_card_clicked(self, photo_id: int) -> None:
        self.select_photo(photo_id)

    def _selected_index(self) -> int:
        for i, card in enumerate(self._cards):
            if card.photo_id == self._selected_id:
                return i
        return -1

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not self._cards:
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key_Escape:
            self.escape_pressed.emit()
            return
        if key == Qt.Key_Return or key == Qt.Key_Enter:
            if self._selected_id is not None:
                self.photo_clicked.emit(self._selected_id)
            return

        cols = self._columns()
        idx = self._selected_index()
        if idx == -1:
            idx = 0
        elif key == Qt.Key_Right:
            idx = min(idx + 1, len(self._cards) - 1)
        elif key == Qt.Key_Left:
            idx = max(idx - 1, 0)
        elif key == Qt.Key_Down:
            idx = min(idx + cols, len(self._cards) - 1)
        elif key == Qt.Key_Up:
            idx = max(idx - cols, 0)
        else:
            super().keyPressEvent(event)
            return

        new_id = self._cards[idx].photo_id
        self.select_photo(new_id)
        self.ensureWidgetVisible(self._cards[idx])

    def _columns(self) -> int:
        available = self.viewport().width() - 24
        col_width = CARD_W + CARD_SPACING
        return max(1, available // col_width)

    def _relayout(self) -> None:
        cols = self._columns()
        for i, card in enumerate(self._cards):
            self._grid.removeWidget(card)
        for i, card in enumerate(self._cards):
            row, col = divmod(i, cols)
            self._grid.addWidget(card, row, col)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._cards:
            new_cols = self._columns()
            current_cols = self._grid.columnCount()
            if new_cols != current_cols:
                self._relayout()
