from __future__ import annotations

import webbrowser
from typing import Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDialog, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QInputDialog, QLabel,
    QMenu, QPushButton, QVBoxLayout, QWidget,
)

from database import Database


class _View(QGraphicsView):
    """QGraphicsView with wheel-zoom centered on the cursor and hand-drag pan."""

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#0b0c0d")))
        self._zoom = 0

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = self._zoom + (1 if event.angleDelta().y() > 0 else -1)
        if -10 <= new_zoom <= 20:
            self.scale(factor, factor)
            self._zoom = new_zoom


class LightboxDialog(QDialog):
    def __init__(self, db: Database, items: list[dict], start_index: int, parent=None):
        """`items`: list of dicts with keys photo_row, faces (list of dicts
        with id, x, y, w, h, person_id, name, color)."""
        super().__init__(parent)
        self.db = db
        self.items = items
        self.index = start_index
        self.setWindowTitle("Photo")
        self.resize(1100, 750)
        self.setStyleSheet(parent.styleSheet() if parent else "")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_bar = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setStyleSheet("padding: 8px; font-weight: 600;")
        top_bar.addWidget(self.title_label)
        top_bar.addStretch()
        self.meta_label = QLabel()
        self.meta_label.setStyleSheet("padding: 8px; color: #9a9ca1;")
        top_bar.addWidget(self.meta_label)
        close_btn = QPushButton("Close")
        close_btn.setObjectName("ghost")
        close_btn.clicked.connect(self.close)
        top_bar.addWidget(close_btn)
        layout.addLayout(top_bar)

        self.scene = QGraphicsScene()
        self.view = _View(self.scene)
        self.view.mousePressEvent = self._wrap_view_click(self.view.mousePressEvent)
        layout.addWidget(self.view, stretch=1)

        nav_bar = QHBoxLayout()
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(self.show_previous)
        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(self.show_next)
        nav_bar.addWidget(prev_btn)
        nav_bar.addStretch()
        self.geo_label = QLabel()
        self.geo_label.setStyleSheet("color: #9a9ca1;")
        nav_bar.addWidget(self.geo_label)
        nav_bar.addStretch()
        nav_bar.addWidget(next_btn)
        layout.addLayout(nav_bar)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._render_current()

    # ------------------------------------------------------------------ #
    def _wrap_view_click(self, original):
        def handler(event):
            item = self.view.itemAt(event.pos())
            if item is None or item is self._pixmap_item:
                if event.button() == Qt.LeftButton and item is None:
                    self.close()
                    return
            original(event)
        return handler

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Left:
            self.show_previous()
        elif event.key() == Qt.Key_Right:
            self.show_next()
        else:
            super().keyPressEvent(event)

    def show_previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._render_current()

    def show_next(self) -> None:
        if self.index < len(self.items) - 1:
            self.index += 1
            self._render_current()

    # ------------------------------------------------------------------ #
    def _render_current(self) -> None:
        self.scene.clear()
        item = self.items[self.index]
        photo = item["photo_row"]
        self.title_label.setText(photo["path"].split("/")[-1])

        pix = QPixmap(photo["path"])
        if pix.isNull():
            self.meta_label.setText("(could not load image)")
            return
        self._pixmap_item = QGraphicsPixmapItem(pix)
        self.scene.addItem(self._pixmap_item)
        self.scene.setSceneRect(QRectF(pix.rect()))
        self.view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self.view._zoom = 0

        for face in item["faces"]:
            self._add_face_box(face)

        meta_bits = []
        if photo["exif_date"]:
            meta_bits.append(photo["exif_date"])
        self.meta_label.setText("  •  ".join(meta_bits))

        if photo["exif_lat"] is not None and photo["exif_lon"] is not None:
            lat, lon = photo["exif_lat"], photo["exif_lon"]
            self.geo_label.setText(f"📍 {lat:.5f}, {lon:.5f}")
            self.geo_label.setCursor(Qt.PointingHandCursor)
            self.geo_label.mousePressEvent = lambda e, la=lat, lo=lon: self._geo_menu(la, lo, e)
        else:
            self.geo_label.setText("")

    def _add_face_box(self, face: dict) -> None:
        rect_item = QGraphicsRectItem(face["x"], face["y"], face["w"], face["h"])
        color = QColor(face["color"])
        pen = QPen(color)
        pen.setWidth(3)
        rect_item.setPen(pen)
        rect_item.setBrush(Qt.NoBrush)
        rect_item.setData(0, face["id"])
        self.scene.addItem(rect_item)

        if face.get("name"):
            # a solid chip behind the name, not just colored text directly on
            # the photo - legibility against a face box color that can be
            # close to the photo's own colors otherwise.
            font = QFont()
            font.setPixelSize(max(12, int(face["h"] * 0.09)))
            font.setWeight(QFont.DemiBold)
            label = QGraphicsSimpleTextItem(face["name"])
            label.setFont(font)
            label.setBrush(QBrush(QColor("white")))
            label_rect = label.boundingRect()
            pad_x, pad_y = 6, 3
            chip = QGraphicsRectItem(
                face["x"], face["y"] + face["h"] + 4,
                label_rect.width() + pad_x * 2, label_rect.height() + pad_y * 2,
            )
            chip.setBrush(QBrush(color))
            chip.setPen(Qt.NoPen)
            self.scene.addItem(chip)
            label.setPos(face["x"] + pad_x, face["y"] + face["h"] + 4 + pad_y)
            self.scene.addItem(label)

        rect_item.setAcceptHoverEvents(True)
        rect_item.setCursor(Qt.PointingHandCursor)
        rect_item.mousePressEvent = lambda e, f=face, r=rect_item: self._face_context_menu(f, e)

    def _face_context_menu(self, face: dict, event) -> None:
        if event.button() != Qt.RightButton:
            return
        menu = QMenu(self)
        move_menu = menu.addMenu("Move to person…")
        move_actions = {}
        for person in self.db.all_persons():
            if person["id"] == face.get("person_id"):
                continue
            act = move_menu.addAction(person["name"])
            move_actions[act] = person["id"]
        split_action = menu.addAction("Split off as new person")
        delete_action = menu.addAction("Delete this face box")

        global_pos = self.view.mapToGlobal(
            self.view.mapFromScene(event.scenePos())
        )
        chosen = menu.exec(global_pos)
        if chosen == split_action:
            import random
            from analyzer import PERSON_COLORS
            color = random.choice(PERSON_COLORS)
            new_person = self.db.create_person(
                f"Persona {len(self.db.all_persons()) + 1}", color
            )
            self.db.set_face_person(face["id"], new_person, pinned=True)
            self._render_current()
        elif chosen == delete_action:
            self.db.delete_face(face["id"])
            self._render_current()
        elif chosen in move_actions:
            self.db.set_face_person(face["id"], move_actions[chosen], pinned=True)
            self._render_current()

    def _geo_menu(self, lat: float, lon: float, event) -> None:
        menu = QMenu(self)
        copy_action = menu.addAction("Copy coordinates")
        open_action = menu.addAction("Open in OpenStreetMap")
        chosen = menu.exec(self.geo_label.mapToGlobal(event.pos()))
        if chosen == copy_action:
            QApplication.clipboard().setText(f"{lat}, {lon}")
        elif chosen == open_action:
            webbrowser.open(f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=16")
