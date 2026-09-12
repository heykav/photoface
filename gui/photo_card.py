from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from gui.thumbnails import get_thumbnail, scale_bbox_to_thumb


class PhotoCard(QFrame):
    clicked = Signal(int)          # photo_id
    right_clicked = Signal(int)    # photo_id

    def __init__(self, photo_row, faces: list, tag_colors: list[str],
                aspect: str = "3:4", show_face_boxes: bool = True, parent=None):
        super().__init__(parent)
        self.photo_id = photo_row["id"]
        self._path = photo_row["path"]
        self._mtime = photo_row["mtime"]
        self._orig_w = photo_row["width"] or 1
        self._orig_h = photo_row["height"] or 1
        self._faces = faces  # list of (x, y, w, h, color, name)
        self._tag_colors = tag_colors
        self._show_face_boxes = show_face_boxes
        self._aspect = aspect
        self._selected = False

        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(180, 240 if aspect == "3:4" else 200)
        self.setStyleSheet(
            "PhotoCard { background-color: #26282c; border-radius: 8px; }"
        )
        self._pixmap = get_thumbnail(self._path, self._mtime)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.RightButton:
            self.right_clicked.emit(self.photo_id)
        else:
            self.clicked.emit(self.photo_id)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -26)

        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            x = rect.x() + (rect.width() - scaled.width()) // 2
            y = rect.y() + (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            if self._show_face_boxes and self._faces:
                thumb_w, thumb_h = scaled.width(), scaled.height()
                for fx, fy, fw, fh, color, name in self._faces:
                    bx, by, bw, bh = scale_bbox_to_thumb(
                        fx, fy, fw, fh, self._orig_w, self._orig_h, thumb_w, thumb_h
                    )
                    pen = QPen(QColor(color))
                    pen.setWidth(2)
                    painter.setPen(pen)
                    painter.setBrush(QColor(color).lighter(150))
                    painter.setOpacity(0.15)
                    painter.drawRect(QRectF(x + bx, y + by, bw, bh))
                    painter.setOpacity(1.0)
                    painter.drawRect(QRectF(x + bx, y + by, bw, bh))
        else:
            painter.fillRect(rect, QColor("#33353a"))
            painter.setPen(QColor("#888"))
            painter.drawText(rect, Qt.AlignCenter, "no preview")

        # tag dots
        if self._tag_colors:
            dot_y = rect.bottom() - 10
            dot_x = rect.right() - 10
            for color in self._tag_colors[:4]:
                painter.setBrush(QColor(color))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(dot_x - 6, dot_y - 6, 8, 8)
                dot_x -= 12

        if self._selected:
            pen = QPen(QColor("#4a7dfc"))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)

        # face names below the thumbnail
        names = ", ".join(sorted({n for *_, n in self._faces if n})) if self._faces else ""
        painter.setPen(QColor("#c0c2c7"))
        painter.drawText(
            self.rect().adjusted(4, 0, -4, -4), Qt.AlignBottom | Qt.AlignLeft, names[:40]
        )
