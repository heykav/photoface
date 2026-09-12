from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect

from gui.theme import BG_CARD, BORDER, TEXT, TEXT_MUTED
from gui.thumbnails import get_thumbnail, scale_bbox_to_thumb

RADIUS = 12


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
        self._hovered = False

        self.setAttribute(Qt.WA_Hover, True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(190, 250 if aspect == "3:4" else 210)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(0, 0, 0, 110))
        self.setGraphicsEffect(shadow)

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

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(outer, RADIUS, RADIUS)
        painter.setClipPath(path)
        painter.fillPath(path, QColor(BG_CARD))

        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                outer.size().toSize(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            x = outer.x() + (outer.width() - scaled.width()) / 2
            y = outer.y() + (outer.height() - scaled.height()) / 2
            painter.drawPixmap(int(x), int(y), scaled)

            if self._show_face_boxes and self._faces:
                thumb_w, thumb_h = scaled.width(), scaled.height()
                for fx, fy, fw, fh, color, _name in self._faces:
                    bx, by, bw, bh = scale_bbox_to_thumb(
                        fx, fy, fw, fh, self._orig_w, self._orig_h, thumb_w, thumb_h
                    )
                    box_path = QPainterPath()
                    box_path.addRoundedRect(QRectF(x + bx, y + by, bw, bh), 4, 4)
                    pen = QPen(QColor(color))
                    pen.setWidthF(2.0)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(box_path)

            names = ", ".join(dict.fromkeys(n for *_, n in self._faces if n))
            if names or self._tag_colors:
                scrim = QLinearGradient(QPointF(0, outer.bottom() - 56), QPointF(0, outer.bottom()))
                scrim.setColorAt(0.0, QColor(0, 0, 0, 0))
                scrim.setColorAt(1.0, QColor(0, 0, 0, 165))
                painter.fillRect(QRectF(outer.x(), outer.bottom() - 56, outer.width(), 56), scrim)

                if names:
                    font = QFont(self.font())
                    # Scale whichever size unit the base font actually uses -
                    # pointSizeF() reports -1 when a widget's font was set by
                    # pixel size instead (as it is under some platform/style
                    # combinations), and setPointSizeF() on a negative value
                    # is a silent no-op that leaves Qt logging a warning.
                    if font.pointSizeF() > 0:
                        font.setPointSizeF(font.pointSizeF() * 0.92)
                    else:
                        font.setPixelSize(max(1, round(font.pixelSize() * 0.92)))
                    font.setWeight(QFont.DemiBold)
                    painter.setFont(font)
                    painter.setPen(QColor("white"))
                    text_rect = QRectF(outer.x() + 10, outer.bottom() - 26, outer.width() - 20, 18)
                    metrics = painter.fontMetrics()
                    elided = metrics.elidedText(names, Qt.ElideRight, int(text_rect.width()))
                    painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, elided)

                if self._tag_colors:
                    dot_y = outer.bottom() - 12
                    dot_x = outer.right() - 12
                    painter.setPen(Qt.NoPen)
                    for color in self._tag_colors[:4]:
                        painter.setBrush(QColor(color))
                        painter.drawEllipse(QPointF(dot_x, dot_y), 4, 4)
                        dot_x -= 11
        else:
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(outer, Qt.AlignCenter, "No preview")

        painter.setClipping(False)
        border_color = QColor("#5b8cff") if self._selected else QColor(BORDER)
        border_width = 2.0 if self._selected else 1.0
        if self._hovered and not self._selected:
            border_color = QColor("#4a4c53")
        pen = QPen(border_color)
        pen.setWidthF(border_width)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
