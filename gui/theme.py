"""A single stylesheet + palette constants for the whole app."""

# Palette - referenced by name from widgets that need to paint manually
# (PhotoCard, Gallery empty-state, Lightbox) rather than only via stylesheet.
BG = "#18191c"
BG_RAISED = "#212226"
BG_CARD = "#232427"
BORDER = "#33343a"
BORDER_SUBTLE = "#2a2b30"
TEXT = "#e7e8ea"
TEXT_MUTED = "#8c8e94"
TEXT_FAINT = "#5c5e64"
ACCENT = "#5b8cff"
ACCENT_HOVER = "#6f99ff"
ACCENT_MUTED = "#3a4a7a"

STYLESHEET = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: -apple-system, "SF Pro Text", "Segoe UI", sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{ background-color: {BG}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

#topBar {{
    background-color: {BG_RAISED};
    border-bottom: 1px solid {BORDER_SUBTLE};
}}
#statusBar {{
    background-color: {BG_RAISED};
    border-top: 1px solid {BORDER_SUBTLE};
    color: {TEXT_MUTED};
}}
#sidebarPanel {{
    background-color: {BG_RAISED};
    border-left: 1px solid {BORDER_SUBTLE};
}}

QPushButton {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 16px;
    color: {TEXT};
}}
QPushButton:hover {{ background-color: #2a2b30; border-color: #45474e; }}
QPushButton:pressed {{ background-color: #1c1d20; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; }}
QPushButton#primary {{
    background-color: {ACCENT};
    border: none;
    color: white;
    font-weight: 600;
}}
QPushButton#primary:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background-color: {ACCENT_MUTED}; color: #a8b6de; }}
QPushButton#ghost {{
    background-color: transparent;
    border: none;
    color: {TEXT_MUTED};
}}
QPushButton#ghost:hover {{ color: {TEXT}; background-color: {BORDER_SUBTLE}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {BG};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 10px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QLineEdit::placeholder {{ color: {TEXT_FAINT}; }}

QListWidget {{
    background-color: transparent;
    border: none;
    outline: none;
}}
QListWidget::item {{
    padding: 8px 10px;
    border-radius: 8px;
    margin: 1px 0px;
}}
QListWidget::item:selected {{ background-color: {ACCENT_MUTED}; color: {TEXT}; }}
QListWidget::item:hover:!selected {{ background-color: {BORDER_SUBTLE}; }}

QCheckBox {{ spacing: 8px; padding: 4px 2px; color: {TEXT_MUTED}; }}
QCheckBox:hover {{ color: {TEXT}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1.5px solid {BORDER};
    border-radius: 4px;
    background: {BG};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: #4a4c53; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QProgressBar {{
    border: none;
    border-radius: 5px;
    text-align: center;
    background-color: {BG};
    color: {TEXT_MUTED};
    height: 10px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 5px; }}

QLabel#sectionHeader {{
    color: {TEXT_FAINT};
    font-weight: 700;
    font-size: 10.5px;
    letter-spacing: 1px;
    padding: 10px 4px 6px 4px;
}}
QLabel#appTitle {{
    color: {TEXT};
    font-size: 15px;
    font-weight: 700;
}}
QLabel#photoCount {{
    color: {TEXT_MUTED};
}}

QMenu {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background-color: {ACCENT_MUTED}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

QToolTip {{
    background-color: {BG_CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 6px;
    border-radius: 6px;
}}
"""
