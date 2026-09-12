"""A single dark stylesheet for the whole app - no per-widget styling."""

STYLESHEET = """
QWidget {
    background-color: #1e1f22;
    color: #dcdde1;
    font-family: -apple-system, "Segoe UI", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog { background-color: #1e1f22; }
QScrollArea { border: none; }
QToolBar {
    background-color: #26282c;
    border: none;
    padding: 6px;
    spacing: 8px;
}
QPushButton {
    background-color: #33353a;
    border: 1px solid #45474d;
    border-radius: 6px;
    padding: 6px 14px;
}
QPushButton:hover { background-color: #3d3f45; }
QPushButton:pressed { background-color: #2a2c30; }
QPushButton:disabled { color: #6b6d72; }
QPushButton#primary {
    background-color: #4a7dfc;
    border: none;
    color: white;
    font-weight: 600;
}
QPushButton#primary:hover { background-color: #5c8bff; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #2a2c30;
    border: 1px solid #45474d;
    border-radius: 4px;
    padding: 4px 6px;
}
QListWidget {
    background-color: #26282c;
    border: none;
    border-radius: 6px;
}
QListWidget::item { padding: 6px; border-radius: 4px; }
QListWidget::item:selected { background-color: #3a5fd9; }
QListWidget::item:hover:!selected { background-color: #2f3136; }
QScrollBar:vertical {
    background: transparent;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #45474d;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QProgressBar {
    border: 1px solid #45474d;
    border-radius: 4px;
    text-align: center;
    background-color: #26282c;
}
QProgressBar::chunk { background-color: #4a7dfc; border-radius: 4px; }
QLabel#sectionHeader {
    color: #9a9ca1;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    padding: 4px 2px;
}
QMenu {
    background-color: #2a2c30;
    border: 1px solid #45474d;
}
QMenu::item:selected { background-color: #3a5fd9; }
"""
