"""Цвета статусов и оформление окна."""

from sirius_proj.domain.models import Status

COLORS = {
    Status.COMPLIANT: "#15785d",
    Status.NON_COMPLIANT: "#b23a3a",
    Status.INSUFFICIENT: "#946518",
}

STYLE = """
QMainWindow { background: #f3f6fa; }
QWidget { font-family: 'DejaVu Sans'; font-size: 13px; color: #1d3449; }
QFrame#card { background: white; border: 1px solid #dce4ed; border-radius: 10px; }
QLabel#title { font-size: 26px; font-weight: 700; }
QLabel#subtitle { color: #607588; }
QLabel#eyebrow { color: #14836c; font-size: 11px; font-weight: 700; }
QPushButton { background: #e8eef5; border: 0; border-radius: 6px; padding: 10px 16px; }
QPushButton:hover { background: #d8e5ef; }
QPushButton#primary { background: #176f63; color: white; font-weight: 700; }
QPushButton#primary:hover { background: #105a50; }
QPushButton:disabled { background: #e3e8ee; color: #8996a3; }
QLineEdit, QComboBox { background: white; border: 1px solid #ccd7e3; border-radius: 5px;
                      padding: 8px; selection-background-color: #176f63; }
QTableWidget { background: white; border: 1px solid #dce4ed; gridline-color: #e9eef4;
               selection-background-color: #e1f2ed; selection-color: #123f38; }
QHeaderView::section { background: #edf2f7; border: 0; padding: 10px; font-weight: 700; }
QPlainTextEdit { background: white; border: 1px solid #dce4ed; border-radius: 6px; padding: 8px; }
QProgressBar { border: 0; background: #e1e8f0; border-radius: 4px; height: 8px; }
QProgressBar::chunk { background: #218b77; border-radius: 4px; }
"""
