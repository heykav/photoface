from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from analyzer import Analyzer


class AnalyzeWorker(QThread):
    progress = Signal(int, int, str)   # done, total, current_path
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, analyzer: Analyzer, folder: Path, parent=None):
        super().__init__(parent)
        self._analyzer = analyzer
        self._folder = folder
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            self._analyzer.analyze_folder(
                self._folder,
                progress=lambda done, total, path: self.progress.emit(done, total, path),
                cancel_check=lambda: self._cancelled,
            )
            if not self._cancelled:
                self._analyzer.recluster_all()
            self.finished_ok.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
