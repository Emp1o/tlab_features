"""Пропуск тестов, которым нужны локальные файлы из data/ (каталог в .gitignore)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_TAIL = (
    ": скачайте данные и запустите пайплайн, см. README, раздел запуска"
)


def missing_local_files(*relative_paths: str) -> list[str]:
    """Пути относительно корня репозитория, которых нет на диске."""
    missing = []
    for relative in relative_paths:
        rel = Path(relative)
        full = rel if rel.is_absolute() else ROOT / rel
        shown = rel.as_posix()
        if not shown.startswith("data/"):
            shown = f"data/{shown}"
        if not full.is_file():
            missing.append(shown)
    return missing


def require_data(*relative_paths: str) -> None:
    """Пропустить тест, если нет хотя бы одного входного файла."""
    missing = missing_local_files(*relative_paths)
    if missing:
        pytest.skip("нет " + ", ".join(missing) + SKIP_TAIL)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_data(*paths): пропуск, если нет локальных файлов в data/",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        paths = []
        for mark in item.iter_markers("needs_data"):
            paths.extend(mark.args)
        if not paths:
            continue
        missing = missing_local_files(*paths)
        if missing:
            item.add_marker(pytest.mark.skip(reason="нет " + ", ".join(missing) + SKIP_TAIL))
