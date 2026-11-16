"""Локальная проверка технических документов по чек-листу."""


def main() -> None:
    from .cli import main as entrypoint

    raise SystemExit(entrypoint())
