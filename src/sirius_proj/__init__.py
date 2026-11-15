"""Local technical document review."""


def main() -> None:
    from .cli import main as entrypoint

    raise SystemExit(entrypoint())
