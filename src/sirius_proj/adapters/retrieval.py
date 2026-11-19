"""Группы для поиска. Абзацы сохраняют адреса, таблицы остаются целыми."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from itertools import groupby

from sirius_proj.adapters.embeddings import E5Encoder
from sirius_proj.adapters.search import HybridIndex
from sirius_proj.domain.models import Block

CHUNK_SIZE = 1400
OVERLAP = 200


def _split_paragraph(block: Block) -> Iterator[Block]:
    if len(block.text) <= CHUNK_SIZE:
        yield block
        return
    start, number = 0, 1
    while start < len(block.text):
        end = min(start + CHUNK_SIZE, len(block.text))
        if end < len(block.text):
            boundary = block.text.rfind(" ", start + 1000, end)
            if boundary > 0:
                end = boundary
        text = block.text[start:end].strip()
        if text:
            yield replace(block, id=f"{block.id}:c{number}", text=text)
        if end == len(block.text):
            break
        start = end - OVERLAP
        number += 1


def _text_groups(blocks: Iterator[Block]) -> Iterator[tuple[Block, ...]]:
    current: list[Block] = []
    size = 0
    for block in blocks:
        for part in _split_paragraph(block):
            if current and size + len(part.text) + 1 > CHUNK_SIZE:
                yield tuple(current)
                last = current[-1]
                current = [last] if len(last.text) <= OVERLAP and ":c" not in part.id else []
                size = sum(len(item.text) + 1 for item in current)
            current.append(part)
            size += len(part.text) + 1
    if current:
        yield tuple(current)


def search_groups(blocks: tuple[Block, ...]) -> Iterator[tuple[Block, ...]]:
    for (table, _, _), group in groupby(
        blocks, key=lambda b: (b.location.table, b.location.heading, b.location.page)
    ):
        if table is not None:
            yield tuple(group)
        else:
            yield from _text_groups(group)


def _search_text(group: tuple[Block, ...]) -> str:
    first = group[0]
    if first.table is not None:
        table = first.table
        return "\n".join(
            (
                first.location.heading,
                table.caption,
                " | ".join(table.headers),
                *(row.text for row in group),
            )
        )
    return "\n".join((first.location.heading, *(block.text for block in group)))


class SectionIndex:
    def __init__(
        self, blocks: tuple[Block, ...], encoder: E5Encoder, cancelled: Callable[[], bool]
    ) -> None:
        self.groups = []
        for group in search_groups(blocks):
            if cancelled():
                raise InterruptedError("Проверка отменена")
            self.groups.append(group)
        self.index = HybridIndex(tuple(map(_search_text, self.groups)), encoder)

    def retrieve(self, requirement: str) -> tuple[Block, ...]:
        return tuple(
            dict.fromkeys(
                block
                for position in self.index.rank(requirement)[:4]
                for block in self.groups[position]
            )
        )
