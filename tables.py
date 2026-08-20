"""Отрисовка markdown-таблиц в моноширинный вид.

Ни Telegram, ни Telegraph не поддерживают <table>, поэтому таблицы показываются
внутри <pre>. Просто склеить ячейки через «|» недостаточно: колонки разной
ширины разъезжаются и таблица становится нечитаемой.

Здесь колонки выравниваются по ширине и отделяются линией под заголовком.
Если таблица всё равно шире экрана (а на телефоне это 40-45 моноширинных
символов), она разворачивается в список «поле: значение» — так читается
при любой ширине, пусть и занимает больше места.
"""
COL_SEP = " │ "
LINE_SEP = "─"
CROSS = "┼"

TELEGRAM_WIDTH = 42  # практический предел до горизонтальной прокрутки на телефоне
TELEGRAPH_WIDTH = 70  # страница-читалка шире обычного сообщения


def parse_row(line: str) -> list[str]:
    """Разбирает строку markdown-таблицы на ячейки."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator(line: str) -> bool:
    """True для строки-разделителя вида |---|:---:|."""
    stripped = line.strip().strip("|").strip()
    if not stripped:
        return False
    return all(set(cell.strip()) <= set("-: ") and "-" in cell
               for cell in stripped.split("|"))


def _as_list(header: list[str], body: list[list[str]]) -> str:
    """Разворот широкой таблицы в список: первая колонка — заголовок пункта."""
    blocks = []
    for row in body:
        title = row[0].strip() or "—"
        lines = [f"▸ {title}"]
        for name, value in zip(header[1:], row[1:]):
            value = value.strip()
            if value:
                label = name.strip()
                lines.append(f"   {label + ': ' if label else ''}{value}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render(rows: list[list[str]], max_width: int = TELEGRAM_WIDTH) -> str:
    """Собирает выровненную таблицу либо список, если она слишком широкая."""
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""

    columns = max(len(r) for r in rows)
    rows = [r + [""] * (columns - len(r)) for r in rows]

    if columns == 1:
        return "\n".join(r[0] for r in rows)

    widths = [max(len(r[c]) for r in rows) for c in range(columns)]
    total = sum(widths) + len(COL_SEP) * (columns - 1)

    header, body = rows[0], rows[1:]
    if total > max_width and body:
        return _as_list(header, body)

    lines = [COL_SEP.join(cell.ljust(w) for cell, w in zip(header, widths)).rstrip()]
    # Разделитель обязан совпадать по ширине с колонками: между ними стоит
    # " │ " (три символа), поэтому в линии это "─┼─".
    lines.append((LINE_SEP + CROSS + LINE_SEP).join(LINE_SEP * w for w in widths))
    for row in body:
        lines.append(COL_SEP.join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def render_markdown_table(block: list[str], max_width: int = TELEGRAM_WIDTH) -> str:
    """Принимает строки markdown-таблицы (со строкой-разделителем) и рисует её."""
    rows = [parse_row(line) for line in block if not is_separator(line)]
    return render(rows, max_width)
