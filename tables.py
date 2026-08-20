"""Отрисовка markdown-таблиц в моноширинный вид.

Ни Telegram, ни Telegraph не поддерживают <table>, поэтому таблицы показываются
внутри <pre>. Просто склеить ячейки через «|» недостаточно: колонки разной
ширины разъезжаются и таблица становится нечитаемой.

Здесь колонки выравниваются по ширине и отделяются линией под заголовком.
Если таблица шире экрана (а на телефоне это 40-45 моноширинных символов),
она разворачивается в список «поле: значение» — так читается при любой ширине.

Отдельная история — код в ячейках. Многострочная ячейка в markdown невозможна,
поэтому модель записывает переносы escape-последовательностями (\\n, \\t), и
без обработки пример кода приезжает одной нечитаемой строкой. Такой код
вынимается из ячейки и печатается настоящими строками, а сама таблица
принудительно разворачивается в список: выровнять многострочную ячейку нельзя.
"""
import re

COL_SEP = " │ "
LINE_SEP = "─"
CROSS = "┼"

TELEGRAM_WIDTH = 42  # практический предел до горизонтальной прокрутки на телефоне
TELEGRAPH_WIDTH = 70  # страница-читалка шире обычного сообщения

_ESCAPES = (("\\r\\n", "\n"), ("\\n", "\n"), ("\\t", "    "), ('\\"', '"'), ("\\'", "'"))
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.S)


def unescape(text: str) -> str:
    """Превращает литеральные \\n и \\t в настоящие переносы и отступы."""
    for source, target in _ESCAPES:
        text = text.replace(source, target)
    return text


def strip_inline(text: str) -> str:
    """Снимает markdown-выделение: в моноширинном блоке звёздочки видны как есть."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\1", text)
    return text.replace("`", "").strip()


def split_cell(cell: str) -> tuple[str, str]:
    """Делит ячейку на обычный текст и код из ``` ``` (уже с переносами)."""
    text = unescape(cell)
    blocks = []

    def take(match):
        blocks.append(match.group(1).strip("\n"))
        return " "

    rest = _FENCE_RE.sub(take, text)
    if not blocks and "\n" in rest and rest.count("\n") >= 2:
        # Код без ограждения: ячейка с переносами всё равно не выровняется
        return "", rest.strip("\n")
    return strip_inline(rest), "\n".join(blocks)


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


def _indent(block: str, prefix: str = "   ") -> list[str]:
    return [prefix + line for line in block.split("\n")]


def _as_list(header: list[tuple[str, str]], body: list[list[tuple[str, str]]]) -> str:
    """Разворот таблицы в список: первая колонка — заголовок пункта."""
    blocks = []
    for row in body:
        title, title_code = row[0]
        lines = [f"▸ {title or '—'}"]
        if title_code:
            lines.extend(_indent(title_code))
        for (name, _), (value, code) in zip(header[1:], row[1:]):
            label = name.strip()
            if value:
                lines.append(f"   {label + ': ' if label else ''}{value}")
            elif code and label:
                lines.append(f"   {label}:")
            if code:
                lines.extend(_indent(code, "     "))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render(rows: list[list[str]], max_width: int = TELEGRAM_WIDTH) -> str:
    """Собирает выровненную таблицу либо список, если она широкая или с кодом."""
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""

    columns = max(len(r) for r in rows)
    rows = [r + [""] * (columns - len(r)) for r in rows]
    prepared = [[split_cell(cell) for cell in row] for row in rows]

    if columns == 1:
        out = []
        for row in prepared:
            text, code = row[0]
            if text:
                out.append(text)
            if code:
                out.append(code)
        return "\n".join(out)

    header, body = prepared[0], prepared[1:]
    # Многострочную ячейку выровнять нельзя — такие таблицы всегда идут списком
    has_code = any(code for row in prepared for _, code in row)

    widths = [max(len(row[c][0]) for row in prepared) for c in range(columns)]
    total = sum(widths) + len(COL_SEP) * (columns - 1)

    if body and (has_code or total > max_width):
        return _as_list(header, body)

    lines = [COL_SEP.join(text.ljust(w) for (text, _), w in zip(header, widths)).rstrip()]
    # Разделитель обязан совпадать по ширине с колонками: между ними стоит
    # " │ " (три символа), поэтому в линии это "─┼─".
    lines.append((LINE_SEP + CROSS + LINE_SEP).join(LINE_SEP * w for w in widths))
    for row in body:
        lines.append(COL_SEP.join(text.ljust(w) for (text, _), w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def render_markdown_table(block: list[str], max_width: int = TELEGRAM_WIDTH) -> str:
    """Принимает строки markdown-таблицы (со строкой-разделителем) и рисует её."""
    rows = [parse_row(line) for line in block if not is_separator(line)]
    return render(rows, max_width)
