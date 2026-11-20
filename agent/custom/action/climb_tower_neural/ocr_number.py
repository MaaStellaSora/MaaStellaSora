"""解析本应只显示一个数值的区域，供运行决策和采集共用。"""
import re


DECIMAL_AMOUNT = re.compile(r"(?:[0-9]+|[1-9][0-9]{0,2}(?:,[0-9]{3})+)")


def decimal_amount(text):
    text = text.strip()
    return int(text.replace(",", "")) if DECIMAL_AMOUNT.fullmatch(text) else None


def assess_amount(accepted, observed):
    """accepted 是通过节点筛选的原文；observed 包含筛选前的原文和置信度。"""
    values = [decimal_amount(text) for text in accepted]
    trusted = [decimal_amount(text) for text, score in observed if score >= .85 and text.strip()]
    numbers = set(values + trusted) - {None}
    if len(numbers) > 1:
        return None, "conflict", tuple(sorted(numbers))
    if None in values or None in trusted:
        return None, "invalid", ()
    if not values:
        return None, "missing", ()
    return values[0], "ok", ()


def assess_ocr_number(result):
    if result is None:
        return None, "missing", ()

    return assess_amount(
        (row.text for row in result.filtered_results if row.score >= .85) if result.hit else (),
        ((row.text, row.score) for row in result.all_results),
    )
