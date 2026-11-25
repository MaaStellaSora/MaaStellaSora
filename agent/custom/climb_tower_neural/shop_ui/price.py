"""Read one located payable amount after the shelf's ink-color filter."""
import re


AMOUNT = re.compile(r"\W*([0-9]+)\W*")


def price_key(name, quantity):
    if name == "potential_drink":
        return "potential_drink"
    if name.startswith("melody_of_"):
        return f"melody_{quantity}"
    return None


def payable_amount(rows, roi, standard_price):
    if standard_price is None:
        return -1, None
    allowed = {standard_price, standard_price // 2, standard_price * 4 // 5}
    amounts = []
    for row in rows:
        # Whole-ROI fallback is not a located number. Never remove internal
        # letters or split a concatenated original/sale pair into a price.
        if tuple(row.box) == tuple(roi):
            continue
        match = AMOUNT.fullmatch(row.text.strip())
        if match and int(match[1]) in allowed:
            amounts.append((int(match[1]), row))
    values = {value for value, _ in amounts}
    if len(values) != 1:
        return -1, None
    value = values.pop()
    located = max((row for amount, row in amounts if amount == value), key=lambda row: row.score)
    return value, list(located.box)
