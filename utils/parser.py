def to_float(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).replace(",", "").replace("£", "").strip()

    if value == "":
        return None

    return float(value)