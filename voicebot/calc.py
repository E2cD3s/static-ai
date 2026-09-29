"""Exact answers a 4B model gets wrong: arithmetic, unit conversions, the time somewhere else.

"what's 17 times 23", "15% of 80", "square root of 144", "convert 5 miles to km", "72 f in celsius",
"what time is it in Tokyo". Found with patterns (no LLM call), worked out in Python, and handed to the model as a
[Calculator ...] note for that one turn - it says the answer in its own words. Anything unclear returns None and
the turn goes on as before (search check etc.).
"""
from __future__ import annotations

import ast
import math
import operator
import re
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------- arithmetic

_CUE = re.compile(r"\b(?:what(?:'s| is| are)|whats|calculate|compute|how much is|solve|equals?|work out|do the math)\b|[=?]", re.I)
_WORDS = [  # spoken maths -> symbols (order matters: longer phrases first)
    (r"\bsquare roots? of\b", " sqrt "), (r"\bcube roots? of\b", " cbrt "),
    (r"\bto the power of\b|\bto the\b(?=\s*\d+(?:st|nd|rd|th)?\s*power)|\braised to\b", " ** "),
    (r"(?<=\d)(?:st|nd|rd|th)?\s*power\b", ""),
    (r"\bmultiplied by\b|\btimes\b|(?<=\d)\s*x\s*(?=\d)|×", " * "),
    (r"\bdivided by\b|\bover\b|÷", " / "),
    (r"\bplus\b|\band\b(?=\s*\d)", " + "), (r"\bminus\b|\bsubtract\b|−", " - "),
    (r"\bsquared\b", " ** 2 "), (r"\bcubed\b", " ** 3 "),
    (r"\bpercent of\b|% of\b", " % of "), (r"\bpercent\b", " % "),
]
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mil": 1e6, "billion": 1e9, "trillion": 1e12}
_NUM = r"\d+(?:\.\d+)?"
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise ValueError("too big")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("sqrt", "cbrt") \
            and len(node.args) == 1:
        x = _eval(node.args[0])
        return math.sqrt(x) if node.func.id == "sqrt" else math.copysign(abs(x) ** (1 / 3), x)
    raise ValueError("not arithmetic")


def _fmt(x: float) -> str:
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        x = int(x)
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:,.6g}" if abs(x) >= 1e-4 else f"{x:.3g}"


def arithmetic(text: str) -> str | None:
    """'what's 17 times 23' -> '17 × 23 = 391'. None unless it's clearly a sum."""
    if not _CUE.search(text):
        return None
    s = text.lower().replace(",", "")
    s = re.sub(rf"({_NUM})\s*({'|'.join(_SCALE)})\b", lambda m: _fmt(float(m[1]) * _SCALE[m[2]]).replace(",", ""), s)
    for pat, rep in _WORDS:
        s = re.sub(pat, rep, s)
    s = re.sub(rf"({_NUM})\s*% of\s*", r"\1 / 100 * ", s)  # 15% of 80
    s = re.sub(rf"({_NUM})\s*%", r"(\1 / 100)", s)
    s = re.sub(r"\b(sqrt|cbrt)\s*(" + _NUM + r")", r"\1(\2)", s)
    best = None
    for m in re.finditer(r"[\d.()\s+\-*/a-z]*", s):  # the longest run of maths in the sentence
        cand = re.sub(r"\b(?!sqrt|cbrt)[a-z]+\b", " ", m.group()).strip(" +-*/")
        cand = re.sub(r"^\(\s*\)|\(\s*\)$", "", cand).strip()
        if best is None or len(cand) > len(best):
            best = cand
    if not best or not re.search(r"\d", best):
        return None
    ops = len(re.findall(r"\*\*|[+\-*/]|sqrt|cbrt", best))
    if ops == 0 or len(re.findall(_NUM, best)) < (1 if "sqrt" in best or "cbrt" in best else 2):
        return None
    if re.fullmatch(r"\d+\s*/\s*\d+", best) and not re.search(r"divided|over|/\s*\d+\s*[?=]|what", text, re.I):
        return None  # "24/7", "10/10" - not a question
    try:
        value = _eval(ast.parse(best, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError, RecursionError):
        return None
    shown = (best.replace("**", "^").replace("*", "×").replace("/", "÷"))
    shown = re.sub(r"\s+", " ", shown)
    return f"{shown} = {_fmt(value)}"


# ---------------------------------------------------------------- units

_UNITS: dict[str, tuple[str, float]] = {}  # alias -> (dimension, factor to base unit)


def _add(dim: str, factor: float, *aliases: str) -> None:
    for a in aliases:
        _UNITS[a] = (dim, factor)


_add("length", 1e-3, "mm", "millimeter", "millimeters", "millimetre", "millimetres")
_add("length", 1e-2, "cm", "centimeter", "centimeters", "centimetre", "centimetres")
_add("length", 1, "m", "meter", "meters", "metre", "metres")
_add("length", 1e3, "km", "kilometer", "kilometers", "kilometre", "kilometres", "kms", "k")
_add("length", 0.0254, "in", "inch", "inches")
_add("length", 0.3048, "ft", "foot", "feet")
_add("length", 0.9144, "yd", "yard", "yards")
_add("length", 1609.344, "mi", "mile", "miles")
_add("mass", 1e-3, "g", "gram", "grams")
_add("mass", 1, "kg", "kilo", "kilos", "kilogram", "kilograms")
_add("mass", 0.45359237, "lb", "lbs", "pound", "pounds")
_add("mass", 0.028349523125, "oz", "ounce", "ounces")
_add("mass", 6.35029318, "st", "stone", "stones")
_add("volume", 1e-3, "ml", "milliliter", "milliliters", "millilitre", "millilitres")
_add("volume", 1, "l", "liter", "liters", "litre", "litres")
_add("volume", 3.785411784, "gal", "gallon", "gallons")
_add("volume", 0.2365882365, "cup", "cups")
_add("volume", 0.0295735295625, "fl oz", "fluid ounce", "fluid ounces")
_add("speed", 1, "km/h", "kmh", "kph", "kilometers per hour", "kilometres per hour")
_add("speed", 1.609344, "mph", "miles per hour")
_add("speed", 3.6, "m/s", "meters per second", "metres per second")
_add("data", 1, "mb", "megabyte", "megabytes")
_add("data", 1000, "gb", "gigabyte", "gigabytes")
_add("data", 1e6, "tb", "terabyte", "terabytes")
_TEMP = {"c": "C", "celsius": "C", "centigrade": "C", "f": "F", "fahrenheit": "F", "k": "K", "kelvin": "K"}
_TEMP_WORD = re.compile(r"\b(?:degrees?|celsius|fahrenheit|kelvin|centigrade)\b|°", re.I)
_UNIT_ALT = "|".join(sorted((re.escape(u) for u in [*_UNITS, *_TEMP, "degrees celsius", "degrees fahrenheit",
                                                       "degrees c", "degrees f"]), key=len, reverse=True))
_CONVERT = re.compile(
    rf"(?P<n>-?{_NUM})\s*(?:°\s*)?(?P<a>{_UNIT_ALT})\b\s*(?:(?:is|are|equals?)\s+)?(?:how many\s+)?"
    rf"(?:in(?:to)?|to|as|=)\s+(?:°\s*)?(?P<b>{_UNIT_ALT})\b|"
    rf"how many\s+(?P<b2>{_UNIT_ALT})\s+(?:are\s+)?(?:in|is|to)\s+(?:a\s+|an\s+|one\s+)?(?P<n2>-?{_NUM})?\s*(?:°\s*)?(?P<a2>{_UNIT_ALT})\b",
    re.I)


def _temp_unit(u: str) -> str | None:
    u = re.sub(r"^degrees?\s+", "", u.lower().strip())
    return _TEMP.get(u)


def convert(text: str) -> str | None:
    """'convert 5 miles to km' -> '5 mi = 8.04672 km'."""
    m = _CONVERT.search(text.replace(",", ""))
    if not m:
        return None
    n = float(m["n"] if m["n"] is not None else (m["n2"] or 1))
    a, b = (m["a"], m["b"]) if m["a"] else (m["a2"], m["b2"])
    a_l, b_l = a.lower(), b.lower()
    ta, tb = _temp_unit(a_l), _temp_unit(b_l)
    both_temp = ta and tb and (_TEMP_WORD.search(text) or (a_l in "cf" and b_l in "cf"))
    if both_temp:
        c = {"C": n, "F": (n - 32) * 5 / 9, "K": n - 273.15}[ta]
        out = {"C": c, "F": c * 9 / 5 + 32, "K": c + 273.15}[tb]
        return f"{_fmt(n)}°{ta} = {_fmt(round(out, 2))}°{tb}"
    ua, ub = _UNITS.get(a_l), _UNITS.get(b_l)
    if not ua or not ub or ua[0] != ub[0] or a_l == b_l:
        return None
    return f"{_fmt(n)} {a} = {_fmt(n * ua[1] / ub[1])} {b}"


# ---------------------------------------------------------------- time zones

_ZONES = {
    "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo", "seoul": "Asia/Seoul", "korea": "Asia/Seoul",
    "south korea": "Asia/Seoul", "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai", "china": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong", "taiwan": "Asia/Taipei", "taipei": "Asia/Taipei", "singapore": "Asia/Singapore",
    "manila": "Asia/Manila", "philippines": "Asia/Manila", "bangkok": "Asia/Bangkok", "thailand": "Asia/Bangkok",
    "vietnam": "Asia/Ho_Chi_Minh", "jakarta": "Asia/Jakarta", "indonesia": "Asia/Jakarta", "india": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata", "dubai": "Asia/Dubai",
    "pakistan": "Asia/Karachi", "israel": "Asia/Jerusalem", "turkey": "Europe/Istanbul", "istanbul": "Europe/Istanbul",
    "moscow": "Europe/Moscow", "russia": "Europe/Moscow", "london": "Europe/London", "uk": "Europe/London",
    "england": "Europe/London", "britain": "Europe/London", "scotland": "Europe/London", "ireland": "Europe/Dublin",
    "dublin": "Europe/Dublin", "paris": "Europe/Paris", "france": "Europe/Paris", "berlin": "Europe/Berlin",
    "germany": "Europe/Berlin", "amsterdam": "Europe/Amsterdam", "netherlands": "Europe/Amsterdam",
    "madrid": "Europe/Madrid", "spain": "Europe/Madrid", "rome": "Europe/Rome", "italy": "Europe/Rome",
    "poland": "Europe/Warsaw", "sweden": "Europe/Stockholm", "norway": "Europe/Oslo", "finland": "Europe/Helsinki",
    "greece": "Europe/Athens", "ukraine": "Europe/Kyiv", "portugal": "Europe/Lisbon", "lisbon": "Europe/Lisbon",
    "egypt": "Africa/Cairo", "cairo": "Africa/Cairo", "south africa": "Africa/Johannesburg", "nigeria": "Africa/Lagos",
    "kenya": "Africa/Nairobi", "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
    "australia": "Australia/Sydney", "perth": "Australia/Perth", "brisbane": "Australia/Brisbane",
    "new zealand": "Pacific/Auckland", "auckland": "Pacific/Auckland", "hawaii": "Pacific/Honolulu",
    "honolulu": "Pacific/Honolulu", "alaska": "America/Anchorage", "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles", "california": "America/Los_Angeles", "seattle": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles", "vegas": "America/Los_Angeles", "las vegas": "America/Los_Angeles",
    "pacific": "America/Los_Angeles", "pst": "America/Los_Angeles", "pdt": "America/Los_Angeles",
    "denver": "America/Denver", "mountain": "America/Denver", "mst": "America/Denver", "arizona": "America/Phoenix",
    "phoenix": "America/Phoenix", "chicago": "America/Chicago", "texas": "America/Chicago", "dallas": "America/Chicago",
    "houston": "America/Chicago", "central": "America/Chicago", "cst": "America/Chicago", "new york": "America/New_York",
    "nyc": "America/New_York", "florida": "America/New_York", "miami": "America/New_York", "boston": "America/New_York",
    "atlanta": "America/New_York", "eastern": "America/New_York", "est": "America/New_York", "edt": "America/New_York",
    "toronto": "America/Toronto", "vancouver": "America/Vancouver", "canada": "America/Toronto",
    "mexico": "America/Mexico_City", "mexico city": "America/Mexico_City", "brazil": "America/Sao_Paulo",
    "sao paulo": "America/Sao_Paulo", "argentina": "America/Argentina/Buenos_Aires",
    "buenos aires": "America/Argentina/Buenos_Aires", "chile": "America/Santiago", "colombia": "America/Bogota",
    "peru": "America/Lima", "utc": "UTC", "gmt": "UTC",
}
_ZONE_ALT = "|".join(sorted((re.escape(z) for z in _ZONES), key=len, reverse=True))
_TIME_IN = re.compile(rf"\btime(?:\s+is\s+it)?\s+(?:in|over in|for)\s+(?:the\s+)?(?P<z>{_ZONE_ALT})\b", re.I)


def zone_time(text: str) -> str | None:
    """'what time is it in Tokyo' -> 'In Tokyo it's 2:14 AM on Saturday'."""
    m = _TIME_IN.search(text)
    if not m:
        return None
    place = m["z"]
    now = datetime.now(ZoneInfo(_ZONES[place.lower()]))
    return f"in {place.title() if len(place) > 3 else place.upper()} it's {now:%-I:%M %p} on {now:%A}"


def note(text: str) -> str | None:
    """The [Calculator ...] note for a message, or None."""
    for fn in (zone_time, convert, arithmetic):
        try:
            found = fn(text)
        except Exception:  # noqa: BLE001 - a helper, never a reason for a turn to fail
            found = None
        if found:
            return (f"[Calculator - exact, trust it over your own math: {found}. Give the answer in digits "
                    "exactly as written here (round long decimals), and don't mention a calculator.]")
    return None
