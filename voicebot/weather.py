"""Weather from Open-Meteo (free, no API key): "what's the weather in Denver", "is it going to rain tomorrow",
"how cold is it in Chicago tonight", "forecast for Tokyo on Saturday". Found with patterns (no LLM call), looked up
over HTTP (place -> coordinates -> forecast, ~0.3-0.6 s, cached 10 min), and handed to the model as a
[Weather ...] note for that turn, like calc.py. No place said -> weather.default_location, or the note asks where.
Anything that fails returns None and the turn carries on (the web search check can still pick it up).
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime

import httpx

log = logging.getLogger("voicebot.weather")

_ASK = re.compile(r"\b(?:what'?s|how'?s|what\s+is|how\s+is|check)\s+(?:the\s+)?weather\b"
                  r"|\bweather\s+(?:in|for|like|today|tomorrow|tonight|outside|forecast|report|this|on|at|out|gonna|going)\b"
                  r"|\bforecast\b|\b(?:is|will)\s+it\s+(?:going\s+to\s+|gonna\s+)?(?:rain|snow|storm|be\s+(?:hot|cold|warm|sunny|nice))\b"
                  r"|\bhow\s+(?:hot|cold|warm|chilly)\s+(?:is\s+it|will\s+it\s+be|is\s+it\s+going\s+to\s+be)\b"
                  r"|\b(?:temperature|temp)\s+(?:in|outside|today|tomorrow|tonight|right\s+now)\b", re.I)
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WHEN = re.compile(r"\b(today|tonight|tomorrow|this\s+weekend|(?:on\s+)?(?:" + "|".join(_DAYS) + r"))\b", re.I)
_PLACE = re.compile(r"\b(?:in|for|at|near|over\s+in|out\s+in)\s+(?P<p>[A-Za-z][A-Za-z .'-]{1,40}?(?:,\s*[A-Za-z][A-Za-z .'-]{1,30}?)?)"
                    r"(?=\s+(?:today|tonight|tomorrow|this|on|right|now|later|next|at|over|during)\b|[?.!,]|$)", re.I)
_NOT_PLACES = {"the morning", "the afternoon", "the evening", "the weekend", "general", "a bit", "a while", "here",
               "my area", "the area", "town", "the city", "celsius", "fahrenheit", "degrees", "f", "c"}

# WMO weather codes -> words
_CODES = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
          51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
          61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
          71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "rain showers", 81: "rain showers",
          82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
          96: "thunderstorms with hail", 99: "thunderstorms with hail"}


def wants(text: str) -> bool:
    return bool(_ASK.search(text))


def _place(text: str) -> str | None:
    for m in _PLACE.finditer(text):
        p = re.sub(r"^(?:the)\s+", "", m["p"].strip(" .'-"), flags=re.I)
        if p.lower() not in _NOT_PLACES and len(p) >= 2:
            return p
    return None


def _day(text: str, today: date) -> tuple[int, str]:
    """(days from today, how to say it)."""
    m = _WHEN.search(text)
    w = (m.group(1).lower().replace("on ", "").strip() if m else "today")
    if w in ("today", "tonight"):
        return 0, w
    if w == "tomorrow":
        return 1, "tomorrow"
    if "weekend" in w:
        return (5 - today.weekday()) % 7, "Saturday"
    ahead = (_DAYS.index(w) - today.weekday()) % 7
    return ahead, w.title()


class Weather:
    def __init__(self, cfg):
        self.cfg = cfg.get("weather") or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.imperial = str(self.cfg.get("units", "imperial")).lower() != "metric"
        self._http: httpx.AsyncClient | None = None
        self._geo: dict[str, dict | None] = {}
        self._cache: dict[tuple, tuple[float, dict]] = {}
        self.stats = {"lookups": 0, "errors": 0}

    async def _get(self, url: str, params: dict) -> dict:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=4.0, headers={"User-Agent": "Static voice bot"})
        r = await self._http.get(url, params=params)
        r.raise_for_status()
        return r.json()

    async def _locate(self, place: str) -> dict | None:
        key = place.lower()
        if key not in self._geo:
            name, _, region = place.partition(",")
            data = await self._get("https://geocoding-api.open-meteo.com/v1/search",
                                   {"name": name.strip(), "count": 5, "language": "en", "format": "json"})
            hits = data.get("results") or []
            region = region.strip().lower()
            if region:  # "Portland, Maine" / "Paris, Texas": prefer the matching state/country
                hits = [h for h in hits if region in f"{h.get('admin1', '')} {h.get('country', '')} {h.get('country_code', '')}".lower()] or hits
            self._geo[key] = hits[0] if hits else None
        return self._geo[key]

    async def note(self, text: str) -> str | None:
        """The [Weather ...] note for a message, or None if it isn't about the weather (or the lookup failed)."""
        if not self.enabled or not wants(text):
            return None
        place = _place(text) or str(self.cfg.get("default_location") or "").strip()
        if not place:
            return ("[Weather - they want the weather but didn't say where, and you have no home town set. "
                    "Ask them which city.]")
        try:
            loc = await self._locate(place)
            if loc is None:
                return f"[Weather - you couldn't find a place called \"{place}\". Ask them to say it another way.]"
            self.stats["lookups"] += 1
            key = (round(loc["latitude"], 2), round(loc["longitude"], 2), self.imperial)
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < 600:
                data = hit[1]
            else:
                deg, wind = ("fahrenheit", "mph") if self.imperial else ("celsius", "kmh")
                data = await self._get("https://api.open-meteo.com/v1/forecast", {
                    "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto", "forecast_days": 7,
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "temperature_unit": deg, "wind_speed_unit": wind})
                self._cache[key] = (time.time(), data)
        except Exception as e:  # noqa: BLE001 - a helper, never a reason for a turn to fail
            self.stats["errors"] += 1
            log.warning("Weather lookup for %r failed: %s", place, e)
            return None
        u, w = ("°F", "mph") if self.imperial else ("°C", "km/h")
        where = ", ".join(x for x in (loc.get("name"), loc.get("admin1") if loc.get("country_code") in ("US", "CA", "AU")
                                      else loc.get("country")) if x)
        local_today = datetime.fromisoformat(data["current"]["time"]).date()
        ahead, when = _day(text, local_today)
        d = data["daily"]
        i = min(ahead, len(d["time"]) - 1)
        day = (f"{when}: {_CODES.get(d['weather_code'][i], 'mixed')}, high {round(d['temperature_2m_max'][i])}{u}, "
               f"low {round(d['temperature_2m_min'][i])}{u}, {d['precipitation_probability_max'][i] or 0}% chance of rain")
        parts = [day]
        if ahead == 0:
            c = data["current"]
            parts.insert(0, f"right now {round(c['temperature_2m'])}{u} (feels like {round(c['apparent_temperature'])}{u}), "
                            f"{_CODES.get(c['weather_code'], 'mixed')}, wind {round(c['wind_speed_10m'])} {w}")
        log.info("🌤 %s: %s", where, "; ".join(parts))
        return (f"[Weather for {where} - live forecast, trust it: {'; '.join(parts)}. Say it in a sentence or two, "
                "in your own words, numbers as they are; only what they asked about.]")

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
