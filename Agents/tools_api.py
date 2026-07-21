"""Remote tools — these call real public APIs. No API keys, no signup.

The important teaching point: to the model, these look exactly like the local
tools. It has no idea one is a dice roll and the other is an HTTPS request. All
it sees is a name, a schema, and a docstring.

The other point: tools fail. Networks time out, cities are not found, an API
returns no data. Always return a clear error STRING rather than raising — the
model reads that string and can recover, apologise, or try something else. A
raised exception just kills the turn.
"""

from datetime import datetime

import requests
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from strands import tool

TIMEOUT = 15

# Open-Meteo returns a numeric WMO weather code rather than text.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "slight rain showers", 81: "moderate rain showers",
    82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city, anywhere in the world.

    Args:
        city: City name, e.g. "Hyderabad", "London", "San Francisco".

    Returns:
        Current temperature, conditions, humidity and wind speed.
    """
    try:
        # Step 1: city name -> latitude/longitude (geocoding).
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=TIMEOUT,
        ).json()

        if not geo.get("results"):
            return f"Could not find a city named {city!r}."

        place = geo["results"][0]
        label = ", ".join(
            filter(None, [place["name"], place.get("admin1"), place.get("country")])
        )

        # Step 2: lat/lon -> current conditions.
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            },
            timeout=TIMEOUT,
        ).json()["current"]

        conditions = WEATHER_CODES.get(weather["weather_code"], "unknown conditions")
        return (
            f"Weather in {label}: {weather['temperature_2m']}°C, {conditions}, "
            f"humidity {weather['relative_humidity_2m']}%, "
            f"wind {weather['wind_speed_10m']} km/h."
        )
    except requests.RequestException as exc:
        return f"Weather service unreachable: {exc}"
    except (KeyError, ValueError) as exc:
        return f"Unexpected response from the weather service: {exc}"


@tool
def get_exchange_rate(from_currency: str, to_currency: str, amount: float = 1.0) -> str:
    """Convert an amount from one currency to another at today's rate.

    Args:
        from_currency: Three-letter source currency code, e.g. "USD".
        to_currency: Three-letter target currency code, e.g. "INR".
        amount: How much to convert. Defaults to 1.

    Returns:
        The converted amount and the rate used.
    """
    source = from_currency.upper()
    target = to_currency.upper()

    try:
        data = requests.get(
            f"https://open.er-api.com/v6/latest/{source}", timeout=TIMEOUT
        ).json()

        if data.get("result") != "success":
            return f"Unknown currency code {source!r}."

        rate = data["rates"].get(target)
        if rate is None:
            return f"Unknown currency code {target!r}."

        return (
            f"{amount:,.2f} {source} = {amount * rate:,.2f} {target} "
            f"(rate: 1 {source} = {rate} {target}, updated {data['time_last_update_utc']})"
        )
    except requests.RequestException as exc:
        return f"Exchange rate service unreachable: {exc}"
    except (KeyError, ValueError) as exc:
        return f"Unexpected response from the exchange rate service: {exc}"


@tool
def get_public_holidays(country_code: str, year: int = 0) -> str:
    """List the public holidays for a country in a given year.

    Coverage is mostly Europe and the Americas. Not every country has data.

    Args:
        country_code: Two-letter ISO country code, e.g. "US", "GB", "DE".
        year: Four-digit year. Defaults to the current year.

    Returns:
        The list of holidays with their dates, or a message if none are available.
    """
    code = country_code.upper()
    year = year or datetime.now().year

    try:
        response = requests.get(
            f"https://date.nager.at/api/v3/PublicHolidays/{year}/{code}",
            timeout=TIMEOUT,
        )

        # This API answers 204 No Content for countries it has no data for.
        if response.status_code == 204 or not response.content:
            return f"No holiday data available for country code {code!r} in {year}."
        if response.status_code == 404:
            return f"{code!r} is not a recognised country code."
        response.raise_for_status()

        holidays = response.json()
        lines = [f"{h['date']}: {h['name']}" for h in holidays]
        return f"Public holidays in {code} for {year} ({len(lines)} total):\n" + "\n".join(lines)
    except requests.RequestException as exc:
        return f"Holiday service unreachable: {exc}"
    except ValueError as exc:
        return f"Unexpected response from the holiday service: {exc}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return the top results with a short summary of each.

    Use this for anything you do not already know, anything that happened
    recently, or any fact that may have changed since your training data.
    Prefer searching over guessing.

    Args:
        query: What to search for. Write it as you would type it into a search
            box — keywords work better than a full sentence.
        max_results: How many results to return, between 1 and 10.

    Returns:
        Numbered results, each with a title, a summary and a URL.
    """
    max_results = max(1, min(max_results, 10))

    try:
        results = DDGS().text(query, max_results=max_results)
    except RatelimitException:
        return (
            "Search is rate limited right now. Wait a few seconds and try again, "
            "or answer from what you already know and say that you could not search."
        )
    except TimeoutException:
        return "Search timed out. Try again with a shorter query."
    except DDGSException as exc:
        return f"Search failed: {exc}"

    if not results:
        return f"No results found for {query!r}."

    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(
            f"{i}. {item['title']}\n   {item['body']}\n   {item['href']}"
        )
    return f"Top {len(results)} results for {query!r}:\n\n" + "\n\n".join(lines)


@tool
def get_cat_fact() -> str:
    """Return one random fact about cats. Use when the user asks for a cat fact.

    Returns:
        A single cat fact as text.
    """
    try:
        return requests.get(
            "https://catfact.ninja/fact", timeout=TIMEOUT
        ).json()["fact"]
    except requests.RequestException as exc:
        return f"Cat fact service unreachable: {exc}"
    except (KeyError, ValueError) as exc:
        return f"Unexpected response from the cat fact service: {exc}"
