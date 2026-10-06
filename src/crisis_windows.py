"""Crisis-window definitions -- single source of truth for the periodization in
Project_Proposal.md §4. Both the by-window topic-model/NER notebook and any
later crisis-window analysis import from here, so a date fix only needs to
happen in one place.

`end=None` means "open-ended, runs to the end of the corpus" -- resolved against
the actual data at runtime (see `resolve_windows()`), not hardcoded to a future
date that would go stale.

An `end` date is INCLUSIVE of that whole day (resolved to 23:59:59.999999), so
consecutive windows are defined on adjacent calendar days -- `end="2020-02-29"` /
`start="2020-03-01"` -- with no gap and no overlap. The windows partition the
timeline: every post falls in exactly one window, which `resolve_windows()` enforces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class CrisisWindow:
    name: str
    start: str  # "YYYY-MM-DD"
    end: str | None  # "YYYY-MM-DD" or None for open-ended (resolved at runtime)
    description: str


# Dates are placeholders reflecting Project_Proposal.md §4's approximate windows,
# anchored to specific, citable event boundaries where one exists. VERIFY before
# treating results as final -- see the module docstring.
CRISIS_WINDOWS: list[CrisisWindow] = [
    CrisisWindow(
        name="baseline",
        start="2018-08-01",  # Greta Thunberg's first school strike (20 Aug 2018)
        end="2020-02-29",
        description="Baseline: Gründungsphase FFF.",
        #content = "Kohleausstieg, Schulstreik, Zeugnisse für Politiker (DE), nachhaltige Zukunft, Klimanotstand (Ö)"
    ),
    CrisisWindow(
        name="covid",
        start="2020-03-01",  # WHO pandemic declaration 11 Mar 2020; first DE lockdown mid-March
        end="2022-02-23",
        description="COVID-19 Pandemic: Lockdown, Netzstreik, Systemrelevanz: schnelles Handeln der Politik bei Corona, nicht aber bei Klima.",
        #content = "Livestreams, FFF Krisenbewegung umstellung während Covis, Aufrufe zum Mitmachen, Engagement junger leute während Covid (Masken) und Solidarität, Verknüpfung zuu Masken wegen Smog, Kohleausstieg, Waldbrand (Ö)"

    ),
    CrisisWindow(
        name="ukraine_energy",
        start="2022-02-24",  # Russian invasion of Ukraine
        end="2022-12-31",
        description="Ukraine War: Versorgungssicherheit, geopolitische Abhängigkeitsdebatte, Preisbremsen Politik (soziales Problem).",
        #content = "Angst und Hoffnung bei Nachrichten, Wahlen und Klimademos, Aufruf zum Mitmachen, Lützerath, Kapitalismus Kritik"
    ),
    CrisisWindow(
        name="cost_of_living",
        start="2023-01-01",
        end="2023-10-06",  # day before the Gaza war onset -- windows must not overlap
        description="Cost-of-living crisis: Heizungsgesetz Debatten, Klimaschutz vs. Bezahlbarkeit, soziale Gerechtigkeit (Mau: ?",
        #content = "Globaler Klimatreik, Solidarität & Emanzipation, Porträts von Aktivist*innen, Klimaschutz Klage"
    ),

    CrisisWindow(
        name="gaza",
        start="2023-10-07",  # Onset of the Gaza war
        end="2024-12-15",
        description="Gaza War: Solidaritätsdebatte, internationale FFF vs. deutsche FFF Positionierung (Geschlossenheit?).",
        #content = "Verkehrschaos Berlin, gegen AfD, Verkehrswende, anti Gas, Beteiligung Plenum"
    ),

    CrisisWindow(
        name="politics",
        start="2024-12-16",  # Failed Vertrauensfrage (16 Dec 2024) -> snap federal election Feb 2025
        end=None,
        description="German Politics: nach gescheiterter Vertrauensfrage, vorgezogene Wahlen & neue Regierungsbildung in Deutschland.",
        #content = "Verkehrschaos Berlin, gegen AfD, Verkehrswende, anti Gas, Beteiligung Plenum"
    ) 
    
]


def resolve_windows(
    windows: list[CrisisWindow], data_max_date: pd.Timestamp
) -> list[tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    """Resolve open-ended (end=None) windows against the actual corpus's max date.

    Returns a list of (name, start_ts, end_ts, description) tuples, timezone-aware
    UTC to match creation_time in the FFF data.
    """
    resolved = []
    for w in windows:
        start_ts = pd.Timestamp(w.start, tz="UTC")
        # `end` is a calendar date meant to include that whole day. pd.Timestamp("2020-02-29")
        # is midnight at the START of that day, which would drop every post published later
        # that day into a gap between windows -- push it to the last microsecond instead.
        end_ts = (
            pd.Timestamp(w.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            if w.end else data_max_date
        )
        resolved.append((w.name, start_ts, end_ts, w.description))

    for (name_a, _start_a, end_a, _d_a), (name_b, start_b, _end_b, _d_b) in zip(resolved, resolved[1:]):
        if start_b <= end_a:
            raise ValueError(
                f"crisis windows '{name_a}' (ends {end_a.date()}) and '{name_b}' (starts {start_b.date()}) "
                f"overlap -- end the earlier one the day before the later one starts."
            )
    return resolved


def filter_by_window(
    df: pd.DataFrame, window: tuple[str, pd.Timestamp, pd.Timestamp, str],
    date_col: str = "creation_time",
) -> pd.DataFrame:
    """Subset df to rows falling within one resolved window (inclusive both ends)."""
    _, start_ts, end_ts, _ = window
    mask = (df[date_col] >= start_ts) & (df[date_col] <= end_ts)
    return df.loc[mask]
