#!/usr/bin/env python3
"""Build a single self-contained interactive HTML dashboard covering all four scored axes
(DPM, subcategories, master frame, inequality arena) -- frame intensity over time, with
toggleable event-type markers (hover for title/description) and crisis-window shading.

Deliberately a single static HTML file with the data baked in as embedded JSON (not fetched
from separate files at load time): works whether opened directly (file://) or served via
GitHub Pages, and needs nothing beyond a browser plus the Plotly.js CDN script.

Run locally (pure pandas/json, no GPU/torch dependency -- unlike the notebooks, this runs
fine on the local Windows venv):

    python scripts/build_dashboard.py

Output: docs/index.html. Enable GitHub Pages once (repo Settings -> Pages -> Deploy from a
branch -> main, folder /docs) and the file is then reachable at
https://<username>.github.io/<repo>/ -- a plain link, nothing else to host or maintain.
Re-run this script and push whenever the underlying LLM scores change; the page has no
server-side component to update separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.key_events import KEY_EVENTS
from src.crisis_windows import CRISIS_WINDOWS, resolve_windows
from src.plot_style import EVENT_TYPE_LABELS

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = REPO_ROOT / "data" / "processed" / "FFF_german.csv"
OUTPUT_PATH = REPO_ROOT / "docs" / "index.html"

# All series are precomputed at every level below and embedded in the page, so the
# aggregation-level control in the dashboard is a free client-side switch (no recomputation,
# no fetch) -- same level set as notebooks/05_llm_scoring_analysis.ipynb's AGG_LEVELS.
AGG_LEVELS = {
    "weekly": "W",
    "monthly": "ME",
    "quarterly": "QE",
    "semiannual": "6ME",
    "yearly": "YE",
}
DEFAULT_AGG_LEVEL = "quarterly"

AXES = [
    {
        "key": "axis_a_dpm_de",
        "label": "DPM (Diagnostic / Prognostic / Motivational)",
        "description": "Diagnostic (naming a problem and assigning responsibility), Prognostic "
                        "(proposing solutions or demands), and Motivational (calls to action) -- "
                        "the three core framing functions identified by Benford & Snow (1988).",
        "scores_path": REPO_ROOT / "data" / "llm_scores" / "axis_a_dpm_de.csv",
        "concepts": ["diagnostic", "prognostic", "motivational"],
    },
    {
        "key": "axis_b_dpm_subcategories_de",
        "label": "DPM Subcategories",
        "description": "Finer-grained functions within each DPM category: blame attribution & "
                        "problem identification (diagnostic); solutions & tactics (prognostic); "
                        "rationale & call-to-action (motivational).",
        "scores_path": REPO_ROOT / "data" / "llm_scores" / "axis_b_dpm_subcategories_de.csv",
        "concepts": ["blame_attribution", "problem_identification", "solutions", "tactics", "rationale", "call_to_action"],
    },
    {
        "key": "axis_c_masterframe_de",
        "label": "Master Frame (Climate Change / Climate Justice)",
        "description": "Whether a post frames climate change primarily as a scientific/ecological "
                        "problem, or explicitly as a matter of justice, rights, and unequal "
                        "responsibility (della Porta & Parks).",
        "scores_path": REPO_ROOT / "data" / "llm_scores" / "axis_c_masterframe_de.csv",
        "concepts": ["climate_change", "climate_justice"],
    },
    {
        "key": "axis_d_arena_de",
        "label": "Inequality Arena",
        "description": "Which of Mau, Lux & Westheuser's (2023) societal inequality arenas "
                        "(top-bottom, inside-outside, us-them, today-tomorrow) a post's climate "
                        "framing extends to or invokes.",
        "scores_path": REPO_ROOT / "data" / "llm_scores" / "axis_d_arena_de.csv",
        "concepts": ["oben_unten", "innen_aussen", "wir_sie", "heute_morgen"],
    },
]


def build_axis_series(scores_path: Path, concepts: list[str], dates: pd.DataFrame) -> dict:
    """Return {level_name: {"dates": [...], "series": {concept: [...]}}} for every level in
    AGG_LEVELS, so the dashboard's aggregation-level control is just picking which precomputed
    series to plot, no client-side resampling."""
    scores = pd.read_csv(scores_path, dtype={"text_id": str})
    # Debiased point estimate: mean across orderings per post (a no-op for a single-ordering
    # run), same rule as notebooks/05_llm_scoring_analysis.ipynb's "Debiased Point Estimates".
    final_scores = scores.groupby("text_id")[concepts].mean()

    merged = final_scores.reset_index().merge(dates, left_on="text_id", right_on="id", how="left")
    merged = merged.dropna(subset=["creation_time"]).set_index("creation_time").sort_index()

    by_level = {}
    for level_name, freq in AGG_LEVELS.items():
        intensity_over_time = merged[concepts].resample(freq).mean()
        by_level[level_name] = {
            "dates": [d.strftime("%Y-%m-%d") for d in intensity_over_time.index],
            "series": {c: [None if pd.isna(v) else round(float(v), 4) for v in intensity_over_time[c]] for c in concepts},
        }
    return by_level


def main() -> None:
    dates = pd.read_csv(SOURCE_PATH, dtype={"id": str})[["id", "creation_time"]]
    dates["creation_time"] = pd.to_datetime(dates["creation_time"], utc=True)
    max_date = dates["creation_time"].max()

    axes_data = {}
    for axis in AXES:
        if not axis["scores_path"].exists():
            print(f"skipping {axis['key']} -- {axis['scores_path']} not found")
            continue
        by_level = build_axis_series(axis["scores_path"], axis["concepts"], dates)
        axes_data[axis["key"]] = {
            "label": axis["label"],
            "description": axis["description"],
            "concepts": axis["concepts"],
            "by_level": by_level,
        }
        bin_counts = ", ".join(f"{name}={len(v['dates'])}" for name, v in by_level.items())
        print(f"{axis['key']}: {bin_counts}")

    resolved_windows = resolve_windows(CRISIS_WINDOWS, max_date)
    crisis_windows_data = [
        {"name": name, "start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"), "description": description}
        for name, start, end, description in resolved_windows
    ]

    events_data = [
        {"date": e.date, "label": e.label, "topic": e.topic, "type": e.type, "highlight": e.highlight}
        for e in KEY_EVENTS
    ]

    event_types = sorted({e["type"] for e in events_data})

    payload = {
        "axes": axes_data,
        "axis_order": [a["key"] for a in AXES if a["key"] in axes_data],
        "agg_levels": list(AGG_LEVELS.keys()),
        "default_agg_level": DEFAULT_AGG_LEVEL,
        "crisis_windows": crisis_windows_data,
        "events": events_data,
        "event_types": event_types,
        "event_type_labels": EVENT_TYPE_LABELS,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Escape "<" -- if any event label/description ever contains the literal text
    # "</script>", embedding it unescaped would prematurely close the surrounding <script>
    # tag and break the page. < is valid JSON and decodes back to "<" at parse time.
    payload_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    html = TEMPLATE.replace("__PAYLOAD__", payload_json)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FFF Germany -- Frame Intensity Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { color-scheme: light; }
  body {
    margin: 0; padding: 20px 24px 44px; background: #ece4d5; color: #2b2620;
    font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  }
  .page { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 21px; margin: 0 0 10px; color: #3d342a; }
  p.intro { margin: 0 0 8px; color: #4a4136; max-width: 900px; }
  p.intro:last-of-type { margin-bottom: 20px; }
  .controls {
    display: flex; flex-wrap: wrap; gap: 22px 28px; align-items: flex-start;
    background: #f8f4e9; border: 1px solid #ddd2b8; border-radius: 8px;
    padding: 16px 20px; margin-bottom: 16px;
  }
  .control-group label.group-label { display: block; font-weight: 600; margin-bottom: 6px; font-size: 13px; color: #3d342a; }
  .control-group select { font-size: 14px; padding: 6px 8px; border-radius: 4px; border: 1px solid #b9ab86; min-width: 260px; background: #fff; }
  #axisDescription, #compareAxisDescription { margin-top: 8px; max-width: 380px; font-size: 12.5px; color: #5a5142; font-style: italic; }
  .checkbox-row { display: flex; flex-wrap: wrap; gap: 10px 16px; max-width: 520px; }
  .checkbox-row label { font-size: 13px; white-space: nowrap; cursor: pointer; }
  #plot-container { background: #f8f4e9; border: 1px solid #ddd2b8; border-radius: 8px; padding: 10px; }
</style>
</head>
<body>
<div class="page">

<h1>FFF Germany -- Frame Intensity Over Time</h1>
<p class="intro">This interactive visualization accompanies the course project for <em>Current Debates on Political Inequality</em> (Summer Semester 2026). The project investigates how the German Fridays for Future movement frames climate change in its Instagram posts.</p>
<p class="intro">The chart below can be explored interactively: select a different axis (see the description shown for each), optionally compare it against a second axis plotted on its own right-hand scale (shown as dashed lines), choose how finely the data is aggregated over time, toggle which event markers are displayed, and hover over an event marker for its title and description.</p>

<div class="controls">
  <div class="control-group">
    <label class="group-label" for="axisSelect">Axis</label>
    <select id="axisSelect"></select>
    <div id="axisDescription"></div>
  </div>
  <div class="control-group">
    <label class="group-label" for="compareAxisSelect">Compare with (optional, dashed)</label>
    <select id="compareAxisSelect"></select>
    <div id="compareAxisDescription"></div>
  </div>
  <div class="control-group">
    <label class="group-label" for="aggSelect">Aggregation level</label>
    <select id="aggSelect"></select>
  </div>
  <div class="control-group">
    <label class="group-label">Event types shown</label>
    <div class="checkbox-row" id="eventTypeCheckboxes"></div>
  </div>
  <div class="control-group">
    <label class="group-label">&nbsp;</label>
    <label><input type="checkbox" id="windowsToggle" checked> Show crisis-window shading</label>
  </div>
</div>

<div id="plot-container"><div id="plot"></div></div>

</div>

<script>
const DATA = __PAYLOAD__;

const EVENT_STYLE = {
  global_strike: "#c0392b",
  eu_election:   "#2980b9",
  bt_election:   "#8e44ad",
  election:      "#7f8c8d",
  strike:        "#16a085",
  campaign:      "#27ae60",
  congress:      "#f39c12",
  COP:           "#2c3e50",
};
const WINDOW_COLORS = ["#f9d5d3", "#d3e5f9", "#d3f9d8", "#f9f0d3", "#e6d3f9"];

function typeLabel(t) {
  return DATA.event_type_labels[t] || t;
}

// Derives a saturated/dark label color from a pastel window-shading color, by hue alone --
// "same color, just more intense" rather than a separately hand-picked color per window.
function hexToHsl(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s;
  const l = (max + min) / 2;
  if (max === min) {
    h = s = 0;
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      default: h = (r - g) / d + 4; break;
    }
    h /= 6;
  }
  return [h * 360, s * 100, l * 100];
}
function hslToHex(h, s, l) {
  s /= 100; l /= 100;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((h / 60) % 2 - 1));
  const m = l - c / 2;
  let r, g, b;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  const toHex = v => Math.round((v + m) * 255).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}
function intensifyColor(hex) {
  const [h] = hexToHsl(hex);
  return hslToHex(h, 65, 32);
}

const axisSelect = document.getElementById("axisSelect");
const axisDescription = document.getElementById("axisDescription");
DATA.axis_order.forEach(key => {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = DATA.axes[key].label;
  axisSelect.appendChild(opt);
});

const compareAxisSelect = document.getElementById("compareAxisSelect");
const compareAxisDescription = document.getElementById("compareAxisDescription");
const noneOpt = document.createElement("option");
noneOpt.value = "";
noneOpt.textContent = "(none)";
compareAxisSelect.appendChild(noneOpt);
DATA.axis_order.forEach(key => {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = DATA.axes[key].label;
  compareAxisSelect.appendChild(opt);
});

const aggSelect = document.getElementById("aggSelect");
DATA.agg_levels.forEach(level => {
  const opt = document.createElement("option");
  opt.value = level;
  opt.textContent = level.charAt(0).toUpperCase() + level.slice(1);
  aggSelect.appendChild(opt);
});
aggSelect.value = DATA.default_agg_level;

const checkboxContainer = document.getElementById("eventTypeCheckboxes");
DATA.event_types.forEach(t => {
  const id = "evt_" + t;
  const wrap = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.className = "eventType";
  cb.value = t;
  cb.id = id;
  // Default on: the curated "highlight" types (global_strike, eu_election, bt_election);
  // others start off so the initial view matches the static plots rather than showing all
  // ~100 events at once.
  cb.checked = DATA.events.some(e => e.type === t && e.highlight);
  wrap.appendChild(cb);
  wrap.appendChild(document.createTextNode(" " + typeLabel(t)));
  checkboxContainer.appendChild(wrap);
});

function render() {
  const axisKey = axisSelect.value;
  const compareKey = compareAxisSelect.value;
  const aggLevel = aggSelect.value;

  const axis = DATA.axes[axisKey];
  const axisSeries = axis.by_level[aggLevel];
  axisDescription.textContent = axis.description || "";

  const compareAxis = compareKey ? DATA.axes[compareKey] : null;
  compareAxisDescription.textContent = compareAxis ? (compareAxis.description || "") : "";

  const checkedTypes = new Set(
    [...document.querySelectorAll(".eventType:checked")].map(el => el.value)
  );
  const showWindows = document.getElementById("windowsToggle").checked;

  // "name" stays a short concept label -- the legend column position (below) plus its
  // per-axis title already say which axis it belongs to. The hovertemplate below carries the
  // full "Axis: concept" text instead, since that's where the un-truncated name is needed.
  const lineTraces = axis.concepts.map(c => ({
    x: axisSeries.dates,
    y: axisSeries.series[c],
    type: "scatter",
    mode: "lines",
    name: c.replace(/_/g, " "),
    legend: "legend",
    line: { width: 2.5, dash: "solid" },
    connectgaps: true,
    hovertemplate: `<b>${axis.label}: ${c.replace(/_/g, " ")}</b><br>%{x}<br>%{y}<extra></extra>`,
  }));

  let compareTraces = [];
  if (compareAxis) {
    const compareSeries = compareAxis.by_level[aggLevel];
    compareTraces = compareAxis.concepts.map(c => ({
      x: compareSeries.dates,
      y: compareSeries.series[c],
      type: "scatter",
      mode: "lines",
      name: c.replace(/_/g, " "),
      legend: "legend2",
      line: { width: 2.5, dash: "dash" },
      yaxis: "y2",
      connectgaps: true,
      hovertemplate: `<b>${compareAxis.label}: ${c.replace(/_/g, " ")}</b><br>%{x}<br>%{y}<extra></extra>`,
    }));
  }

  const filteredEvents = DATA.events.filter(e => checkedTypes.has(e.type));
  const allValues = axis.concepts.flatMap(c => axisSeries.series[c]).filter(v => v !== null);
  const maxY = allValues.length ? Math.max(...allValues) * 1.1 : 1;

  const eventTrace = {
    x: filteredEvents.map(e => e.date),
    y: filteredEvents.map(() => maxY),
    type: "scatter",
    mode: "markers",
    name: "Events",
    marker: {
      size: 8,
      color: filteredEvents.map(e => EVENT_STYLE[e.type] || "#888"),
      symbol: "triangle-down",
      line: { width: 1, color: "#fff" },
    },
    text: filteredEvents.map(e => `<b>${escapeHtml(e.label)}</b><br>${escapeHtml(e.topic)}`),
    hovertemplate: "%{text}<extra></extra>",
    showlegend: false,
  };

  const shapes = [];
  const annotations = [];
  if (showWindows) {
    DATA.crisis_windows.forEach((w, i) => {
      const color = WINDOW_COLORS[i % WINDOW_COLORS.length];
      shapes.push({
        type: "rect", xref: "x", yref: "paper",
        x0: w.start, x1: w.end, y0: 0, y1: 1,
        fillcolor: color,
        opacity: 0.45, line: { width: 0 }, layer: "below",
      });
      // Short label (text before the first ":" in the window's description) centered in the
      // window and near the plot's bottom edge, in a more saturated version of the window's
      // own shading color.
      const startMs = new Date(w.start).getTime();
      const endMs = new Date(w.end).getTime();
      const midDate = new Date((startMs + endMs) / 2).toISOString().slice(0, 10);
      annotations.push({
        x: midDate, xref: "x",
        y: 0.04, yref: "paper",
        xanchor: "center", yanchor: "bottom",
        text: w.description.split(":")[0],
        showarrow: false,
        font: { size: 11, color: intensifyColor(color) },
      });
    });
  }
  filteredEvents.forEach(e => {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: e.date, x1: e.date, y0: 0, y1: 1,
      line: { color: EVENT_STYLE[e.type] || "#888", width: 1.2 },
    });
  });

  // Plotly titles don't auto-wrap long text -- break it into explicit lines (axis label(s) on
  // their own line) rather than letting a long combined "A vs. B" title run off the sides.
  const aggLabel = aggLevel.charAt(0).toUpperCase() + aggLevel.slice(1);
  const titleText = compareAxis
    ? `Frame Intensity Over Time<br>${axis.label} (solid)<br>vs. ${compareAxis.label} (dashed)<br><span style="font-size:11px;font-weight:normal">${aggLabel} aggregation</span>`
    : `Frame Intensity Over Time<br>${axis.label}<br><span style="font-size:11px;font-weight:normal">${aggLabel} aggregation</span>`;
  const titleLines = compareAxis ? 4 : 3;
  const legendRows = compareAxis ? Math.max(axis.concepts.length, compareAxis.concepts.length) : 0;
  const marginTop = 20 + titleLines * 20;
  // In compare mode the legends sit below the x-axis tick labels + "Date" axis title (that
  // chrome needs ~65px of clearance, matching the -0.15 paper-y offset used below), then the
  // legend's own title + one row per concept.
  const marginBottom = compareAxis ? 100 + legendRows * 18 : 60;

  const layout = {
    title: { text: titleText },
    xaxis: { title: { text: "Date" } },
    yaxis: { title: { text: compareAxis ? `${axis.label} (solid)` : "Mean intensity" } },
    shapes: shapes,
    annotations: annotations,
    hovermode: "closest",
    hoverlabel: { align: "left" },
    margin: { t: marginTop, r: compareAxis ? 60 : 20, b: marginBottom, l: 60 },
    height: 420 + marginTop + marginBottom,
    paper_bgcolor: "#f8f4e9",
    plot_bgcolor: "#fdfbf5",
  };

  if (compareAxis) {
    // Both axes are 0-4 scored scales -- fixing both to the same range keeps a shared,
    // comparable sense of scale rather than each auto-scaling to its own data range.
    layout.yaxis.range = [0, 4];
    layout.yaxis2 = {
      title: { text: `${compareAxis.label} (dashed)` },
      overlaying: "y",
      side: "right",
      range: [0, 4],
    };
    // Two separate legends (Plotly.js multi-legend support) so the primary axis's solid
    // lines stay in their own column, separate from the secondary axis's dashed lines,
    // rather than one interleaved list.
    layout.legend = {
      orientation: "v",
      title: { text: axis.label, font: { size: 11 } },
      x: 0.22, xanchor: "center",
      y: -0.15, yanchor: "top",
    };
    layout.legend2 = {
      orientation: "v",
      title: { text: compareAxis.label, font: { size: 11 } },
      x: 0.78, xanchor: "center",
      y: -0.15, yanchor: "top",
    };
  } else {
    layout.legend = { orientation: "h", y: -0.15 };
  }

  Plotly.react("plot", [...lineTraces, ...compareTraces, eventTrace], layout, { responsive: true, displaylogo: false });
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : s;
  return div.innerHTML;
}

axisSelect.addEventListener("change", render);
compareAxisSelect.addEventListener("change", render);
aggSelect.addEventListener("change", render);
document.getElementById("windowsToggle").addEventListener("change", render);
checkboxContainer.addEventListener("change", render);

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
