# IMC-4-SIGQuant

## Overview

This repository contains the working codebase for an IMC Prosperity 4 team project. It combines:
- a starter trading strategy implementation (`trading.py`),
- a local data visualizer for order book and trade inspection (`visualizer/`),
- Round 1 market data (`DataCapsules/ROUND1/`),
- and reference/planning material for strategy iteration.

The goal is to support fast strategy development, debugging, and collaboration during the competition.

## Features

- **Trading strategy scaffold** in `trading.py` with product-specific logic.
- **Interactive visualizer** built with `matplotlib` and `pandas` to inspect market behavior.
- **Round 1 data bundle** with three days of prices and trades.
- **Reference strategy file** from last year (`frankfurtHedgeHogs_imc3.py`) for idea generation.
- **Internal planning notes** in `PROJECT_OVERVIEW.md`.

## Repository Structure

```text
.
├── trading.py
├── visualizer/
├── DataCapsules/
│   └── ROUND1/
├── OfficialLogs/
│   └── Run1(137859)/
├── PROJECT_OVERVIEW.md
├── frankfurtHedgeHogs_imc3.py
└── README.md
```

## Data (`DataCapsules/ROUND1`)

`DataCapsules/ROUND1/` contains the Round 1 input dataset used for strategy analysis and replay:
- three days of market prices,
- trade data suitable for visual inspection and offline reasoning.

Use this dataset with the visualizer and strategy development workflow to validate assumptions and tune quoting logic.

## Visualizer

The visualizer (`visualizer/`) is a custom `matplotlib` + `pandas` dashboard for one-product-at-a-time market inspection. It supports:

- toggling products,
- toggling order book depth levels,
- toggling days,
- inspecting trades,
- zooming into time ranges.

### Launch

```bash
python3 -m visualizer.main dashboard --show
```

### Launch With Official Run Overlay

Use the official submission logs in `OfficialLogs/` to add your own trades, the official run order book, and the P&L/position summary:

```bash
python3 -m visualizer.main dashboard --show --logs OfficialLogs --run "Run1(137859)"
```

### Snapshot View

Render a single order-book snapshot for a product/day/timestamp:

```bash
python3 -m visualizer.main snapshot --product ASH_COATED_OSMIUM --day 0 --timestamp 500000 --show
```

### Visualizer Controls

Inside the live dashboard:

- use the radio buttons or left/right arrow keys to switch products
- use the `Layers`, `Depth`, and `Days` toggles on the left
- drag on the top chart to zoom time
- scroll to zoom time
- hold `Shift` while scrolling to zoom price
- double-click or press `r` to reset the zoom

## Strategy Summary (`trading.py`)

Current implemented logic:

- **`ASH_COATED_OSMIUM`**  
  Fixed-fair-value market making around a stable center.

- **`INTARIAN_PEPPER_ROOT`**  
  Drift-aware market making with an upward-trending fair value.

Both are structured for iterative extension as more products and signals are added.

## Setup / Installation

Install dependencies from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

## Usage

Validate strategy file syntax:

```bash
python3 -m py_compile trading.py
```

Run the visualizer:

```bash
python3 -m visualizer.main dashboard --show
```

Run the visualizer with official logs:

```bash
python3 -m visualizer.main dashboard --show --logs OfficialLogs --run "Run1(137859)"
```

Render a snapshot without opening the full dashboard:

```bash
python3 -m visualizer.main snapshot --product INTARIAN_PEPPER_ROOT --day 1 --timestamp 99900 --show
```

## Current Limitations / Next Steps

- Strategy coverage is currently limited to two products.
- Fair value models are intentionally simple and should be expanded with stronger signal inputs.
- Risk controls, inventory management refinements, and execution robustness can be improved.
- Visualizer workflows are local/manual; additional automation for repeatable analysis would help collaboration.
