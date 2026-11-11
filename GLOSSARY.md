# TradingScope

A multi-agent system that analyzes one stock for one date, produces a risk-checked trading plan, and later evaluates that plan against observed market prices.

## Language

### Subject and dates

**Ticker**:
The exchange symbol of the single stock an analysis is about.
_Avoid_: Company of interest, company name, symbol

**Analysis Date**:
The calendar date an analysis is produced for; evaluation treats sessions on or after it as the future.
_Avoid_: Trade date, run date

**Last Completed Session**:
The most recent trading session whose full daily data was available when the analysis was produced.
_Avoid_: Latest trading date

**Session**:
One trading day of the market.
_Avoid_: Trading day bar

**Session Date**:
The calendar date of the trading session a daily price bar describes.
_Avoid_: Trade date (for a bar's date), analysis date (for observed prices)

### Analysis

**Stage Output**:
The result produced by one analysis stage for a ticker and analysis date, identified by the role that produced it (market, fundamentals, news, social media, research manager, trader, portfolio manager).
_Avoid_: Analysis (for a single stage), decision (for a whole stage output), report

**Decision**:
The directional call and action shared by every stage output, with its confidence and reasoning.
_Avoid_: Recommendation, verdict

### Trading plan

**Trade Intent**:
What the plan does to the position: open long, reduce long, close long, open short, cover short, or hold.
_Avoid_: Trade type, order type

**Price Plan**:
The prices a stage output commits to: execution price or range, target price, stop loss, and invalidation conditions.
_Avoid_: Trade plan (for prices alone), levels

**Execution Price**:
The price at which the plan's trade intent should be carried out, whether that buys or sells; may be bounded by an **Execution Range**.
_Avoid_: Entry price (for exits), order price

**Execution Range**:
The lowest and highest prices at which the planned trade should be carried out.
_Avoid_: Entry range (for exits)

### Evaluation

**Fill**:
The first session in which the simulated execution range was touched, so the plan's trade is assumed to have happened.
_Avoid_: Entry triggered, execution (for the simulated event)

**Fill Price**:
The simulated price of a fill, taken from the overlap of the execution range and that session's high-low range.
_Avoid_: Actual entry price, execution price (for the simulated value)

**Underlying Return**:
The ticker's own price change from the session before the analysis date to the close of the last session in the evaluation horizon.
_Avoid_: Benchmark return, actual return, market return

**Strategy Return**:
The performance attributable to the simulated trade over the evaluation horizon; for a reduction or exit, it measures relative performance after the sale or cover, not the return of the full position.
_Avoid_: Trade return, PnL
