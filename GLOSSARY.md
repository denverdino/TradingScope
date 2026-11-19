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

**Decision Layer**:
The stages after the analysts that turn their evidence into a trading plan: the bull and bear researchers, research manager, trader, risk debaters, and portfolio manager.
_Avoid_: Decision agents, downstream agents

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

**Horizon**:
The number of sessions, starting with the first session on or after the analysis date, over which a stage output is judged.
_Avoid_: Holding period, window

**Evaluation**:
The deterministic judgment of one stage output against observed prices at one horizon: whether its decision's direction was right and, for stage outputs with a price plan, how the simulated trade performed.
_Avoid_: Lesson, reflection, review

**Matured Evaluation**:
An evaluation whose horizon ended before a given analysis date, so it was knowable when that analysis was produced and may inform it.
_Avoid_: Completed evaluation, past evaluation

**Benchmark Return**:
The price change over the same horizon of the ticker's sector fund, or of the broad market when the ticker has no sector mapping.
_Avoid_: Market return, index return

**Excess Return**:
The underlying return minus the benchmark return over the same horizon; direction skill is judged on it.
_Avoid_: Alpha, relative return

**Shock Session**:
A session within the horizon whose move relative to the benchmark, or whose opening gap, is far larger than the ticker's normal daily range.
_Avoid_: Black swan, outlier, abnormal day

**Scheduled Event**:
A ticker-specific event, such as an earnings release, whose date was knowable when the analysis was produced; a shock session on it is not excused.
_Avoid_: Known event, catalyst

**Direction Skill**:
How often stage outputs called direction correctly, measured only on evaluations without an unscheduled or unclassified shock session.
_Avoid_: Accuracy, hit rate (unqualified)

**Risk Control**:
How well simulated trades limited losses and captured gains across all evaluations, including those with shock sessions.
_Avoid_: Trade performance, PnL

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

### Learning from evaluations

**Track Record**:
Statistics computed from matured evaluations, for one ticker or across all tickers, that describe how a stage's past decisions performed.
_Avoid_: Performance history, stats

**Lesson**:
A claim about how future analysis should change, scoped to one ticker or to all tickers, that cites the matured evaluations supporting it.
_Avoid_: Memory, insight, reflection

**Confirmed Lesson**:
A lesson whose cited support passed the admission checks and has not been retired; the only kind shown to the decision layer.
_Avoid_: Active lesson, validated memory

**Lesson Snapshot**:
The lessons and track record as of one session, never changed afterward; an analysis reads the latest snapshot covering only sessions before its analysis date.
_Avoid_: Memory store, memory state

**Calibration Brief**:
The small part of a lesson snapshot shown to the decision layer: a track record summary and a few confirmed lessons, framed as evidence rather than instructions.
_Avoid_: Injected memory, lessons prompt

**Consolidation**:
The offline step that derives, corrects, and prunes lessons by checking them against matured evaluations and the stage outputs they judged.
_Avoid_: Dreaming, reflection, learning
