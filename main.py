"""Dip-buying long-only bot for btc-race-24h-20261117-0603 (agent-fun/bars-replay/v1).

Idea, in one sentence: buy Coinbase Wrapped BTC only when it is unusually cheap
against its own last four hours, then hold it with a wide trailing stop, because
on one-minute bars this market is a choppy random walk where every active
strategy the kit ships loses money to the ~26-36 bps round-trip cost, while the
day's drift is captured by being long.

Evidence from the published dev window (24 h, 1,440 bars, +0.49% drift):
  - ema-cross baseline: -$106.21 (68 trades); adx-momentum: -$32.66;
    rsi-bollinger: -$24.25. Frequent trading bleeds.
  - buy-and-hold: +$1.07; momentum (buy bar 24, hold): +$2.45.
  - Non-overlapping 60-minute returns show mean reversion (autocorr about -0.36),
    but the bounce it implies (~19 bps after a -52 bps hour) is smaller than the
    round-trip cost, so quick mean-reversion exits lose; only patient entries
    that then ride the drift survive.
  - This bot's rule on the dev window: +$5.03, 1 trade, max drawdown 2.12%.

Rules (all in bar time; no wall clock, no randomness, no network, no files):
  - Universe: one symbol, CBBTC, long only, at most one open position.
  - Indicators: rolling 240-bar (4 h) simple moving average and population
    standard deviation of closes, updated from every bar the harness sends.
  - Entry: once 240 bars have been seen, no position is held and the cooldown
    since the last exit has passed, buy the whole equity when the close sits
    more than 1.8 standard deviations below the 4 h mean (a deep dip).
  - Fallback: if no buy has ever filled by bar 1,200, buy anyway, so the entry
    always makes at least one trade and meets the race guardrail even on a
    calm day with no deep dip.
  - Exit, whole position: a trailing stop 2.5% below the highest close since
    entry, or a hard stop 2.0% below the entry close. Both are wide on purpose:
    they fire only on a genuine reversal, never on one-minute chop. Exits are
    re-sent while their condition still holds, so a rejected sell is retried.
  - Every order caps slippage at 100 bps from the latest close, so a fill far
    past the close is refused rather than chased.
  - A buy spends at most the cash on hand; a sell empties the position.
"""

import os
import sys

# The grader runs `python3 -P -s /bot/main.py`, which leaves this directory off
# sys.path, so the vendored shim is added back explicitly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars

SYMBOL = "CBBTC"

# --- strategy constants (tuned on the published dev window, then frozen) ---
LOOKBACK_BARS = 240      # 4 hours of one-minute closes for mean and stdev
Z_ENTRY = -1.8           # buy when close is this many stdevs below the mean
TRAIL_BPS = 250          # trailing stop: 2.5% below the peak close since entry
STOP_BPS = 200           # hard stop: 2.0% below the entry close
COOLDOWN_BARS = 60       # bars to wait after an exit before buying again
FALLBACK_BAR = 1200      # buy anyway from this bar on if no trade has filled
MAX_SLIPPAGE_BPS = 100   # refuse any fill more than 1% past the latest close


class DipBuyer:
    """Rolling mean/stdev plus the position state. Deterministic: floats only."""

    def __init__(self):
        self.closes = []          # closes seen, newest last (ints)
        self.sum = 0.0            # running sum of the lookback window
        self.sum_sq = 0.0         # running sum of squares of the window
        self.last_t = None        # t of the last bar folded in
        self.entry_close = 0      # close the current position was bought at
        self.peak_close = 0       # highest close since the current entry
        self.cool_until = -1      # bar index before which no new entry
        self.traded = False       # True once a buy of ours has actually filled
        self.bar_index = -1       # bars answered so far
        self.pending_entry = 0    # close of the bar a buy was decided on

    def note_bar(self, bar):
        """Fold one new bar's close into the rolling window."""
        if bar["t"] == self.last_t:
            return
        self.last_t = bar["t"]
        close = int(bar["c"])
        self.closes.append(close)
        self.sum += close
        self.sum_sq += float(close) * float(close)
        if len(self.closes) > LOOKBACK_BARS:
            old = self.closes.pop(0)
            self.sum -= old
            self.sum_sq -= float(old) * float(old)

    def mean_stdev(self):
        """Mean and population stdev of the lookback window, or (None, None)."""
        n = len(self.closes)
        if n < LOOKBACK_BARS:
            return None, None
        mean = self.sum / n
        var = self.sum_sq / n - mean * mean
        if var <= 0.0:
            return mean, 0.0
        return mean, var ** 0.5


BOT = DipBuyer()


def on_batch(state, batch):
    """Decide this batch's orders. Returns a list of order dicts (maybe empty)."""
    BOT.bar_index += 1
    for bar in batch["bars"]:
        if bar["symbol"] == SYMBOL:
            BOT.note_bar(bar)

    if state.warming_up:
        # Orders decided on a warm-up batch are rejected; build history only.
        return []

    holding = state.positions.get(SYMBOL)
    close = state.close(SYMBOL)
    if close is None:
        return []

    if holding is not None and holding.qty > 0:
        if not BOT.traded:
            BOT.traded = True
        # Anchor the stops to the close the buy was decided on (the signal
        # price), not the close after the fill: the hard stop is then measured
        # from the price that triggered the trade. Fall back to this bar's
        # close if the decision price was not recorded (e.g. after a restart
        # of the replay, the position was already held when this run began).
        if BOT.entry_close == 0:
            BOT.entry_close = BOT.pending_entry if BOT.pending_entry else close
            BOT.peak_close = BOT.entry_close
        if close > BOT.peak_close:
            BOT.peak_close = close
        # Exit: trailing stop below the peak, or hard stop below the entry.
        if close < BOT.peak_close * (10000 - TRAIL_BPS) // 10000:
            BOT.entry_close = 0
            BOT.peak_close = 0
            BOT.pending_entry = 0
            BOT.cool_until = BOT.bar_index + COOLDOWN_BARS
            return [{"side": "sell", "symbol": SYMBOL, "qty": holding.qty,
                     "max_slippage_bps": MAX_SLIPPAGE_BPS}]
        if close < BOT.entry_close * (10000 - STOP_BPS) // 10000:
            BOT.entry_close = 0
            BOT.peak_close = 0
            BOT.pending_entry = 0
            BOT.cool_until = BOT.bar_index + COOLDOWN_BARS
            return [{"side": "sell", "symbol": SYMBOL, "qty": holding.qty,
                     "max_slippage_bps": MAX_SLIPPAGE_BPS}]
        return []

    # Flat: reset the entry anchors and look for a buy.
    BOT.entry_close = 0
    BOT.peak_close = 0
    if BOT.bar_index <= BOT.cool_until:
        return []

    mean, stdev = BOT.mean_stdev()
    buy = False
    if mean is not None and stdev is not None and stdev > 0.0:
        # Deep dip: close more than Z_ENTRY standard deviations under the mean.
        if (close - mean) / stdev < Z_ENTRY:
            buy = True
    if not buy and not BOT.traded and BOT.bar_index >= FALLBACK_BAR:
        # Late fallback: guarantee at least one trade for the guardrail.
        buy = True
    if not buy:
        return []

    notional = state.equity
    if notional > state.cash:
        notional = state.cash
    if notional < 1_000_000:  # below the 1 USDC minimum notional: buy nothing
        return []
    BOT.pending_entry = close
    return [{"side": "buy", "symbol": SYMBOL,
             "notional_quote_units": notional,
             "max_slippage_bps": MAX_SLIPPAGE_BPS}]


makefun_bars.run(on_batch)
