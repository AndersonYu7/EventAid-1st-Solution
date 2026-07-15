"""Classical event-stream denoising for EVFI preprocessing.

Two stages, applied to (N, 4) [t, x, y, p] arrays sorted by t (the format
returned by evlib.dataset.Sequence.load_events):

1. Hot-pixel removal: event counts are accumulated per pixel over the FULL
   sequence (``EventDenoiser.fit``); pixels whose count exceeds
   ``mean + hot_k * std`` (computed over the whole sensor grid) are dropped
   from every window.

2. BAF (Delbruck-style background activity filter): an event is kept only if
   at least one of its 8 spatial neighbors fired an event within ``tau``
   microseconds before it. Implemented as a sequential scan over time-sorted
   events with a per-pixel last-timestamp map, numba-JIT'd for speed. The
   timestamp map is updated for every incoming event (kept or not), as in the
   classical filter.

Pipeline role (1st-place EventAid-F solution, team yunyu8):
  MEMBER INFERENCE preprocessing in
  data prep -> training -> member inference -> fusion -> submission.
  Optionally applied to the real EventAid event stream before voxelization
  for the event-conditioned ensemble members (EMA-E variants, TimeLens,
  TimeLens-XL, REFID, CBMNet); cleaner voxels reduce hallucinated texture
  from sensor noise in low-activity regions.

Inputs:  (N, 4) float64 [t, x, y, p] arrays, time-sorted (the exact format
         returned by evlib.dataset.Sequence.load_events).
Outputs: filtered copies of the same arrays (rows removed, order preserved).

Usage example:
  from evlib.denoise import EventDenoiser
  den = EventDenoiser(seq.width, seq.height, mode="hp+baf", tau=5000.0)
  den.fit(seq.load_events(t_first, t_last))   # hot-pixel mask, full stream
  ev_clean = den(seq.load_events(t0, t1))     # then filter each window
  print(den.stats())
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit

_NEG = -1e18  # "never fired" timestamp


@njit(cache=True)
def _baf_scan(t, x, y, last_ts, tau, keep):
    """Sequential BAF over time-sorted events (numba-compiled).

    keep[i] = True iff any of the 8 spatial neighbors of event i fired within
    tau microseconds before it. last_ts is the per-pixel last-event-timestamp
    map, updated for EVERY event regardless of the keep decision.
    """
    h, w = last_ts.shape
    for i in range(t.shape[0]):
        ti = t[i]
        xi = x[i]
        yi = y[i]
        ok = False
        # Scan the 3x3 neighborhood (excluding the center pixel itself).
        for dy in range(-1, 2):
            yy = yi + dy
            if yy < 0 or yy >= h:
                continue
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                xx = xi + dx
                if xx < 0 or xx >= w:
                    continue
                if ti - last_ts[yy, xx] <= tau:
                    ok = True
                    break
            if ok:
                break
        keep[i] = ok
        last_ts[yi, xi] = ti  # update even for rejected events (classical BAF)


class EventDenoiser:
    """Per-sequence denoiser. ``fit`` on the full event stream once, then call
    on each event window. Accumulates simple runtime / reduction stats."""

    def __init__(self, width: int, height: int, mode: str = "hp+baf",
                 tau: float = 5000.0, hot_k: float = 5.0):
        assert mode in ("hp", "hp+baf"), mode
        self.width = width
        self.height = height
        self.mode = mode
        self.tau = float(tau)
        self.hot_k = float(hot_k)
        self.hot_mask = np.zeros((height, width), dtype=bool)  # True = hot
        self.n_hot = 0
        # stats
        self.fit_seconds = 0.0
        self.filter_seconds = 0.0
        self.n_in = 0
        self.n_after_hot = 0
        self.n_out = 0

    def fit(self, events: np.ndarray) -> "EventDenoiser":
        """Compute the hot-pixel mask from the full-sequence events."""
        t0 = time.time()
        counts = np.zeros(self.height * self.width, dtype=np.int64)
        if events.shape[0]:
            x = events[:, 1].astype(np.int64)
            y = events[:, 2].astype(np.int64)
            valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            np.add.at(counts, y[valid] * self.width + x[valid], 1)
        # A pixel is "hot" if its whole-sequence event count is an outlier
        # (> mean + hot_k * std over the full sensor grid).
        thr = counts.mean() + self.hot_k * counts.std()
        self.hot_mask = (counts > thr).reshape(self.height, self.width)
        self.n_hot = int(self.hot_mask.sum())
        self.fit_seconds += time.time() - t0
        return self

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Filter one time-sorted event window; returns a filtered copy."""
        t0 = time.time()
        self.n_in += events.shape[0]
        if events.shape[0] == 0:
            return events

        # Stage 1: drop out-of-bounds events and events on hot pixels.
        x = events[:, 1].astype(np.int64)
        y = events[:, 2].astype(np.int64)
        inb = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        keep = inb & ~self.hot_mask[np.clip(y, 0, self.height - 1),
                                    np.clip(x, 0, self.width - 1)]
        events = events[keep]
        self.n_after_hot += events.shape[0]

        # Stage 2: background-activity filter. The last-timestamp map is
        # reset per window, so isolated events at a window start are dropped.
        if self.mode == "hp+baf" and events.shape[0]:
            last_ts = np.full((self.height, self.width), _NEG, dtype=np.float64)
            keep2 = np.zeros(events.shape[0], dtype=np.bool_)
            _baf_scan(np.ascontiguousarray(events[:, 0]),
                      events[:, 1].astype(np.int64),
                      events[:, 2].astype(np.int64),
                      last_ts, self.tau, keep2)
            events = events[keep2]

        self.n_out += events.shape[0]
        self.filter_seconds += time.time() - t0
        return events

    def stats(self) -> str:
        kept_hot = self.n_after_hot / max(self.n_in, 1)
        kept = self.n_out / max(self.n_in, 1)
        return (f"mode={self.mode} tau={self.tau:.0f}us hot_px={self.n_hot} | "
                f"events {self.n_in} -> {self.n_after_hot} (hp, {kept_hot:.1%})"
                f" -> {self.n_out} (final, {kept:.1%}) | "
                f"fit {self.fit_seconds:.2f}s + filter {self.filter_seconds:.2f}s")
