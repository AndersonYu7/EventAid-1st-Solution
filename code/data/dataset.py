"""Shared data utilities for the EventAid Frame Interpolation Challenge.

Pipeline role (1st-place EventAid-F solution, team yunyu8):
  DATA PREP + MEMBER INFERENCE support, used by every stage of
  data prep -> training -> member inference -> fusion -> submission.
  This is the single canonical reader for the official challenge_data layout:
  every ensemble-member inference script, the fusion/blending code, the
  statistics tooling (scripts/analyze_event_stats.py) and the submission
  builder (scripts/build_final.py) go through these helpers, so all models see
  identical frames, event windows and voxel grids.

Inputs -- one challenge sequence dir <data_dir>/<split>/<skip>/<name>/ with:
  shape.txt        "width height"
  frame_info.txt   header + rows "path timestamp_us [input]|[TODO]"
  event_info.txt   header + rows "path t_start_us t_end_us"
  event .txt files with "t x y p" integer rows (p in {0, 1})

Outputs / API:
  load_sequence / iter_sequences  -> Sequence objects (frames + event index)
  Sequence.load_frame             -> float32 RGB in [0, 1]
  Sequence.load_events            -> (N, 4) float64 [t, x, y, p] time-sorted
  events_to_voxel                 -> float32 (n_bins, H, W) bilinear voxel
  save_result_png                 -> uint8 PNG in the submission results layout

Usage example:
  from evlib.dataset import load_sequence, events_to_voxel
  seq = load_sequence("challenge_data", "validation", "7skip", "seq_00")
  ev = seq.load_events(seq.frames[0].timestamp, seq.frames[8].timestamp)
  vox = events_to_voxel(ev, 16, seq.height, seq.width)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

SKIPS = ("1skip", "3skip", "7skip", "15skip")


@dataclass(frozen=True)
class Frame:
    """One row of frame_info.txt: an input anchor or a to-be-predicted frame."""
    index: int
    path: str        # relative to sequence dir for inputs; submission path for TODO
    timestamp: int   # microseconds
    is_input: bool   # True = given anchor frame, False = TODO (to predict)


@dataclass(frozen=True)
class EventFile:
    """One row of event_info.txt: an event chunk covering [t_start, t_end)."""
    path: str        # relative to sequence dir
    t_start: int
    t_end: int


@dataclass
class Sequence:
    """One challenge sequence: frame index plus lazily-loaded event chunks."""
    split: str
    skip: str
    name: str
    root: Path           # sequence directory
    width: int
    height: int
    frames: list[Frame]
    event_files: list[EventFile]

    @property
    def inputs(self) -> list[Frame]:
        """Given anchor frames (available on disk)."""
        return [f for f in self.frames if f.is_input]

    @property
    def todos(self) -> list[Frame]:
        """Frames the challenge asks us to predict (submission targets)."""
        return [f for f in self.frames if not f.is_input]

    def load_frame(self, frame: Frame) -> np.ndarray:
        """Load an input frame as float32 RGB in [0, 1], shape (H, W, 3)."""
        assert frame.is_input
        with Image.open(self.root / frame.path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

    def load_events(self, t_start: int, t_end: int) -> np.ndarray:
        """Load events with t_start <= t < t_end as (N, 4) float64 [t, x, y, p].

        Polarity is returned as stored: {0, 1}.
        """
        chunks = []
        for ef in self.event_files:
            # Skip chunks whose time range does not overlap the query window.
            if ef.t_end <= t_start or ef.t_start >= t_end:
                continue
            arr = _load_event_file(str(self.root / ef.path))
            if arr.size == 0:
                continue
            mask = (arr[:, 0] >= t_start) & (arr[:, 0] < t_end)
            chunks.append(arr[mask])
        if not chunks:
            return np.empty((0, 4), dtype=np.float64)
        ev = np.concatenate(chunks, axis=0)
        # Stable sort by timestamp so equal-t events keep file order.
        return ev[np.argsort(ev[:, 0], kind="stable")]


@lru_cache(maxsize=64)
def _load_event_file(path: str) -> np.ndarray:
    # Files are "t x y p" integer text rows; loadtxt is slow, use fromstring.
    # lru_cache keeps recently-used chunks in RAM: consecutive anchor gaps
    # reuse the same files, so this avoids re-parsing large text files.
    with open(path, "rb") as f:
        data = np.fromstring(f.read(), dtype=np.float64, sep=" ")  # noqa: NPY201
    if data.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    return data.reshape(-1, 4)


def load_sequence(data_dir: str | Path, split: str, skip: str, name: str) -> Sequence:
    """Parse one sequence dir (shape/frame_info/event_info) into a Sequence."""
    root = Path(data_dir) / split / skip / name
    w, h = map(int, (root / "shape.txt").read_text().split())

    frames = []
    for i, line in enumerate((root / "frame_info.txt").read_text().splitlines()):
        if i == 0 or not line.strip():  # skip header row and blank lines
            continue
        p, ts, kind = line.split()
        # Frame index from the filename stem; input frames end in "_img".
        stem = Path(p).stem
        idx = int(stem[:-4] if stem.endswith("_img") else stem)
        frames.append(Frame(idx, p, int(ts), kind == "[input]"))
    frames.sort(key=lambda f: f.index)

    event_files = []
    for i, line in enumerate((root / "event_info.txt").read_text().splitlines()):
        if i == 0 or not line.strip():  # skip header row and blank lines
            continue
        p, t0, t1 = line.split()
        event_files.append(EventFile(p, int(t0), int(t1)))
    event_files.sort(key=lambda e: e.t_start)

    return Sequence(split, skip, name, root, w, h, frames, event_files)


def iter_sequences(data_dir: str | Path, splits=("validation", "test"), skips=SKIPS):
    """Yield every Sequence under data_dir for the given splits/skips."""
    data_dir = Path(data_dir)
    for split in splits:
        for skip in skips:
            d = data_dir / split / skip
            if not d.exists():
                continue
            for seq_dir in sorted(p for p in d.iterdir() if p.is_dir()):
                yield load_sequence(data_dir, split, skip, seq_dir.name)


def events_to_voxel(
    events: np.ndarray,
    n_bins: int,
    height: int,
    width: int,
    t_start: float | None = None,
    t_end: float | None = None,
) -> np.ndarray:
    """Standard bilinear (in time) voxel grid, polarity in {-1, +1}.

    Returns float32 (n_bins, H, W). Matches the formulation used by
    TimeLens/E-RAFT-style models.
    """
    voxel = np.zeros((n_bins, height, width), dtype=np.float32)
    if events.shape[0] == 0:
        return voxel
    t = events[:, 0].astype(np.float64)
    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    p = events[:, 3].astype(np.float32)
    p = np.where(p > 0, 1.0, -1.0).astype(np.float32)  # {0,1} -> {-1,+1}

    # Normalize timestamps to bin coordinates [0, n_bins-1]; explicit
    # t_start/t_end pin the window to the anchor gap even when the first/last
    # events fall strictly inside it.
    if t_start is None:
        t_start = t[0]
    if t_end is None:
        t_end = t[-1]
    denom = max(t_end - t_start, 1e-9)
    tn = (t - t_start) / denom * (n_bins - 1)

    t0 = np.floor(tn).astype(np.int64)
    dt = (tn - t0).astype(np.float32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    # Split each event bilinearly between its two neighboring temporal bins.
    for b, w_ in ((t0, 1.0 - dt), (t0 + 1, dt)):
        m = valid & (b >= 0) & (b < n_bins) & (w_ > 0)
        np.add.at(voxel, (b[m], y[m], x[m]), p[m] * w_[m])
    return voxel


def save_result_png(out_dir: str | Path, skip: str, seq_name: str, index: int, img: np.ndarray) -> Path:
    """Save float image in [0,1] (H, W, 3) to the submission results layout.

    Note: rint + uint8 here is the final quantization -- the challenge scores
    these saved PNGs, so sub-0.5-grey-level float gains do not survive.
    """
    path = Path(out_dir) / skip / seq_name / f"{index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return path
