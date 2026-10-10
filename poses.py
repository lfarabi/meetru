"""Recorded hand poses: build a template from a few seconds of frames, then match live hands to it."""
from collections import Counter

import numpy as np

from tracking import WRIST, Hand, palm_size

RADIUS_FLOOR = 0.05      # a perfectly still recording would otherwise give a radius no live hand could meet
DEFAULT_SENSITIVITY = 1.5
_PAIRS = np.triu_indices(21, 1)


def hand_descriptor(world: np.ndarray) -> np.ndarray:
    """All 210 landmark-pair distances in palm sizes: unchanged by rotation, scale and mirroring.

    # ponytail: shape only, not direction (thumbs-up == thumbs-down); add the palm normal to the descriptor if users ask
    """
    d = np.linalg.norm(world[:, None] - world[None], axis=2)[_PAIRS]
    return d / palm_size(world)


def _descriptor(hands: list[Hand]) -> np.ndarray:
    """One hand: its descriptor. Two hands: both, plus how far apart the wrists are (pixels, since each
    hand's world landmarks are centred on itself) in palm sizes."""
    parts = [hand_descriptor(h.world) for h in hands]
    if len(hands) == 2:
        a, b = hands
        gap = np.linalg.norm(a.lm[WRIST, :2] - b.lm[WRIST, :2]) / np.mean([palm_size(a.lm), palm_size(b.lm)])
        parts.append(np.array([gap]))
    return np.concatenate(parts)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_template(name: str, frames: list[list[Hand]]) -> dict:
    """Average the frames into a JSON-ready template; raises ValueError("no_hand" | "unsteady")."""
    counts = Counter(len(f) for f in frames if f)
    if not counts:
        raise ValueError("no_hand")
    n, seen = counts.most_common(1)[0]
    if seen < len(frames) / 2:
        raise ValueError("unsteady")
    # Left-to-right, so a 2-hand template always lists the hands in the same order.
    samples = np.array([_descriptor(sorted(f, key=lambda h: h.lm[WRIST, 0])) for f in frames if len(f) == n])
    mean = samples.mean(axis=0)
    spread = np.sqrt(np.mean((samples - mean) ** 2, axis=1))
    return {"name": name, "hands": n, "mean": mean.tolist(),
            "radius": max(RADIUS_FLOOR, float(np.percentile(spread, 95))), "sensitivity": DEFAULT_SENSITIVITY}


def distance(template: dict, hands: list[Hand], primary: Hand | None) -> float | None:
    """RMS gap between the live hands and the template; None when the hand count doesn't fit."""
    mean = np.array(template["mean"])
    if template["hands"] == 1:
        return None if primary is None else _rms(_descriptor([primary]), mean)
    if len(hands) != 2:
        return None
    # Either hand may be on the left (the user can swap them or mirror the pose), so try both orders.
    return min(_rms(_descriptor(order), mean) for order in (hands, hands[::-1]))


def score(template: dict, hands: list[Hand], primary: Hand | None) -> float | None:
    """1 = identical, 0 = at the edge of the template's tolerance, negative = outside it."""
    d = distance(template, hands, primary)
    return None if d is None else 1 - d / (template["radius"] * template["sensitivity"])


def best_match(templates: list[dict], hands: list[Hand], primary: Hand | None) -> str | None:
    """Name of the template the hands fit best (score > 0), else None."""
    scored = [(s, t["name"]) for t in templates if (s := score(t, hands, primary)) is not None and s > 0]
    return max(scored)[1] if scored else None
