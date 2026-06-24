"""Multi-frame time integration for NRT solar imagery.

Collapses N co-registered frames of the same wavelength into a single frame using
a selectable, NaN-aware reduction. ``median`` is the default because it rejects
cosmic-ray hits and transient flare spikes; ``mean``/``sum`` boost signal-to-noise.

This is deliberately decoupled from any data source (unlike the Fido-bound
``FidoTimeIntProcessor``) and operates on plain arrays so it is trivially testable.
"""
import numpy as np

_REDUCERS = {
    "median": np.nanmedian,
    "mean": np.nanmean,
    "sum": np.nansum,
}


def integrate_frames(frames, method="median"):
    """Reduce a stack of same-shape 2D frames to one float32 frame.

    Args:
        frames: non-empty sequence of 2D arrays, all the same shape.
        method: one of ``"median"``, ``"mean"``, ``"sum"``.

    Returns:
        A 2D ``float32`` array. With a single input frame the result equals that
        frame (regression-safe with the previous single-frame pipeline).
    """
    if method not in _REDUCERS:
        raise ValueError(
            f"Unknown integration method {method!r}; expected one of {sorted(_REDUCERS)}"
        )
    if len(frames) == 0:
        raise ValueError("integrate_frames requires at least one frame")

    stack = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    return _REDUCERS[method](stack, axis=0).astype(np.float32)
