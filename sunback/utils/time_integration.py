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

# SB-20: "sigclip" rejects, per pixel, samples more than SIGCLIP_SIGMA robust standard
# deviations (astropy mad_std) from the median, then takes the mean of the rest. The
# robust spread matters: with a plain standard deviation a single hit can never sit
# more than (N-1)/sqrt(N) sigma out, so at sigma 3 nothing is clipped below N = 11.
SIGCLIP_SIGMA = 3.0
METHODS = tuple(sorted(_REDUCERS)) + ("sigclip",)


def _sigclip_mean(stack):
    import warnings

    from astropy.stats import sigma_clip
    from astropy.utils.exceptions import AstropyUserWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)  # NaN pixels are masked, as nanmean skips them
        clipped = sigma_clip(stack, sigma=SIGCLIP_SIGMA, maxiters=None, cenfunc="median", stdfunc="mad_std",
                             axis=0, masked=True)
    return np.ma.filled(clipped.mean(axis=0).astype(np.float32), np.nan)


def integrate_frames(frames, method="median"):
    """Reduce a stack of same-shape 2D frames to one float32 frame.

    Args:
        frames: non-empty sequence of 2D arrays, all the same shape.
        method: one of ``"median"``, ``"mean"``, ``"sum"``, ``"sigclip"`` (SB-20).

    Returns:
        A 2D ``float32`` array. With a single input frame the result equals that
        frame (regression-safe with the previous single-frame pipeline).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown integration method {method!r}; expected one of {list(METHODS)}")
    if len(frames) == 0:
        raise ValueError("integrate_frames requires at least one frame")

    stack = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    if method == "sigclip":
        return _sigclip_mean(stack)
    return _REDUCERS[method](stack, axis=0).astype(np.float32)
