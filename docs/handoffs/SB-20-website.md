# SB-20 handoff: the 171 A/B integration pair

Stacked on SB-16 (#83). What sunback publishes once `python -m sunback.run.run_ab_pair` runs, and
what is still Gilly's to decide.

## Keys (under SUNBACK_PREFIX, like every reducer write)

- `lab/ab_171_current.png`: today's 171 integrated with the production setting (5 frames, median),
  through RHEF exactly as the Filter Lab's rhef tile.
- `lab/ab_171_candidate.png`: the same newest frame, integrated with the candidate (15 frames,
  `sigclip`: per-pixel samples more than 3 robust sigma (astropy `mad_std`) from the median are
  dropped, then the mean of the rest).
- `lab/ab_171.json`:

```json
{"generated": "...Z",
 "source": {"wave": "171", "frames_available": 15, "newest": "AIA..._0171.fits"},
 "box": [x0, y0, x1, y1],
 "note": "...",
 "current":   {"frames": 5,  "frames_asked": 5,  "method": "median",  "oldest": "...", "t_rec": "...", "std": 0.0, "key": "lab/ab_171_current.png"},
 "candidate": {"frames": 15, "frames_asked": 15, "method": "sigclip", "oldest": "...", "t_rec": "...", "std": 0.0, "key": "lab/ab_171_candidate.png"}}
```

`std` is computed: numpy std of the integrated data (before RHEF) inside `box`, a centred square
whose side is one eighth of the image side. The box is chosen, not checked to be quiet Sun on a
given day; say so on the page. When fewer than 15 frames are listed, `candidate.frames` is the real
count and `frames_asked` stays 15.

## Website (via the Website register)

- A Lab section beside the Filter Lab: the SB-16 swipe control with `current` on the left and
  `candidate` on the right, the settings printed under each half ("5 frames, median" and
  "15 frames, sigma-clipped mean"), both `std` values with the word "computed" and the box drawn or
  given in pixels, and the JSON's `note` line ("Visualization, not a calibrated radiance").
- RHEF cites "Gilly & Cranmer 2025".

## STOP: Gilly's steps

1. Scheduling. This branch does not touch `.github/workflows/GitCloudRunHourly.yml` (a production
   file). The proposed step, after "Daily Filter Lab (SB-16)", same freshness rule:

   ```yaml
   - name: Daily A/B pair (SB-20)
     continue-on-error: true
     env:
       SUNBACK_PREFIX: ${{ inputs.prefix }}
     run: |
       if python devtools/scripts/check_freshness.py --key lab/ab_171.json --prefix "${SUNBACK_PREFIX}" --threshold 82800; then
         echo "A/B pair is under a day old: skip"
       else
         python -m sunback.run.run_ab_pair
       fi
   ```

2. A staging run (`prefix=staging/`), then a week of pairs reviewed by Gilly.
3. Switching production: `SUNBACK_INTEGRATION_FRAMES=15` and `SUNBACK_INTEGRATION_METHOD=sigclip`
   for the reducer, and the Lambda's `INTEGRATION_METHOD` environment value
   (`aws_lambda/video_builder/handler.py`, read into each fragment's `integration` block), so the
   manifest reports the real method. Each is a separate yes.

## Not done, by decision

Noise gating stays as it is: Gilly's 2026-10-15 default is no `noisegate` dependency, sigma
clipping only. `NoiseGateProcessor.py` is unchanged and still imports `noisegate`, which no
dependency file lists.

## Unverified

NRT frames per hour for 171 were not counted here (the container cannot reach JSOC). The
listing module's docstring says about 3-minute cadence, which would put 15 frames at about
45 minutes (estimated); the fetcher scans the current and previous hour buckets.
