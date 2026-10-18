# SB-16 handoff to the Website register: swipe and Filter Lab

What sunback publishes once SB-16 is merged and the Lambda is redeployed (Gilly's steps). Keys and
fields are in `aws_lambda/video_builder/CONTRACT.md`.

## Swipe in every AIA lightbox

- Left: `raw1k` from the product's fragment (`raw/rhef_<id>_raw_1k.png`), the same integrated frame
  with a plain log stretch. Right: `img1k` (the RHEF still). Same size and orientation, pixel for pixel.
- Only the nine AIA singles carry `raw1k`. Rainbow, composite_uv and dem have no raw view in v1: say
  "no raw view for composites" instead of hiding the control silently.
- When `raw1k` is absent, show the still alone.
- Control: two stacked images, a range input driving `clip-path` on the top one, arrow keys step it,
  both halves labelled ("log stretch" and "RHEF"), and the line "Visualization, not a calibrated radiance."
- The raw view is written after the stills, so for a few minutes it can be one run older than `img1k`.

## Filter Lab (lab section or lab.html)

- Data: `lab/filter_lab.json`; images: each tile's `key` (`lab/filter_<name>_171.png`).
- Show the tiles in the JSON's order, the `params` under each tile, `sunkit_image_version`,
  `source.t_rec`, and every `skipped` entry with its error (do not hide them).
- Swipe between any two tiles with the same control as above.
- Captions: RHEF and Upsilon cite "Gilly & Cranmer 2025"; the others say "citation: see the
  sunkit-image documentation". Carry the JSON's `note` line.
- The lab is regenerated at most once a day; `generated` is its time.
