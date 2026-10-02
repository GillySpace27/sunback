# ci-pending

Config that is written and checked but deliberately not active.

- `dependabot.yml` (SB-13): parked because, in `.github/`, it would start weekly pip PRs on merge
  against `requirements-server.txt` and `requirements-exact.txt`, which change nothing that runs.
  To activate: move it back to `.github/dependabot.yml` in the same change that adds
  `requirements-reducer.txt` (SB-13, frozen from the production image; needs Gilly's yes).
