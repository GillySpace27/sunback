REM Superseded by .github/workflows/release.yml and RELEASING.md (SB-12, 2026-10-02).
REM Kept for reference; do not use for a release.
git push --follow-tags
python -m twine upload dist/*