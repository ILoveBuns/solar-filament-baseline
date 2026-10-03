# Final submission checklist

Deadline recorded from the official competition announcement: **2026-11-15**. Recheck the live announcement before submission.

| Requirement | Current evidence | State / next action |
|---|---|---|
| Strongest verified method | Classical pipeline, platform ref `55113085`, public score `0.48` | Ready; entrant should confirm in Kaggle UI |
| End-to-end notebook | All four code cells in `kaggle/final_classical_pipeline.ipynb` completed locally against all 180 official test images; receipt matches CLI and July output | Ready; entrant should preserve a final Kaggle-hosted copy |
| Pinned environment | Minimal final route in `requirements.txt`; non-selected experiments in `requirements-experiments.txt` | Ready; final notebook installs only the final route |
| Public source repository | Source and MIT license present | **Blocked on entrant authorization:** repository is still private |
| Four-page technical report | Sources plus `report/solar-filament-report.pdf`; automated gate in `scripts/verify_report.sh` | Ready: exactly four pages, embedded fonts, no overflow or undefined references; entrant must confirm author/affiliation |
| Reproducibility evidence | Unit tests, notebook schema/decode checks, and `FINAL_CLASSICAL_EXECUTION.md` (11,520 audited rows; zero overlaps; deterministic SHA-256) | Ready locally; preserve final Kaggle output before filing |
| PQ-calibrated comparison | `PQ_CALIBRATION.md`; final `q=0.18` candidate aggregate 70-image untouched PQ 0.10831 vs 0.06000 (1.81×); separate 2,160-row full CSV audited with SHA-256 `fa4de92b…1bc9c` | **Platform validation required:** compare only if submissions remain open; do not replace 0.48 from local evidence |
| Google Form submission | Organizer's final-submission form | **Entrant-only:** identity, contact details, declarations, PDF/repo links |

## Final gate

1. Confirm refs `55113085`, `55354276`, and `55354612` and their displayed scores in Kaggle.
2. If the platform still permits a comparison, submit the exact ignored
   `outputs/submission-pq-calibrated-q018.csv`, record its reference/score, and retain the `0.48` route
   unless this candidate is officially better.
3. Run the selected final classical notebook with the official dataset attached; compare its CSV against `FINAL_CLASSICAL_EXECUTION.md`, download it, and preserve the executed notebook.
4. Confirm author/affiliation fields, then run the report gate (or import the three LaTeX files into the official Overleaf project) and preserve the verified four-page PDF.
5. Review repository history for accidental secrets or non-redistributable data; do not commit competition data or generated submissions.
6. Explicitly authorize making the repository public, then verify the anonymous/public URL.
7. Personally complete the Google Form and save its confirmation receipt.
