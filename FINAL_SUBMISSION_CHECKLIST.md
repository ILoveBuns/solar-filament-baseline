# Final submission checklist

Deadline recorded from the official competition announcement: **2026-11-15**. Recheck the live announcement before submission.

| Requirement | Current evidence | State / next action |
|---|---|---|
| Strongest verified method | Classical pipeline, platform ref `55113085`, public score `0.48` | Ready; entrant should confirm in Kaggle UI |
| End-to-end notebook | `kaggle/final_classical_pipeline.ipynb` | Ready for an official-data Kaggle run |
| Pinned environment | Minimal final route in `requirements.txt`; non-selected experiments in `requirements-experiments.txt` | Ready; final notebook installs only the final route |
| Public source repository | Source and MIT license present | **Blocked on entrant authorization:** repository is still private |
| Four-page technical report | `report/main.tex`, `report/preamble.tex`, `report/main.bib` | Official `acmart/sigconf` structure ready; compile in the organizer's Overleaf project, inspect layout, and export one four-page PDF |
| Reproducibility evidence | Unit tests plus notebook schema/decode checks | Run final notebook and preserve outputs before filing |
| Google Form submission | Organizer's final-submission form | **Entrant-only:** identity, contact details, declarations, PDF/repo links |

## Final gate

1. Confirm refs `55113085`, `55354276`, and `55354612` and their displayed scores in Kaggle.
2. Run the final classical notebook with the official dataset attached; download the CSV and preserve the executed notebook.
3. Import the three LaTeX files into the official Overleaf project, confirm author/affiliation fields, compile, and verify exactly four pages.
4. Review repository history for accidental secrets or non-redistributable data; do not commit competition data or generated submissions.
5. Explicitly authorize making the repository public, then verify the anonymous/public URL.
6. Personally complete the Google Form and save its confirmation receipt.
