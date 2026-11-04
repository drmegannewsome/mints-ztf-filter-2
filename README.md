# mints-ztf-filter-2
Filter ZTF candidates by user choices and visualize the output lightcurves and host systems for a shortlist of user-selected candidates

## How it fits together

* `.github/workflows/scan.yml` runs `ztf_filter_6000.py` every day and commits `docs/candidates.csv`.
* `docs/index.html` (GitHub Pages) is the single review page: it reads `candidates.csv`, lets you narrow the list live
  with the filter bar, shows the ALeRCE and Legacy Survey views for each target, and builds the follow-up list.
  (`docs/candidate_gui.html` only redirects to it.)
* Stage 6 of the scan checks each final candidate for an archival host spectrum with
  `spectra_finding/desi_agngal_crossmatch.py` and stores the result in the `spec_*` columns of `candidates.csv`.

## Archival-spectra check (Stage 6)

The check needs a local copy of the DESI DR1 AGN/Galaxy VAC (`agngal_desi.fits`, needs `pip install fitsio`).

* **GitHub Actions:** the workflow downloads it on the first run and caches it. Nothing to do.
* **Locally:** put the file at `spectra_finding/data/agngal_desi.fits` (git-ignored), or pass `--spec-vac PATH`,
  or set `DESI_AGNGAL_VAC`.
* If the file or `fitsio` is missing the scan still completes; `spec_checked` is `False` and the `spec_*` columns are blank.
* Note this is a match against the VAC, which is a subset of DESI DR1, so "no match" means "not in that catalog".

Useful flags: `--spec-radius ARCSEC` (default 2.0), `--no-spec-check`.
