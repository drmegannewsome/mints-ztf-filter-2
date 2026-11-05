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

## Memory between runs (`state/scan_state.json`)

The scan remembers what it has decided, so each daily run only screens **new** objects:

| verdict | what happens next |
|---|---|
| not blue (or stopped being blue) | never re-screened |
| blue, but not within `--sigma-max` of its host | never re-screened (the sigma is stored, so changing `--sigma-max` later needs no new iinuclear run) |
| blue and nuclear (a tracked candidate) | only its photometry is refreshed each day (colour, rising, latest mag); nuclear, Gaia-AGN and spectrum results are reused. If it is no longer blue it is dropped |
| fewer than 3 g/r epochs | re-screened only once it has gained detections |
| no usable host for iinuclear | retried up to 3 times, then parked |

The workflow commits this file with `candidates.csv`. Records are dropped once an object is >100 days old.
`--reset-state` rebuilds from scratch, `--no-state` ignores it, and `--recheck-not-blue-days N` re-screens young "not blue"
objects that gain detections. If the settings that define a verdict change (blue threshold, epoch rules, `--no-forced`), the file
is ignored and left untouched.

## Time budget

`--max-minutes 200` (job limit is 240) is there for the first, cold-start run. Newest objects are screened first; when the time is
up the scan **saves its progress**, publishes what it has, and `docs/last_run.txt` says `PARTIAL SCAN`. The next run continues
with only the objects not yet decided. HTTP calls also have timeouts so one stalled request cannot hang the job.

Quick local test of one object (incl. the DESI check): `python ztf_filter_6000.py --target ZTF26abyczrf --spec-vac PATH/agngal_desi.fits`
