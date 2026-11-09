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
| Gaia star | checked once (a star's Gaia verdict never changes); only new objects go to the X-Match each day |
| fewer than 3 g/r epochs | re-screened only once it has gained detections |
| no usable host for iinuclear | retried up to 3 times, then parked |

The workflow commits this file with `candidates.csv`. Records are dropped once an object is >100 days old.
`--reset-state` rebuilds from scratch, `--no-state` ignores it, and `--recheck-not-blue-days N` re-screens young "not blue"
objects that gain detections. If the settings that define a verdict change (blue threshold, epoch rules, `--no-forced`), the file
is ignored and left untouched.

## Time budget and the default 90-day window

The workflow scans the last **90 days** (`--days 90`). `--max-minutes 200` (job limit 240) is there for the first, cold-start run:
newest objects are screened first; when the time is up the scan **saves its progress**, publishes what it has, and
`docs/last_run.txt` says `PARTIAL SCAN`. The next run continues with only the objects not yet decided.

Every stage logs a minute-stamped line, and a watchdog (budget + 25 min) prints every thread's stack and exits non-zero if
anything is still running, so a stall shows up in the log instead of as a silent 4-hour cancellation. The state file is
committed even then.

**If the CDS X-Match service is down** (as it was in Nov 2026): the first failed request trips a circuit breaker and the bulk
star check is skipped for that run (it costs about a minute, not hours). Objects that become candidates are then star-checked one by
one against the Gaia archive ("Stage 4b"), and objects not yet checked are retried in bulk on later runs. Nothing needs changing when
CDS comes back.

Rough cost of a cold start over 90 days (~17,000 objects): ALeRCE photometry runs at ~150 objects/min with `--rate 5`, i.e. about
2 hours; iinuclear is quick (~200 blue objects in under 2 minutes). Once the state file is full, a daily run only handles the ~200 new
objects plus the tracked candidates.

Quick local test of one object (incl. the DESI check): `python ztf_filter_6000.py --target ZTF26abyczrf --spec-vac PATH/agngal_desi.fits`
