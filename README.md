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
* `spec_agn` is `True` when a DESI match is a **QSO** (`SPECTYPE == QSO`, the Redrock fit) or has the VAC's own **AGN_ANY** bit
  (AGN_MASKBITS bit 0: any AGN diagnostic: QuasarNet, MgII, broad lines, BPT, WISE ...). `spec_agn_why` says which. Clean rows
  (ZWARN == 0) are used when there are any, otherwise all matches. `False` = checked, nothing AGN-like (or no DESI match). The GUI
  puts these objects in "In Likely AGN" alongside the Gaia AGN matches.

Useful flags: `--spec-radius ARCSEC` (default 2.0), `--no-spec-check`.

## Memory between runs (`state/scan_state.json`)

The scan remembers what it has decided, so each daily run only screens **new** objects:

| verdict | what happens next |
|---|---|
| not blue (or stopped being blue) | never re-screened |
| blue, but not within `--sigma-max` of its host | never re-screened (the sigma is stored, so changing `--sigma-max` later needs no new iinuclear run) |
| blue and nuclear (a tracked candidate) | only its photometry is refreshed each day (colour, rising, latest mag); nuclear, Gaia-AGN and spectrum results are reused. If it is no longer blue it is dropped |
| Gaia star | checked once (a star's Gaia verdict never changes); only new objects go to the X-Match each day |
| fewer than 4 g+r epochs | re-screened only once it has gained detections |
| no usable host for the nuclear test ("Host Unclear", below) | published as a candidate with a blank sigma, and tracked like any other candidate |

The workflow commits this file with `candidates.csv`. Records are dropped once an object is >100 days old.
`--reset-state` rebuilds from scratch, `--no-state` ignores it, and `--recheck-not-blue-days N` re-screens young "not blue"
objects that gain detections. If the settings that define a verdict change (blue threshold, epoch rules, `--no-forced`), the file
is ignored and left untouched: run once with `--reset-state` to adopt the new settings (a cold start, about 2 hours).

## What a candidate must satisfy

* **Epochs:** at least **3** matched epochs. An epoch is a g night and an r night (nights = detections less than 0.5 d apart,
  per band) taken within **1.5 days** of each other; the i band is never used. `ndet` / `--min-ndet` is only a coarse pre-filter and
  counts alerts in every band, so it says nothing about g/r. The g/r points can be alerts or forced photometry (S/N > ~3.6);
  `candidates.csv` has `n_alert_g`, `n_alert_r`, `n_alert_i` and `n_forced_gr` so you can see which.
* **Positive differences only:** a g/r point counts only if the source is *brighter* than the reference image (`isdiffpos` = +1, alerts and
  forced photometry alike). A negative difference is a subtraction residual or a variable star dipping below its reference, not a transient
  rising (ZTF22abieeab had five negative i-band alerts and g/r "detections" that were all negative residuals at 16 mag). `n_negative_gr` in the CSV
  counts what was dropped; `REQUIRE_POSITIVE_DIFF` in the script turns the rule off.
* **Blue:** the latest 3 matched epochs have a weighted g-r < -0.05.
* **Nuclear:** only `nuclear_sigma <= --sigma-max`, default **2.4477**, which is iinuclear's own p > 0.05. `nuclear_sigma` is
  sqrt(chi2) of the 2-D offset between the mean ZTF position and the host, so p = exp(-sigma^2/2): sigma 1 = p 0.61, 2 = 0.135,
  2.45 = 0.05, 3 = 0.011. There is no separate p-value cut; `is_nuclear` / `nuclear_pvalue` are iinuclear's p > 0.05 flag and the
  p-value (a row kept with `--sigma-max 3` can show `is_nuclear = False`).
* **Not a star** (Gaia, below) and not a CV/Nova. Spectra, Gaia-AGN, host type: informational only.
* **Host Unclear:** an object with no usable host in any reference is *not* dropped. It is kept as a candidate with `host_unclear = True`,
  a blank `nuclear_sigma` and the reason in `host_note`, goes through the Gaia, AGN and spectrum stages like the rest, and the GUI lists it in
  its own "Host Unclear" bin below "In Likely AGN" (an object that is host-unclear and also AGN-matched stays in Host Unclear; its AGN badge
  still shows). A host reference counts as unusable when its centre is **more than 5" from the ZTF position and its error is 3" or more**
  (`HOST_UNCLEAR_SEP_ARCSEC`, `HOST_UNCLEAR_ERR_ARCSEC`); both conditions are needed. An unusable reference falls through to the next one
  (LS, PS1, SDSS). Such a reference is almost never the transient's host, and with a large error it passes any sigma test.
  The GUI sigma filter never hides a Host Unclear row (it has no sigma).

**Re-checking stars on candidates that are already published** (`--recheck-stars`, off by default): re-runs the one-by-one Gaia
archive check on every candidate about to be published, ignoring the cached verdicts, and prints one line per candidate
(`STAR` / `not a star`, the Gaia source's separation, G, pm and parallax significance, or "no Gaia source within 2"").
The per-object Gaia archive calls share 3 slots and make 2 attempts (rate-limit responses wait 5x longer between them) (the archive allows 5 parallel jobs per user and rejects beyond its queue). Newly rejected objects are banked as stars and never come back. Nothing else is re-screened. A bulk X-Match request of 100+ objects
that returns no rows at all is now treated as a failed request (retried later) rather than "no star here".

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

**Service errors are never turned into verdicts.** iinuclear hides its own network failures, so a PanSTARRS or ALeRCE outage used to
look like "no host". The scan now tells the two apart: a real "no PS1/SDSS host" is recorded (and retried up to 3 times), while a failed
lookup is retried in place, left undecided, and stops the nuclear stage if 20 in a row fail. ALeRCE 429s slow every thread down together
and the pace recovers on its own.

**No second ALeRCE fetch for the nuclear test.** iinuclear's `get_data()` re-downloads each object's detections from ALeRCE, which Stage 2
already has. The scan passes those positions in, so the nuclear stage makes no ALeRCE calls. The sigma is still computed with iinuclear's own
`get_galaxy_center` and `check_nuclear`; only the host reference and its error are chosen here.

**Host reference: Legacy Survey, then PS1, then SDSS.** For each object the first reference that has a host within 3" is used:
1. Legacy Survey DR10 Tractor (NOIRLab Data Lab, plain `requests`, anonymous token; no `astro-datalab` install needed, and please do not install it,
   it pins old numpy/pandas/astropy). Error = sqrt(formal Tractor error^2 + 0.2"^2); the floor dominates.
2. PS1 DR2 (MAST), with iinuclear's own error model. Legacy Survey does not cover the Galactic plane, so low-|b| objects land here.
3. SDSS DR18 (SkyServer) as a last resort. SkyServer has been returning HTTP 403 pages intermittently, which iinuclear reports as a header-column error.
A reference whose centre is >5" away with an error >=3" is skipped too (see Host Unclear). A reference that errors is skipped and recorded; if nothing has a host and any reference errored, the object is a service error (retried later, not
cached as "no host"). Only when every reference answered and none has a host is it "no host". `nuclear_ref` and `host_type` (Tractor morphology, e.g.
PSF/REX/SER) are saved per object and in the CSV. `--no-legacy-survey` restores the PS1-then-SDSS behaviour.
The state version is now 4 (host-unclear records carry their full row): run once with `--reset-state`. A Gaia query that still fails after its 2 attempts leaves the object in (kept and retried next run, never banked as "not a star").

`python ls_reference.py --smoke` checks the service and column names; `python ls_reference.py --at RA DEC` diagnoses one position;
`python compare_references.py --from-state state/scan_state.json [--status nohost]` compares the old and Tractor references on the same detections.

**Runs never overlap or go stale.** Scans share one queue (`concurrency`), each checks out the latest commit when it starts, a daily
run that GitHub starts less than 10 h after another scan is skipped, results are also saved as a workflow artifact (14 days), and the
push prefers the new results if the branch moved. Nothing re-triggers a scan automatically: the only triggers are the daily cron and the
manual button (GitHub often starts the cron run hours late).

Rough cost of a cold start over 90 days (~17,000 objects): ALeRCE photometry runs at ~150 objects/min with `--rate 5`, i.e. about
2 hours; iinuclear is quick (~200 blue objects in under 2 minutes). Once the state file is full, a daily run only handles the ~200 new
objects plus the tracked candidates.

Quick local test of one object (incl. the DESI check): `python ztf_filter_6000.py --target ZTF26abyczrf --spec-vac PATH/agngal_desi.fits`

## Looking at one object (`inspect_object.py`)

`python inspect_object.py ZTF22abieeab` prints what the scan sees for that object: the ALeRCE object-table row (name year vs
`firstmjd`), the raw band codes and counts in the alert detections and in the forced photometry, the g/r nights and matched
epochs that result, the Gaia star check (2" cone, a dense-field sanity cone, and the nearest sources out to 6"), the Legacy Survey sources (DR10 and DR9), and the host reference and nuclear sigma exactly as the scan computes them. Use it when a
candidate looks wrong (star, i-band-only data).
