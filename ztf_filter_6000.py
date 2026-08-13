"""
early_color_classifier.py

A FLEET-inspired pipeline that scans ALeRCE for recently discovered ZTF
transients and flags ones that are currently *blue*, *not a star*, *not a
CV/Nova*, and *nuclear* (within ~1 sigma of their host center).

Pipeline (cheap/bulk filters first, expensive per-object ones last):

  Stage 1   Recency  : ALeRCE objects whose FIRST detection is within the last
                       ~3 months (firstmjd window), ndet>=5. One paginated query.
  Stage 1a  CV/Nova  : ONE bulk query for objects the forced-photometry classifier
                       calls CV/Nova >= 0.3, and subtract them. Objects without
                       that classifier are kept.
  Stage 1b  Not star : bulk Gaia DR3 crossmatch (CDS X-Match) on the object-table
                       positions; drop significant proper-motion/parallax sources.
  Stage 2   Photometry: pull ZTF *difference* photometry -- alert (query_detections)
                       AND forced (query_forced_photometry) merged -- build matched
                       g/r epochs, keep objects with >=3 matched epochs.
  Stage 3   Blue     : weighted-mean g-r over the 3 latest matched epochs < -0.05.
  Stage 4   Nuclear  : iinuclear on the survivors; keep those within `sigma_max`.

A g/r "epoch" = a g and r measurement within 2 days of each other. Rise/fall
compares the latest epoch against the weighted mean of the prior 2.

Inspired by FLEET (Gomez et al.): https://github.com/gmzsebastian/fleet
Nuclear test via iinuclear:        https://github.com/gmzsebastian/iinuclear

Requirements:  pip install alerce iinuclear numpy pandas astropy astroquery matplotlib

CLI examples:
    python early_color_classifier.py --days 60 --min-ndet 5 --workers 1
    python early_color_classifier.py --plot --plotdir figs --out candidates.csv
    python early_color_classifier.py --no-forced --keep-cvnova   # ablations
"""

import os
import time
import threading
import contextlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from alerce.core import Alerce
try:
    from alerce.exceptions import ObjectNotFoundError
except Exception:
    class ObjectNotFoundError(Exception):
        pass

# --------------------------------------------------------------------------- #
#  astroquery<->MAST workaround: MAST sometimes returns the string "None" for
#  nulls and astroquery tries to cast it to float ("could not convert string to
#  float: 'None'"). Sanitize to real None (-> NaN) so iinuclear's PanSTARRS query
#  works. Remove once a fixed astroquery is released.
# --------------------------------------------------------------------------- #
import astroquery.mast.services as _mast_svc

_orig_json_to_table = _mast_svc._json_to_table

def _json_to_table_safe(json_obj, *args, **kwargs):
    data_key = kwargs.get("data_key", args[0] if args else "data")
    if isinstance(json_obj, dict) and data_key in json_obj:
        for row in json_obj[data_key]:
            if isinstance(row, dict):
                for k, v in row.items():
                    if v == "None":
                        row[k] = None
    return _orig_json_to_table(json_obj, *args, **kwargs)

_mast_svc._json_to_table = _json_to_table_safe

from astroquery.gaia import Gaia
from iinuclear.utils import get_data, get_galaxy_center, check_nuclear

# ZTF filter ids in ALeRCE
FID_G, FID_R = 1, 2

# Tunables
COLOR_BLUE_THRESHOLD = -0.05   # g - r < this  ==>  "blue"
GR_MATCH_WINDOW_DAYS = 2.0     # max |t_g - t_r| to call a g/r pair one epoch
NIGHT_GAP_DAYS = 0.5           # gap that separates one night from the next (per band)
MIN_EPOCHS = 3                 # need 3 matched epochs (latest 1 vs prior 2; color uses 3)
N_COLOR_EPOCHS = 3             # color uses the 3 latest matched epochs
MAG_ERR_FLOOR = 0.01           # guard against zero/NaN photometric errors
MAX_AGE_DAYS = 90.0            # "found in the last 3 months"
MIN_DETECTIONS = 5             # coarse Stage-1 ndet pre-filter
NUCLEAR_SIGMA_MAX = 1.0        # keep transients within this many sigma of host
FORCED_MAX_MAGERR = 0.30       # keep forced-phot points at least ~3-sigma (SNR>~3)

# CV/Nova rejection (forced-photometry light-curve classifier)
CVNOVA_CLASSIFIER = "lc_classifier_BHRF_forced_phot"  # verify via query_classifiers()
CVNOVA_CLASS = "CV/Nova"
CVNOVA_PROB = 0.3

# Gaia stellar-rejection thresholds (FLEET-style)
GAIA_MATCH_RADIUS_ARCSEC = 2.0
GAIA_PM_SIG_THRESH = 5.0
GAIA_PLX_SIG_THRESH = 5.0
GAIA_PROXIMITY_ARCSEC = 1.0    # reject transients within this of a (bright) Gaia star
GAIA_PROXIMITY_MAXMAG = 19.0   # ...only if that Gaia source is at least this bright (Gmag)
# Gaia DR3 QSO-candidate annotation (Stage 5; informational, never a cut)
GAIA_AGN_MATCH_RADIUS_ARCSEC = 2.0
GAIA_AGN_MIN_PROB = 0.25        # 0 = flag any qso_candidates match; raise to purify

# Index of catalog_result in get_data()'s return; ras/decs are 0,1.
CATALOG_INDEX = 4
# get_galaxy_center -> (ra, dec, mean_ra_offset, mean_dec_offset, error_arcsec)
# check_nuclear     -> (sigma, chi2, p_value, is_nuclear)
GALCENTER_RA_IDX, GALCENTER_DEC_IDX, GALCENTER_ERR_IDX = 0, 1, 4
NUCLEAR_SIGMA_IDX, NUCLEAR_PVAL_IDX, NUCLEAR_FLAG_IDX = 0, 2, 3


# --------------------------------------------------------------------------- #
#  Utilities
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _silence(active=True):
    """Mute chatty third-party prints/warnings. Redirecting sys.stdout/stderr and
    the warnings filters mutates GLOBAL state (not thread-safe), so only do it on
    the main thread; in worker threads this is a no-op (chattier but safe)."""
    if not active or threading.current_thread() is not threading.main_thread():
        yield
        return
    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _astuple(x):
    return x if isinstance(x, tuple) else (x,)


_TRANSIENT_SIGNATURES = ("500", "502", "503", "504", "timeout", "timed out",
                         "gateway", "temporarily unavailable", "connection")
_RATE_LIMIT_SIGNATURES = ("403", "429", "forbidden", "too many requests", "rate limit")


def _is_transient_error(exc):
    msg = str(exc).lower()
    return any(s in msg for s in _TRANSIENT_SIGNATURES)


def _is_rate_limit_error(exc):
    msg = str(exc).lower()
    return any(s in msg for s in _RATE_LIMIT_SIGNATURES)


class _RateLimiter:
    """Global throttle: spaces request starts >= 1/max_per_sec apart across all
    threads. max_per_sec<=0 disables it."""
    def __init__(self, max_per_sec):
        self.min_interval = (1.0 / max_per_sec) if (max_per_sec and max_per_sec > 0) else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next = max(now, self._next) + self.min_interval


_RATE_LIMITER = _RateLimiter(0)


def _with_retries(func, *args, _tries=5, _base_delay=2.0, _quiet=False, **kwargs):
    """Throttled call that retries transient (5xx/timeout) and rate-limit (403/429)
    errors with exponential backoff; other errors raise immediately."""
    for attempt in range(1, _tries + 1):
        _RATE_LIMITER.acquire()
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            rate_limited = _is_rate_limit_error(exc)
            if attempt >= _tries or not (rate_limited or _is_transient_error(exc)):
                raise
            delay = (5.0 if rate_limited else _base_delay) * (2 ** (attempt - 1))
            if not _quiet:
                kind = "rate-limited (403/429)" if rate_limited else "transient API error"
                print(f"    [{kind}; attempt {attempt}/{_tries}, backing off {delay:.0f}s]")
            time.sleep(delay)


_thread_local = threading.local()


def _get_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = Alerce()
    return _thread_local.client


def _parallel_map(func, items, workers, on_error=None):
    """Apply func(item) across items with `workers` threads (I/O-bound); returns
    (item, result) pairs for calls that didn't raise. workers<=1 is sequential."""
    results = []
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(func, it): it for it in items}
            for fut in as_completed(futures):
                it = futures[fut]
                try:
                    results.append((it, fut.result()))
                except Exception as exc:
                    if on_error:
                        on_error(it, exc)
    else:
        for it in items:
            try:
                results.append((it, func(it)))
            except Exception as exc:
                if on_error:
                    on_error(it, exc)
    return results


def weighted_mean(values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    errors = np.where(np.isfinite(errors) & (errors > 0), errors, MAG_ERR_FLOOR)
    w = 1.0 / errors**2
    return float(np.sum(w * values) / np.sum(w)), float(np.sqrt(1.0 / np.sum(w)))


# --------------------------------------------------------------------------- #
#  Stage 1 / 1a: recent discoveries and the CV/Nova exclusion set
# --------------------------------------------------------------------------- #
def _safe_query_objects(client, **params):
    try:
        return _with_retries(client.query_objects, **params)
    except TypeError:
        params.pop("survey", None)
        try:
            return _with_retries(client.query_objects, **params)
        except ObjectNotFoundError:
            return None
    except ObjectNotFoundError:
        return None


def query_recent_discoveries(client, days=MAX_AGE_DAYS, min_ndet=MIN_DETECTIONS,
                             max_objects=None, classifier=None, class_name=None,
                             probability=None, page_size=500, max_pages=400):
    """DataFrame (oid, meanra, meandec) of objects with firstmjd within `days`,
    ndet>=min_ndet (+ optional classifier cut). Positions come from the object
    table -- no lightcurves pulled here."""
    now = Time.now().mjd
    base = dict(survey="ztf",
                firstmjd=[now - days, now],
                ndet=[min_ndet, 10_000_000],
                order_by="firstmjd", order_mode="DESC",
                page_size=page_size, format="pandas")
    if classifier:
        base["classifier"] = classifier
    if class_name:
        base["class_name"] = class_name
    if probability is not None:
        base["probability"] = probability

    frames, page = [], 1
    while page <= max_pages:
        df = _safe_query_objects(client, page=page, **base)
        if df is None or len(df) == 0:
            break
        frames.append(df)
        if max_objects and sum(len(f) for f in frames) >= max_objects:
            break
        if len(df) < page_size:
            break
        page += 1

    if not frames:
        return pd.DataFrame(columns=["oid", "meanra", "meandec"])
    out = pd.concat(frames, ignore_index=True)
    if max_objects:
        out = out.iloc[:max_objects]
    ra_col = next((c for c in ("meanra", "ra") if c in out.columns), None)
    dec_col = next((c for c in ("meandec", "dec") if c in out.columns), None)
    return pd.DataFrame({
        "oid": out["oid"].astype(str).values,
        "meanra": out[ra_col].values if ra_col else np.nan,
        "meandec": out[dec_col].values if dec_col else np.nan,
    })


def query_cvnova_oids(client, days=MAX_AGE_DAYS, min_ndet=MIN_DETECTIONS,
                      classifier=CVNOVA_CLASSIFIER, class_name=CVNOVA_CLASS,
                      probability=CVNOVA_PROB):
    """Set of oids the forced-photometry classifier calls CV/Nova >= probability,
    among recent objects -- one bulk query. Objects lacking that classifier aren't
    returned (so they're kept). Fails open (empty set) on error."""
    try:
        df = query_recent_discoveries(client, days, min_ndet, classifier=classifier,
                                      class_name=class_name, probability=probability)
        return set(df["oid"].tolist())
    except Exception as exc:
        print(f"    [CV/Nova pre-filter failed ({classifier}/{class_name}): {exc}; "
              f"skipping this cut -- verify the classifier name via query_classifiers()]")
        return set()


# --------------------------------------------------------------------------- #
#  Stage 2: difference photometry (alert + forced) -> matched g/r epochs
# --------------------------------------------------------------------------- #
def get_difference_photometry(oid, client, use_forced=True, debug_forced=False):
    """Return (bands, ra, dec). bands = {1:(mjd,mag,err) g, 2:(...) r}, combining
    alert difference photometry (query_detections: magpsf/sigmapsf) and -- if
    use_forced -- significant ZTF forced difference photometry. ra/dec = median
    detection position. Set debug_forced=True to print the forced-photometry
    columns and how many points were merged (use on a single object)."""
    def _q(fn):
        try:
            return _with_retries(fn, oid, survey="ztf", format="pandas")
        except TypeError:
            return _with_retries(fn, oid, format="pandas")

    det = _q(client.query_detections)
    if det is None or len(det) == 0:
        raise ValueError(f"No detections for {oid}")

    ra_all = np.asarray(det["ra"], dtype=float)
    dec_all = np.asarray(det["dec"], dtype=float)
    pos_ok = np.isfinite(ra_all) & np.isfinite(dec_all)
    ra0 = float(np.median(ra_all[pos_ok])) if pos_ok.any() else np.nan
    dec0 = float(np.median(dec_all[pos_ok])) if pos_ok.any() else np.nan

    per_band = {FID_G: [[], [], []], FID_R: [[], [], []]}   # mjd, mag, err

    def _add(mjd, fid, mag, err, max_err=None):
        mjd = np.asarray(mjd, dtype=float)
        fid = np.asarray(fid, dtype=float)
        mag = np.asarray(mag, dtype=float)
        err = np.asarray(err, dtype=float)
        n_kept, n_finite = 0, 0
        for f in (FID_G, FID_R):
            base = ((fid == f) & np.isfinite(mjd) & np.isfinite(mag)
                    & np.isfinite(err) & (err > 0))
            n_finite += int(base.sum())
            sel = (base & (err <= max_err)) if max_err is not None else base
            per_band[f][0].extend(mjd[sel])
            per_band[f][1].extend(mag[sel])
            per_band[f][2].extend(err[sel])
            n_kept += int(sel.sum())
        return n_kept, n_finite

    _add(det["mjd"], det["fid"], det["magpsf"], det["sigmapsf"])

    if use_forced:
        try:
            fp = _q(client.query_forced_photometry)
        except Exception as exc:
            fp = None
            if debug_forced:
                print(f"    [forced photometry query failed for {oid}: {exc}]")
        if fp is not None and len(fp) > 0:
            magcol = next((c for c in ("mag", "magpsf", "mag_tot", "mag_corr") if c in fp.columns), None)
            errcol = next((c for c in ("e_mag", "sigmapsf", "emag", "mag_err", "e_mag_corr") if c in fp.columns), None)
            fidcol = "fid" if "fid" in fp.columns else next((c for c in ("band", "filterid") if c in fp.columns), None)
            if magcol and errcol and "mjd" in fp.columns and fidcol:
                n_kept, n_finite = _add(fp["mjd"], fp[fidcol], fp[magcol], fp[errcol], max_err=FORCED_MAX_MAGERR)
                if debug_forced:
                    print(f"    [forced: {len(fp)} rows; {n_finite} finite g/r detections, "
                          f"kept {n_kept} with e_mag<={FORCED_MAX_MAGERR} "
                          f"(mag='{magcol}' err='{errcol}' fid='{fidcol}')]")
            elif debug_forced:
                print(f"    [forced: {len(fp)} rows but no usable columns; have "
                      f"{list(fp.columns)}]")
        elif debug_forced:
            print(f"    [forced photometry empty for {oid}]")

    bands = {}
    for f in (FID_G, FID_R):
        mjd = np.asarray(per_band[f][0], dtype=float)
        mag = np.asarray(per_band[f][1], dtype=float)
        err = np.asarray(per_band[f][2], dtype=float)
        order = np.argsort(mjd)
        bands[f] = (mjd[order], mag[order], err[order])
    return bands, ra0, dec0


def collapse_nightly(mjd, mag, err, gap_days=NIGHT_GAP_DAYS):
    if len(mjd) == 0:
        return np.array([]), np.array([]), np.array([])
    nm, na, ne, start = [], [], [], 0
    for i in range(1, len(mjd) + 1):
        if i == len(mjd) or (mjd[i] - mjd[i - 1]) > gap_days:
            m, e = weighted_mean(mag[start:i], err[start:i])
            nm.append(float(np.mean(mjd[start:i]))); na.append(m); ne.append(e)
            start = i
    return np.array(nm), np.array(na), np.array(ne)


def build_gr_epochs(bands, match_window=GR_MATCH_WINDOW_DAYS):
    g_mjd, g_mag, g_err = collapse_nightly(*bands[FID_G])
    r_mjd, r_mag, r_err = collapse_nightly(*bands[FID_R])
    # Consider every g/r pair within the window and take the CLOSEST pairs first
    # (each night used once). This near-maximum matching avoids the old
    # time-ordered greedy leaving a tight (e.g. forced-phot) pair unmatched
    # because a neighbouring night "stole" its partner.
    candidates = []
    for gi in range(len(g_mjd)):
        for ri in range(len(r_mjd)):
            dt = abs(g_mjd[gi] - r_mjd[ri])
            if dt <= match_window:
                candidates.append((dt, gi, ri))
    candidates.sort(key=lambda c: c[0])
    used_g, used_r, epochs = set(), set(), []
    for dt, gi, ri in candidates:
        if gi in used_g or ri in used_r:
            continue
        used_g.add(gi)
        used_r.add(ri)
        comb_mag, comb_err = weighted_mean([g_mag[gi], r_mag[ri]],
                                           [g_err[gi], r_err[ri]])
        epochs.append({
            "mjd": 0.5 * (g_mjd[gi] + r_mjd[ri]),
            "g_mag": g_mag[gi], "g_err": g_err[gi],
            "r_mag": r_mag[ri], "r_err": r_err[ri],
            "color": g_mag[gi] - r_mag[ri],
            "color_err": np.hypot(g_err[gi], r_err[ri]),
            "comb_mag": comb_mag, "comb_err": comb_err,
        })
    epochs.sort(key=lambda e: e["mjd"])
    return epochs


def measure_color(epochs, n=N_COLOR_EPOCHS, threshold=COLOR_BLUE_THRESHOLD):
    latest = epochs[-n:]
    color, color_err = weighted_mean([e["color"] for e in latest],
                                     [e["color_err"] for e in latest])
    return {"gr_color": color, "gr_color_err": color_err,
            "is_blue": color < threshold, "n_color_epochs": len(latest)}


def measure_rise(epochs):
    """Latest epoch vs the weighted mean of the prior 2 epochs (smaller mag =
    brighter = rising)."""
    latest, prior = epochs[-1:], epochs[-3:-1]
    lmag, _ = weighted_mean([e["comb_mag"] for e in latest],
                            [e["comb_err"] for e in latest])
    pmag, _ = weighted_mean([e["comb_mag"] for e in prior],
                            [e["comb_err"] for e in prior])
    return {"latest_mag": lmag, "prior_mag": pmag,
            "delta_mag": lmag - pmag, "is_rising": lmag < pmag}


def screen_photometry(oid, client, use_forced=True, debug_forced=False):
    """Stage 2/3 per object. Returns a result dict, or None if too few epochs."""
    bands, ra0, dec0 = get_difference_photometry(oid, client, use_forced=use_forced,
                                                 debug_forced=debug_forced)
    epochs = build_gr_epochs(bands)
    if len(epochs) < MIN_EPOCHS:
        return None
    out = {"oid": oid, "ztf_name": oid, "ra": ra0, "dec": dec0,
           "n_matched_epochs": len(epochs), "_bands": bands, "_epochs": epochs}
    out.update(measure_color(epochs))
    out.update(measure_rise(epochs))
    return out


# --------------------------------------------------------------------------- #
#  Stage 1b: Gaia stellar rejection
# --------------------------------------------------------------------------- #
def is_gaia_star(ra, dec, radius_arcsec=GAIA_MATCH_RADIUS_ARCSEC,
                 pm_sig_thresh=GAIA_PM_SIG_THRESH, plx_sig_thresh=GAIA_PLX_SIG_THRESH,
                 proximity_arcsec=GAIA_PROXIMITY_ARCSEC, proximity_maxmag=GAIA_PROXIMITY_MAXMAG,
                 quiet=True):
    """Per-object Gaia cone search (used as the X-Match fallback and for --target).
    Returns (is_star, info): significant PM/parallax OR a bright Gaia source within
    proximity_arcsec. Fails open."""
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return False, {}
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    try:
        with _silence(quiet):
            r = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec).get_results()
    except Exception as exc:
        if not quiet:
            print(f"    [Gaia query failed at {ra:.5f},{dec:.5f}: {exc}]")
        return False, {"gaia_query_failed": True}
    if r is None or len(r) == 0:
        return False, {"gaia_match": False}
    if "dist" in r.colnames:
        r.sort("dist")
    row, cols = r[0], r.colnames

    def val(c):
        if c not in cols:
            return np.nan
        v = row[c]
        return np.nan if (v is None or np.ma.is_masked(v)) else float(v)

    pm_sig = _pm_significance(val("pmra"), val("pmdec"), val("pmra_error"), val("pmdec_error"))
    plx, plx_e = val("parallax"), val("parallax_error")
    plx_sig = abs(plx / plx_e) if (np.isfinite(plx) and np.isfinite(plx_e) and plx_e > 0) else np.nan
    gmag = val("phot_g_mean_mag")
    sep_arcsec = (float(row["dist"]) * 3600.0) if "dist" in cols else np.inf
    near = (sep_arcsec <= proximity_arcsec and (not np.isfinite(gmag) or gmag <= proximity_maxmag))
    is_star = ((np.isfinite(pm_sig) and pm_sig > pm_sig_thresh) or
               (np.isfinite(plx_sig) and plx_sig > plx_sig_thresh) or near)
    return bool(is_star), {"gaia_match": True,
                           "gaia_sep_arcsec": sep_arcsec if np.isfinite(sep_arcsec) else None,
                           "gaia_gmag": float(gmag) if np.isfinite(gmag) else None,
                           "pm_sig": float(pm_sig) if np.isfinite(pm_sig) else None,
                           "plx_sig": float(plx_sig) if np.isfinite(plx_sig) else None}


def _pm_significance(pmra, pmdec, pmra_e, pmdec_e):
    if np.all(np.isfinite([pmra, pmdec, pmra_e, pmdec_e])) and pmra_e > 0 and pmdec_e > 0:
        denom = np.sqrt((pmra * pmra_e) ** 2 + (pmdec * pmdec_e) ** 2)
        if denom > 0:
            return (pmra ** 2 + pmdec ** 2) / denom
    return np.nan


def gaia_star_mask(positions, radius_arcsec=GAIA_MATCH_RADIUS_ARCSEC,
                   pm_sig_thresh=GAIA_PM_SIG_THRESH, plx_sig_thresh=GAIA_PLX_SIG_THRESH,
                   proximity_arcsec=GAIA_PROXIMITY_ARCSEC, proximity_maxmag=GAIA_PROXIMITY_MAXMAG,
                   chunk=2000, quiet=True):
    """Bulk Gaia DR3 stellar check via CDS X-Match (one request per chunk).
    positions: iterable of (oid, ra_deg, dec_deg). Returns dict oid -> (is_star,
    info). Rejects sources with significant PM/parallax OR a bright Gaia source
    (Gmag <= proximity_maxmag) within proximity_arcsec. Chunk failure -> per-object."""
    from astropy.table import Table
    from astroquery.xmatch import XMatch
    rows = [(str(o), float(ra), float(dec)) for (o, ra, dec) in positions
            if np.isfinite(ra) and np.isfinite(dec)]
    out = {}
    for i in range(0, len(rows), chunk):
        sub = rows[i:i + chunk]
        tbl = Table(rows=sub, names=["oid", "ra", "dec"])
        try:
            with _silence(quiet):
                res = XMatch.query(cat1=tbl, cat2="vizier:I/355/gaiadr3",
                                   max_distance=radius_arcsec * u.arcsec,
                                   colRA1="ra", colDec1="dec")
        except Exception as exc:
            print(f"    [bulk Gaia X-Match failed (chunk {i // chunk}): {exc}; "
                  f"falling back to per-object]")
            for (o, ra, dec) in sub:
                out[o] = is_gaia_star(ra, dec, radius_arcsec, pm_sig_thresh, plx_sig_thresh, quiet=True)
            continue
        if res is None or len(res) == 0:
            continue
        need = ("pmRA", "pmDE", "e_pmRA", "e_pmDE", "Plx", "e_Plx")
        missing = [c for c in need if c not in res.colnames]
        if missing:
            print(f"    [X-Match columns {list(res.colnames)} missing {missing}; "
                  f"cannot assess stars this chunk]")
            continue
        nearest = {}
        for r in res:
            o, sep = str(r["oid"]), float(r["angDist"])
            if o not in nearest or sep < nearest[o][0]:
                nearest[o] = (sep, r)
        for o, (sep, r) in nearest.items():
            def fv(col):
                try:
                    v = r[col]
                    return float(v) if not np.ma.is_masked(v) else np.nan
                except Exception:
                    return np.nan
            pm_sig = _pm_significance(fv("pmRA"), fv("pmDE"), fv("e_pmRA"), fv("e_pmDE"))
            plx, plx_e = fv("Plx"), fv("e_Plx")
            plx_sig = abs(plx / plx_e) if (np.isfinite(plx) and np.isfinite(plx_e) and plx_e > 0) else np.nan
            gmag = fv("Gmag")
            near = (sep <= proximity_arcsec and (not np.isfinite(gmag) or gmag <= proximity_maxmag))
            is_star = ((np.isfinite(pm_sig) and pm_sig > pm_sig_thresh) or
                       (np.isfinite(plx_sig) and plx_sig > plx_sig_thresh) or near)
            out[o] = (bool(is_star),
                      {"gaia_sep_arcsec": sep,
                       "gaia_gmag": float(gmag) if np.isfinite(gmag) else None,
                       "pm_sig": float(pm_sig) if np.isfinite(pm_sig) else None,
                       "plx_sig": float(plx_sig) if np.isfinite(plx_sig) else None})
    return out


# --------------------------------------------------------------------------- #
#  Stage 4: nuclear test
# --------------------------------------------------------------------------- #
def _clear_iinuclear_cache(name, base_dir="."):
    import glob
    for pat in (os.path.join(base_dir, "coords", f"{name}_coords.*"),
                os.path.join(base_dir, "catalogs", f"{name}.*"),
                os.path.join(base_dir, "images", f"{name}_*.*")):
        for f in glob.glob(pat):
            try:
                os.remove(f)
            except OSError:
                pass


def is_object_nuclear(identifier, quiet=True, _retry=True):
    """Return (is_nuclear, p_value, sigma); all None if no usable host catalog or
    on error. plot=False skips iinuclear's (unused) image download. A 'used_catalog'
    crash usually means a stale/empty cached catalog -> clear it and retry once."""
    try:
        with _silence(quiet):
            try:
                out = get_data(identifier, plot=False)
            except TypeError:
                out = get_data(identifier)
            ras, decs = out[0], out[1]
            catalog_result = out[CATALOG_INDEX]
            if catalog_result is None or len(catalog_result) == 0:
                return None, None, None
            gc = _astuple(get_galaxy_center(catalog_result))
            ra_gal, dec_gal = gc[GALCENTER_RA_IDX], gc[GALCENTER_DEC_IDX]
            err_arcsec = gc[GALCENTER_ERR_IDX]
            nuc = _astuple(check_nuclear(ras, decs, ra_gal, dec_gal, err_arcsec))
            return (bool(nuc[NUCLEAR_FLAG_IDX]),
                    float(nuc[NUCLEAR_PVAL_IDX]),
                    float(nuc[NUCLEAR_SIGMA_IDX]))
    except Exception as exc:
        msg = str(exc).lower()
        if _retry and "used_catalog" in msg:
            _clear_iinuclear_cache(identifier)
            return is_object_nuclear(identifier, quiet=quiet, _retry=False)
        no_host = ("isfinite" in msg or "weights sum to zero" in msg
                   or "zero-size array" in msg or "used_catalog" in msg)
        if not (quiet and no_host):
            print(f"    [nuclear check failed for {identifier}: {exc}]")
        return None, None, None

# --------------------------------------------------------------------------- #
#  Stage 5: Gaia DR3 AGN / QSO-candidate annotation
# --------------------------------------------------------------------------- #
# gaiadr3.qso_candidates is deliberately COMPLETE rather than PURE (~6.6M rows),
# so a bare positional match is a weak statement. We therefore also return a
# purity "grade":
#     "likely"    -- Gaia-CRF3 source, or DSC joint label 'quasar', or the
#                    variability classifier's best class is 'AGN'
#     "candidate" -- in the table, but none of the above
# The table carries no positions of its own, so we join to gaiadr3.gaia_source.

_AGN_ADQL = """
SELECT TOP 5
       q.source_id,
       q.classlabel_dsc_joint,
       q.classprob_dsc_combmod_quasar,
       q.vari_best_class_name,
       q.vari_best_class_score,
       q.vari_agn_membership_score,
       q.gaia_crf_source,
       q.astrometric_selection_flag,
       q.redshift_qsoc,
       DISTANCE(POINT('ICRS', g.ra, g.dec),
                POINT('ICRS', {ra:.8f}, {dec:.8f})) AS ang_sep
FROM gaiadr3.qso_candidates AS q
JOIN gaiadr3.gaia_source AS g ON g.source_id = q.source_id
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec),
                   CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius_deg:.10f}))
ORDER BY ang_sep ASC
"""


def _row_val(row, col):
    """Scalar from an astropy Row, or None if absent/masked."""
    try:
        if col not in row.colnames:
            return None
        v = row[col]
    except Exception:
        return None
    if v is None or np.ma.is_masked(v):
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.str_):
        return str(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return str(v)


def gaia_agn_flag(ra, dec, radius_arcsec=GAIA_AGN_MATCH_RADIUS_ARCSEC,
                  min_prob=GAIA_AGN_MIN_PROB, quiet=True):
    """Is there a Gaia DR3 QSO candidate at this position?

    Returns (is_agn, info). is_agn is True/False, or None if the query failed
    (i.e. "undetermined", distinct from a clean non-detection). Fails open --
    this never removes a candidate, it only annotates one.
    """
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return None, {}
    adql = _AGN_ADQL.format(ra=float(ra), dec=float(dec),
                            radius_deg=radius_arcsec / 3600.0)
    try:
        with _silence(quiet):
            res = _with_retries(lambda: Gaia.launch_job(adql).get_results(), _quiet=quiet)
    except Exception as exc:
        if not quiet:
            print(f"    [Gaia AGN query failed at {ra:.5f},{dec:.5f}: {exc}]")
        return None, {"agn_query_failed": True}

    if res is None or len(res) == 0:
        return False, {"gaia_agn_match": False}

    row = res[0]
    dsc_label = _row_val(row, "classlabel_dsc_joint")
    vari_class = _row_val(row, "vari_best_class_name")
    qso_prob = _row_val(row, "classprob_dsc_combmod_quasar")
    crf = bool(_row_val(row, "gaia_crf_source"))
    sep_deg = _row_val(row, "ang_sep")

    strong = (crf
              or (isinstance(dsc_label, str) and dsc_label.strip().lower() == "quasar")
              or (isinstance(vari_class, str) and vari_class.strip().upper() == "AGN"))
    grade = "likely" if strong else "candidate"

    if min_prob > 0 and not strong:
        is_agn = bool(qso_prob is not None and qso_prob >= min_prob)
    else:
        is_agn = True

    info = {
        "gaia_agn_match": True,
        "grade": grade,
        "source_id": _row_val(row, "source_id"),
        "sep_arcsec": (sep_deg * 3600.0) if sep_deg is not None else None,
        "qso_prob": qso_prob,
        "dsc_label": dsc_label,
        "vari_class": vari_class,
        "vari_agn_score": _row_val(row, "vari_agn_membership_score"),
        "gaia_crf_source": crf,
        "redshift_qsoc": _row_val(row, "redshift_qsoc"),
        "n_matches": len(res),
    }
    return is_agn, info

# --------------------------------------------------------------------------- #
#  Optional plotting
# --------------------------------------------------------------------------- #
def plot_lightcurve(result, savepath=None):
    import matplotlib.pyplot as plt
    bands, epochs = result["_bands"], result["_epochs"]
    fig, ax = plt.subplots(figsize=(8, 5))
    gm, gmag, gerr = bands[FID_G]
    rm, rmag, rerr = bands[FID_R]
    ax.errorbar(gm, gmag, yerr=gerr, fmt="o", ms=3, alpha=0.3, color="tab:green", label="g (all)")
    ax.errorbar(rm, rmag, yerr=rerr, fmt="o", ms=3, alpha=0.3, color="tab:red", label="r (all)")
    e_mjd = [e["mjd"] for e in epochs]
    ax.plot(e_mjd, [e["g_mag"] for e in epochs], "s", ms=9, mfc="none", color="tab:green", label="matched g")
    ax.plot(e_mjd, [e["r_mag"] for e in epochs], "s", ms=9, mfc="none", color="tab:red", label="matched r")
    prior, latest = epochs[-3:-1], epochs[-1:]
    ax.axvspan(prior[0]["mjd"], prior[-1]["mjd"], color="gray", alpha=0.12, label="prior 2 (rise ref.)")
    ax.axvspan(latest[0]["mjd"] - 0.5, latest[0]["mjd"] + 0.5, color="gold", alpha=0.25, label="latest (rise test)")
    ax.invert_yaxis()
    ax.set_xlabel("MJD"); ax.set_ylabel("difference magnitude")
    blue = "blue" if result["is_blue"] else "not blue"
    trend = "rising" if result["is_rising"] else "falling"
    sig = result.get("nuclear_sigma")
    sig_s = f"{sig:.2f}sigma" if sig is not None else "sigma n/a"
    ax.set_title(f"{result['ztf_name']}   |   g-r = {result['gr_color']:+.2f} ({blue})"
                 f"   |   {trend}   |   {sig_s} from host")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150); plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------- #
#  Full scan
def run_scan(days=MAX_AGE_DAYS, min_ndet=MIN_DETECTIONS, max_objects=None,
             require_rising=False, reject_stars=True, reject_cvnova=True,
             cvnova_classifier=CVNOVA_CLASSIFIER, cvnova_prob=CVNOVA_PROB,
             use_forced=True, sigma_max=NUCLEAR_SIGMA_MAX, workers=1, rate=5.0,
             flag_agn=True, agn_radius=GAIA_AGN_MATCH_RADIUS_ARCSEC,
             agn_min_prob=GAIA_AGN_MIN_PROB,
             plotdir=None, client=None):
    if client is None:
        client = Alerce()
    global _RATE_LIMITER
    _RATE_LIMITER = _RateLimiter(rate)

    recent = query_recent_discoveries(client, days, min_ndet, max_objects)
    print(f"Stage 1  recency : {len(recent)} transients found in last {days:.0f} d "
          f"(ndet>={min_ndet})")
    if workers > 1:
        print(f"         (using {workers} worker threads for photometry/nuclear)")

    # Stage 1a: drop likely CVs/Novae (one bulk classifier query).
    if reject_cvnova:
        cvnova = query_cvnova_oids(client, days, min_ndet,
                                   classifier=cvnova_classifier, probability=cvnova_prob)
        before = len(recent)
        recent = recent[~recent["oid"].isin(cvnova)].reset_index(drop=True)
        print(f"Stage 1a CV/Nova : removed {before - len(recent)} CV/Nova "
              f"(p>={cvnova_prob}); {len(recent)} remain")

    # Stage 1b: reject stars before any photometry (bulk Gaia X-Match).
    positions = list(zip(recent["oid"], recent["meanra"], recent["meandec"]))
    if reject_stars:
        star_map = gaia_star_mask(positions)
        oids = [o for (o, _ra, _dec) in positions if not star_map.get(o, (False, {}))[0]]
        print(f"Stage 1b stars   : removed {len(positions) - len(oids)} Gaia stars "
              f"before photometry; {len(oids)} remain")
        default_star = False
    else:
        star_map = {}
        oids = [o for (o, _ra, _dec) in positions]
        default_star = None

    # Stage 2: photometry (alert + forced) + epochs, only on the survivors.
    photo = _parallel_map(
        lambda oid: screen_photometry(oid, _get_client(), use_forced=use_forced), oids, workers,
        on_error=lambda oid, exc: print(f"    [photometry failed for {oid}: {exc}]"))
    photo_pass = [s for _oid, s in photo if s is not None]
    for s in photo_pass:
        star, ginfo = star_map.get(s["oid"], (default_star, {}))
        s["is_star"] = star
        s["gaia_pm_sig"] = ginfo.get("pm_sig")
        s["gaia_plx_sig"] = ginfo.get("plx_sig")
    print(f"Stage 2  epochs  : {len(photo_pass)} have >= {MIN_EPOCHS} matched g/r epochs")

    blue = [s for s in photo_pass if s["is_blue"]]
    print(f"Stage 3  blue    : {len(blue)} are currently blue (g-r < {COLOR_BLUE_THRESHOLD})")

    # Stage 4: nuclear test.
    nuc = _parallel_map(lambda s: is_object_nuclear(s["oid"]), blue, workers)
    for s, (is_nuc, p_val, sigma) in nuc:
        s["is_nuclear"], s["nuclear_pvalue"], s["nuclear_sigma"] = is_nuc, p_val, sigma
    finalists = [s for s in blue
                 if s.get("nuclear_sigma") is not None and s["nuclear_sigma"] <= sigma_max]
    n_far = sum(1 for s in blue
                if s.get("nuclear_sigma") is not None and s["nuclear_sigma"] > sigma_max)
    undetermined = [s["oid"] for s in blue if s.get("nuclear_sigma") is None]
    print(f"Stage 4  nuclear : {len(finalists)} within {sigma_max:g}sigma of host  "
          f"({n_far} farther, {len(undetermined)} no host / undetermined)")
    if undetermined:
        print(f"         host-undetermined (no PS1/SDSS catalog match; worth a "
              f"manual look): {', '.join(undetermined)}")

    if require_rising:
        finalists = [s for s in finalists if s["is_rising"]]
        print(f"         rising  : {len(finalists)} of those are also rising")
    
    # Stage 5: Gaia AGN annotation -- informational only
    for s in finalists:
        s.setdefault("gaia_agn", None)
    if flag_agn and finalists:
        agn_res = _parallel_map(
            lambda s: gaia_agn_flag(s["ra"], s["dec"], agn_radius, agn_min_prob),
            finalists, workers,
            on_error=lambda s, exc: print(f"    [AGN check failed for {s['oid']}: {exc}]"))
        for s, (is_agn, info) in agn_res:
            s["gaia_agn"] = is_agn
            s["gaia_agn_grade"] = info.get("grade")
            s["gaia_agn_prob"] = info.get("qso_prob")
            s["gaia_agn_sep_arcsec"] = info.get("sep_arcsec")
            s["gaia_agn_z"] = info.get("redshift_qsoc")
        n_agn = sum(1 for s in finalists if s.get("gaia_agn") is True)
        n_likely = sum(1 for s in finalists if s.get("gaia_agn_grade") == "likely")
        n_unk = sum(1 for s in finalists if s.get("gaia_agn") is None)
        print(f"Stage 5  AGN flag: {n_agn}/{len(finalists)} match a Gaia DR3 QSO "
              f"candidate within {agn_radius:g}\" ({n_likely} graded 'likely', "
              f"{n_unk} undetermined)")

    if plotdir and finalists:
        os.makedirs(plotdir, exist_ok=True)
        for s in finalists:
            plot_lightcurve(s, savepath=os.path.join(plotdir, f"{s['oid']}.png"))
        print(f"         plots   : wrote {len(finalists)} figure(s) to {plotdir}/")

    return finalists


def results_table(finalists):
    cols = ["oid", "ra", "dec", "n_matched_epochs", "gr_color", "is_blue",
            "is_rising", "delta_mag", "is_star", "gaia_pm_sig", "gaia_plx_sig",
            "nuclear_sigma", "nuclear_pvalue", "is_nuclear",
            "gaia_agn", "gaia_agn_grade", "gaia_agn_prob",
            "gaia_agn_sep_arcsec", "gaia_agn_z"]
    rows = [{c: s.get(c) for c in cols} for s in finalists]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
#  Single-object path (debug)
# --------------------------------------------------------------------------- #
def classify_one(name, client=None, want_plot=False, reject_stars=True, use_forced=True):
    if client is None:
        client = Alerce()
    if isinstance(name, str) and name.upper().startswith("ZTF"):
        now = Time.now().mjd
        try:
            obj = client.query_objects(oid=[name], survey="ztf", format="pandas")
        except TypeError:
            obj = client.query_objects(oid=[name], format="pandas")
        age = now - float(np.asarray(obj["firstmjd"], dtype=float)[0])
        if age > MAX_AGE_DAYS:
            print(f"{name}: discovered {age:.0f} d ago -> older than {MAX_AGE_DAYS:.0f} d; skipped.")
            return None

    s = screen_photometry(name, client, use_forced=use_forced, debug_forced=True)
    if s is None:
        print(f"{name}: fewer than {MIN_EPOCHS} matched g/r epochs; cannot screen.")
        return None
    if reject_stars:
        s["is_star"], ginfo = is_gaia_star(s["ra"], s["dec"])
        s["gaia_pm_sig"] = ginfo.get("pm_sig"); s["gaia_plx_sig"] = ginfo.get("plx_sig")
    else:
        s["is_star"] = None
    s["is_nuclear"], s["nuclear_pvalue"], s["nuclear_sigma"] = is_object_nuclear(name)
    s["gaia_agn"], _agn = gaia_agn_flag(s["ra"], s["dec"], quiet=False)
    s["gaia_agn_grade"] = _agn.get("grade")
    s["gaia_agn_prob"] = _agn.get("qso_prob")
    s["gaia_agn_sep_arcsec"] = _agn.get("sep_arcsec")
    s["gaia_agn_z"] = _agn.get("redshift_qsoc")

    blue = "BLUE" if s["is_blue"] else "not blue"
    trend = "RISING" if s["is_rising"] else "FALLING"
    star = {True: "STAR", False: "not a star", None: "star n/a"}[s["is_star"]]
    sig = s["nuclear_sigma"]
    agn = {True: f"GaiaAGN({s['gaia_agn_grade']})", False: "no Gaia AGN",
           None: "AGN n/a"}[s["gaia_agn"]]
    print(f"{name}: g-r={s['gr_color']:+.3f} ({blue}), {trend}, {star}, "
          f"sigma={f'{sig:.2f}' if sig is not None else 'n/a'}, "
          f"is_nuclear={s['is_nuclear']}, {agn}, {s['n_matched_epochs']} epochs")
    if want_plot:
        plot_lightcurve(s)
    return s


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Scan ALeRCE for recent blue, non-stellar, non-CV, nuclear ZTF transients.")
    p.add_argument("--days", type=float, default=MAX_AGE_DAYS, help="discovery window in days (default 90)")
    p.add_argument("--min-ndet", type=int, default=MIN_DETECTIONS, help="minimum total detections pre-filter (default 5)")
    p.add_argument("--max-objects", type=int, default=None, help="cap recent objects screened (testing)")
    p.add_argument("--workers", type=int, default=1, help="parallel threads for per-object stages (default 1; 4-8 with --rate)")
    p.add_argument("--rate", type=float, default=5.0, help="max ALeRCE requests/sec across threads (default 5; 0 disables)")
    p.add_argument("--sigma-max", type=float, default=NUCLEAR_SIGMA_MAX, help="keep transients within this many sigma of host (default 1.0)")
    p.add_argument("--no-forced", action="store_true", help="use alert photometry only (skip forced photometry)")
    p.add_argument("--keep-cvnova", action="store_true", help="skip the CV/Nova classifier cut")
    p.add_argument("--cvnova-prob", type=float, default=CVNOVA_PROB, help="CV/Nova probability above which to reject (default 0.3)")
    p.add_argument("--cvnova-classifier", default=CVNOVA_CLASSIFIER, help="forced-phot classifier name for the CV/Nova cut")
    p.add_argument("--keep-stars", action="store_true", help="skip the Gaia stellar-rejection stage")
    p.add_argument("--rising", action="store_true", help="also require final candidates to be rising")
    p.add_argument("--plot", action="store_true", help="save a lightcurve plot for each final candidate")
    p.add_argument("--plotdir", default="plots", help="directory for plots when --plot is set")
    p.add_argument("--out", default=None, metavar="CSV", help="write the candidate table to this CSV path")
    p.add_argument("--target", default=None, help="screen a single object (ZTF/IAU name or coords) and exit")
    p.add_argument("--no-agn-flag", action="store_true",
                   help="skip the Gaia DR3 QSO-candidate annotation of final candidates")
    p.add_argument("--agn-radius", type=float, default=GAIA_AGN_MATCH_RADIUS_ARCSEC,
                   help="match radius in arcsec for the Gaia AGN flag (default 2.0)")
    p.add_argument("--agn-min-prob", type=float, default=GAIA_AGN_MIN_PROB,
                   help="require classprob_dsc_combmod_quasar >= this for 'candidate'-grade "
                        "matches to count as AGN (default 0 = any match counts)")
    
    args = p.parse_args()

    if args.target:
        classify_one(args.target, want_plot=args.plot, reject_stars=not args.keep_stars,
                     use_forced=not args.no_forced)
    else:
        finalists = run_scan(days=args.days, min_ndet=args.min_ndet, max_objects=args.max_objects,
                             require_rising=args.rising, reject_stars=not args.keep_stars,
                             reject_cvnova=not args.keep_cvnova, cvnova_classifier=args.cvnova_classifier,
                             cvnova_prob=args.cvnova_prob, use_forced=not args.no_forced,
                             sigma_max=args.sigma_max, workers=args.workers, rate=args.rate,
                             flag_agn=not args.no_agn_flag, agn_radius=args.agn_radius,
                             agn_min_prob=args.agn_min_prob,
                             plotdir=args.plotdir if args.plot else None)
        table = results_table(finalists)
        print("\n=== Final candidates (blue + non-stellar + non-CV + nuclear) ===")
        print(table.to_string(index=False) if len(table) else "(none)")
        if args.out:
            table.to_csv(args.out, index=False)
            print(f"\nwrote {len(table)} row(s) -> {args.out}")
