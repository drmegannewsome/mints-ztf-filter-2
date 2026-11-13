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
  Stage 5   Gaia AGN : informational Gaia DR3 QSO-candidate flag (never a cut).
  Stage 6   Spectra  : informational archival-spectrum check of the final candidates
                       against the DESI DR1 AGN/Galaxy VAC, via
                       spectra_finding/desi_agngal_crossmatch.py (never a cut).

Memory: verdicts are cached per object in state/scan_state.json (see ScanState), so a
daily run only screens objects it has not decided yet and merely refreshes the photometry
of the candidates it is already tracking. Objects that are not blue, or not nuclear, are
never re-screened.

A g/r "epoch" = a g and r measurement within 2 days of each other. Rise/fall
compares the latest epoch against the weighted mean of the prior 2.

Inspired by FLEET (Gomez et al.): https://github.com/gmzsebastian/fleet
Nuclear test via iinuclear:        https://github.com/gmzsebastian/iinuclear

Requirements:  pip install alerce iinuclear numpy pandas astropy astroquery matplotlib
               pip install fitsio     # only for the Stage 6 archival-spectra check

CLI examples:
    python early_color_classifier.py --days 60 --min-ndet 5 --workers 1
    python early_color_classifier.py --plot --plotdir figs --out candidates.csv
    python early_color_classifier.py --no-forced --keep-cvnova   # ablations
"""

import os
import sys
import json
import time
import threading
import collections
import contextlib
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import pandas as pd
import requests
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

# --------------------------------------------------------------------------- #
#  Network timeouts. The alerce client calls session.request(...) with NO
#  timeout, so one stalled response hangs its worker thread forever -- and the
#  thread pool then waits for it forever. Give every requests call that doesn't
#  set its own timeout a (connect, read) limit; a timeout is treated as a
#  transient error by _with_retries below.
# --------------------------------------------------------------------------- #
REQUEST_TIMEOUT = (10, 90)          # seconds: (connect, read)
MAST_TIMEOUT = 120                  # astroquery's MAST default is 600 s per request

_orig_session_request = requests.Session.request

def _request_with_timeout(self, method, url, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = REQUEST_TIMEOUT
    return _orig_session_request(self, method, url, **kwargs)

requests.Session.request = _request_with_timeout

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

try:
    from astroquery.mast import conf as _mast_conf
    _mast_conf.timeout = MAST_TIMEOUT
except Exception:
    pass

from astroquery.gaia import Gaia
from iinuclear.utils import (get_data, get_galaxy_center, check_nuclear, query_panstarrs,
                             query_sdss, calc_separations)
try:                                  # Legacy Survey Tractor host positions (see ls_reference.py)
    import ls_reference as _LS
except Exception:                     # missing/broken module: the scan still runs on PS1/SDSS
    _LS = None

# ZTF filter ids in ALeRCE
FID_G, FID_R = 1, 2

# Tunables
COLOR_BLUE_THRESHOLD = -0.05   # g - r < this  ==>  "blue"
GR_MATCH_WINDOW_DAYS = 1.5     # max |t_g - t_r| to call a g AND r night one epoch (i band is never used)
NIGHT_GAP_DAYS = 0.5           # gap that separates one night from the next (per band)
MIN_EPOCHS = 3                 # need 3 matched g+r epochs (colour uses the latest 3; rise = latest vs prior 2)
N_COLOR_EPOCHS = 3             # color uses the 3 latest matched epochs
MAG_ERR_FLOOR = 0.01           # guard against zero/NaN photometric errors
MAX_AGE_DAYS = 90.0            # "found in the last 3 months"
MIN_DETECTIONS = 5             # coarse Stage-1 ndet pre-filter
NUCLEAR_P_MIN = 0.05           # iinuclear's own rule: nuclear if p > 0.05 ...
NUCLEAR_SIGMA_MAX = float(np.sqrt(-2.0 * np.log(NUCLEAR_P_MIN)))   # ... = sigma <= 2.4477 (2 dof: p = exp(-sigma^2/2))
FORCED_MAX_MAGERR = 0.30       # keep forced-phot points at least ~3-sigma (SNR>~3)
HOST_UNCLEAR_SEP_ARCSEC = 5.0   # a host whose centre is MORE than this from the ZTF position ...
HOST_UNCLEAR_ERR_ARCSEC = 3.0   # ... AND whose error is at least this is no usable reference: "Host Unclear"
REQUIRE_POSITIVE_DIFF = True   # only points where the source is BRIGHTER than the reference (isdiffpos = +1).
                               # A negative difference is a subtraction residual / variable star, not a transient rising

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
XMATCH_EMPTY_SUSPECT = 100     # a bulk X-Match request this big that returns NO row is a glitch, not a result
# Gaia DR3 QSO-candidate annotation (Stage 5; informational, never a cut)
GAIA_AGN_MATCH_RADIUS_ARCSEC = 2.0
GAIA_AGN_MIN_PROB = 0.25        # 0 = flag any qso_candidates match; raise to purify

# Archival host spectra (Stage 6; informational, never a cut)
_HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(_HERE, "spectra_finding")
SPEC_VAC_DEFAULT = os.path.join(SPEC_DIR, "data", "agngal_desi.fits")   # git-ignored
SPEC_MATCH_RADIUS_ARCSEC = 2.0   # vs DESI TARGET_RA/DEC; ZTF positions are good to ~<1"

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


class _ThreadStream:
    """Stand-in for sys.stdout/sys.stderr. By default it forwards to the real stream; a
    thread can opt in (via _capture_thread) to have what IT prints collected instead --
    safe with many worker threads, unlike swapping sys.stdout around a call."""
    def __init__(self, base):
        self.base, self.local = base, threading.local()

    def write(self, text):
        buf = getattr(self.local, "buf", None)
        if buf is not None:
            buf.append(text)
            return len(text)
        return self.base.write(text)

    def flush(self):
        self.base.flush()

    def __getattr__(self, name):
        return getattr(self.base, name)


_capture_lock = threading.Lock()


@contextlib.contextmanager
def _capture_thread():
    """Collect everything the current thread prints (stdout and stderr) into a list."""
    with _capture_lock:
        for name in ("stdout", "stderr"):
            if not isinstance(getattr(sys, name), _ThreadStream):
                setattr(sys, name, _ThreadStream(getattr(sys, name)))
    buf = []
    streams = (sys.stdout, sys.stderr)
    for st in streams:
        st.local.buf = buf
    try:
        yield buf
    finally:
        for st in streams:
            st.local.buf = None


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
    """Global throttle: spaces request starts >= min_interval apart across all threads.
    max_per_sec<=0 disables it. It also ADAPTS: when a server answers 403/429, penalize()
    makes every thread pause together and slows the pace (x1.5, at most once per 5 s, down
    to 0.5 requests/s); after 60 clean requests it speeds back up (x0.85) toward the
    configured rate. So it settles just under the server's real limit instead of every
    thread sleeping alone and retrying into the same wall."""
    SLOWEST_INTERVAL = 2.0

    def __init__(self, max_per_sec):
        self.base_interval = (1.0 / max_per_sec) if (max_per_sec and max_per_sec > 0) else 0.0
        self.min_interval = self.base_interval
        self._lock = threading.Lock()
        self._next = 0.0
        self._ok = 0
        self._last_penalty = -1e9
        self.n_penalties = 0

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

    def penalize(self, pause_s):
        if self.base_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            self._next = max(self._next, now + pause_s)       # everyone waits, not just the caller
            self._ok = 0
            if now - self._last_penalty > 5.0:
                self._last_penalty = now
                self.n_penalties += 1
                self.min_interval = min(self.min_interval * 1.5, self.SLOWEST_INTERVAL)

    def reward(self):
        if self.base_interval <= 0:
            return
        with self._lock:
            self._ok += 1
            if self._ok >= 60 and self.min_interval > self.base_interval:
                self._ok = 0
                self.min_interval = max(self.base_interval, self.min_interval * 0.85)


_RATE_LIMITER = _RateLimiter(0)


def _with_retries(func, *args, _tries=5, _base_delay=2.0, _quiet=False, **kwargs):
    """Throttled call that retries transient (5xx/timeout) and rate-limit (403/429)
    errors with exponential backoff; other errors raise immediately."""
    for attempt in range(1, _tries + 1):
        _RATE_LIMITER.acquire()
        try:
            result = func(*args, **kwargs)
            _RATE_LIMITER.reward()
            return result
        except Exception as exc:
            rate_limited = _is_rate_limit_error(exc)
            if attempt >= _tries or not (rate_limited or _is_transient_error(exc)):
                raise
            delay = (5.0 if rate_limited else _base_delay) * (2 ** (attempt - 1))
            if not _quiet:
                kind = "rate-limited (403/429)" if rate_limited else "transient API error"
                print(f"    [{kind}; attempt {attempt}/{_tries}, backing off {delay:.0f}s]")
            if rate_limited and _RATE_LIMITER.base_interval > 0:
                _RATE_LIMITER.penalize(delay)       # all threads pause; the next acquire() waits it out
            else:
                time.sleep(delay)


_thread_local = threading.local()

# Gaia archive (since the Oct 2026 release): at most 5 parallel jobs per user and a queue of 50; jobs
# beyond that are rejected. With --workers 8 the per-object star / AGN checks would be rejected and
# (failing open) let stars through, so they share a few slots and retry.
_GAIA_SLOTS = threading.BoundedSemaphore(3)
_GAIA_RETRY_BASE = 3.0


def _gaia_call(fn, tries=2):
    for k in range(tries):
        with _GAIA_SLOTS:
            try:
                return fn()
            except Exception as exc:
                if k == tries - 1:
                    raise
                wait = _GAIA_RETRY_BASE * (2 ** k) * (5 if _is_rate_limit_error(exc) else 1)
        time.sleep(wait)


def _get_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = Alerce()
    return _thread_local.client


_T0 = time.monotonic()
_PARTIAL = []          # one note per stage that was cut short by its time budget


def _stage(msg):
    """print() with minutes elapsed since the scan started, flushed immediately,
    so a stalled run's log shows exactly where the time went."""
    print(f"[{(time.monotonic() - _T0) / 60:6.1f} min] {msg}", flush=True)


def _parallel_map(func, items, workers, on_error=None, label="objects",
                  budget_min=None, heartbeat=60.0):
    """Apply func(item) across items with `workers` threads (I/O-bound); returns
    (item, result) pairs for calls that didn't raise. workers<=1 is sequential.

    Prints a progress line every `heartbeat` s. If `budget_min` is set and runs
    out, the unfinished items are abandoned (their threads are left to die with
    the process), the stage is recorded in _PARTIAL, and what finished is returned."""
    items = list(items)
    results, done = [], 0
    deadline = time.monotonic() + budget_min * 60 if budget_min else None
    last_beat = time.monotonic()

    def _handle(it, exc=None, value=None):
        if exc is not None:
            if on_error:
                on_error(it, exc)
        else:
            results.append((it, value))

    def _cut_short():
        _PARTIAL.append(f"{label}: {len(items) - done} of {len(items)} unfinished")
        _stage(f"   !! {label}: time budget ({budget_min:g} min) reached with "
               f"{done}/{len(items)} done; continuing with what finished")

    if workers and workers > 1:
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = {pool.submit(func, it): it for it in items}
        pending = set(futures)
        try:
            while pending:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    _cut_short()
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED,
                                         timeout=heartbeat if left is None else min(heartbeat, left))
                for fut in finished:
                    done += 1
                    it = futures[fut]
                    try:
                        _handle(it, value=fut.result())
                    except Exception as exc:
                        _handle(it, exc=exc)
                if time.monotonic() - last_beat >= heartbeat:
                    last_beat = time.monotonic()
                    _stage(f"   {label}: {done}/{len(items)} done")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    else:
        for it in items:
            if deadline is not None and time.monotonic() > deadline:
                _cut_short()
                break
            try:
                _handle(it, value=func(it))
            except Exception as exc:
                _handle(it, exc=exc)
            done += 1
            if time.monotonic() - last_beat >= heartbeat:
                last_beat = time.monotonic()
                _stage(f"   {label}: {done}/{len(items)} done")
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

    frames, page, t0 = [], 1, time.monotonic()
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

    _stage(f"   alerce object query{' (' + class_name + ')' if class_name else ''}: "
           f"{page} page(s), {time.monotonic() - t0:.0f} s")
    if not frames:
        return pd.DataFrame(columns=["oid", "meanra", "meandec"])
    out = pd.concat(frames, ignore_index=True)
    if max_objects:
        out = out.iloc[:max_objects]
    ra_col  = next((c for c in ("meanra", "ra") if c in out.columns), None)
    dec_col = next((c for c in ("meandec", "dec") if c in out.columns), None)
    res = pd.DataFrame({
        "oid": out["oid"].astype(str).values,
        "meanra":  out[ra_col].values  if ra_col  else np.nan,
        "meandec": out[dec_col].values if dec_col else np.nan,
    })
    for c in ("firstmjd", "ndet"):
        res[c] = out[c].values if c in out.columns else np.nan
    return res


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
def _positive_mask(sign, n):
    """True where a point is a POSITIVE difference (source brighter than the reference). ZTF/ALeRCE code it
    as +1/-1 (or 't'/'f'); a missing value is kept, and so is everything if there is no such column."""
    if sign is None:
        return np.ones(n, dtype=bool)
    ser = pd.Series(np.asarray(sign))
    num = pd.to_numeric(ser, errors="coerce")
    if num.notna().any():
        return ((num > 0) | num.isna()).to_numpy()
    txt = ser.astype(str).str.strip().str.lower()
    return (~txt.isin(["f", "false", "-1", "n", "no", "neg"])).to_numpy()


def get_difference_photometry(oid, client, use_forced=True, debug_forced=False, return_positions=False):
    """Return (bands, ra, dec). bands = {1:(mjd,mag,err) g, 2:(...) r}, combining
    alert difference photometry (query_detections: magpsf/sigmapsf) and -- if
    use_forced -- significant ZTF forced difference photometry. ra/dec = median
    detection position. With return_positions=True a 4th value is returned: (ras, decs), the
    finite per-detection alert positions -- the very arrays iinuclear would otherwise re-fetch
    from ALeRCE for the nuclear test. Set debug_forced=True to print the forced-photometry
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
    counts = {"alert_g": 0, "alert_r": 0, "alert_i": 0, "forced_gr": 0, "negative_gr": 0}

    def _add(mjd, fid, mag, err, max_err=None, sign=None):
        mjd = np.asarray(mjd, dtype=float)
        fid = np.asarray(fid, dtype=float)
        mag = np.asarray(mag, dtype=float)
        err = np.asarray(err, dtype=float)
        pos = _positive_mask(sign, len(mjd)) if REQUIRE_POSITIVE_DIFF else np.ones(len(mjd), bool)
        n_kept, n_finite = 0, 0
        for f in (FID_G, FID_R):
            base = ((fid == f) & np.isfinite(mjd) & np.isfinite(mag)
                    & np.isfinite(err) & (err > 0))
            counts["negative_gr"] += int((base & ~pos).sum())      # dropped: source fainter than the reference
            base &= pos
            n_finite += int(base.sum())
            sel = (base & (err <= max_err)) if max_err is not None else base
            per_band[f][0].extend(mjd[sel])
            per_band[f][1].extend(mag[sel])
            per_band[f][2].extend(err[sel])
            n_kept += int(sel.sum())
        return n_kept, n_finite

    _add(det["mjd"], det["fid"], det["magpsf"], det["sigmapsf"],
         sign=det["isdiffpos"] if "isdiffpos" in det.columns else None)
    fid_det = np.asarray(det["fid"], dtype=float)
    counts.update(alert_g=int((fid_det == FID_G).sum()), alert_r=int((fid_det == FID_R).sum()),
                  alert_i=int((fid_det == 3).sum()))                     # i band is only counted, never used

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
                n_kept, n_finite = _add(fp["mjd"], fp[fidcol], fp[magcol], fp[errcol], max_err=FORCED_MAX_MAGERR,
                                        sign=fp["isdiffpos"] if "isdiffpos" in fp.columns else None)
                counts["forced_gr"] = int(n_kept)
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
    bands["counts"] = counts          # diagnostics only; every loop over bands uses FID_G / FID_R
    if return_positions:
        return bands, ra0, dec0, (ra_all[pos_ok], dec_all[pos_ok])
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
    bands, ra0, dec0, (ras, decs) = get_difference_photometry(
        oid, client, use_forced=use_forced, debug_forced=debug_forced, return_positions=True)
    epochs = build_gr_epochs(bands)
    if len(epochs) < MIN_EPOCHS:
        return None
    out = {"oid": oid, "ztf_name": oid, "ra": ra0, "dec": dec0,
           "n_matched_epochs": len(epochs), "_bands": bands, "_epochs": epochs,
           "n_alert_g": bands["counts"]["alert_g"], "n_alert_r": bands["counts"]["alert_r"],
           "n_alert_i": bands["counts"]["alert_i"], "n_forced_gr": bands["counts"]["forced_gr"],
           "n_negative_gr": bands["counts"]["negative_gr"],
           "_ras": ras, "_decs": decs}
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
            r = _gaia_call(lambda: Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec).get_results())
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
                   chunk=500, quiet=True):
    """Bulk Gaia DR3 stellar check via CDS X-Match (one request per chunk).
    positions: iterable of (oid, ra_deg, dec_deg). Returns dict oid -> (is_star, info)
    for EVERY position whose chunk was answered (info has "gaia_match": False if Gaia has
    nothing there). Rejects sources with significant PM/parallax OR a bright Gaia source
    (Gmag <= proximity_maxmag) within proximity_arcsec.

    Failure handling (the service has been down for days at a time): a chunk that fails
    with a CONNECTION error trips a circuit breaker -- the rest of the run is skipped at once
    rather than timing out chunk after chunk (that cost 107 min once). Any other failure is
    retried as four smaller requests, and two failed chunks in a row also trip it. Objects
    not checked come back as (False, {"gaia_query_failed": True}): kept, re-checked on a later
    run, and run_scan checks the ones that become candidates one by one instead.
    There is deliberately NO per-object fallback here: thousands of one-at-a-time Gaia
    archive jobs take hours."""
    from astropy.table import Table
    from astroquery.xmatch import XMatch
    rows = [(str(o), float(ra), float(dec)) for (o, ra, dec) in positions
            if np.isfinite(ra) and np.isfinite(dec)]
    out = {}

    def _parse(res, sub):
        for (o, _ra, _dec) in sub:
            out[o] = (False, {"gaia_match": False})
        if res is None or len(res) == 0:
            return
        need = ("pmRA", "pmDE", "e_pmRA", "e_pmDE", "Plx", "e_Plx")
        missing = [c for c in need if c not in res.colnames]
        if missing:
            print(f"    [X-Match columns {list(res.colnames)} missing {missing}; "
                  f"cannot assess stars this chunk]")
            for (o, _ra, _dec) in sub:
                out[o] = (False, {"gaia_query_failed": True})
            return
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
                      {"gaia_match": True,
                       "gaia_sep_arcsec": sep,
                       "gaia_gmag": float(gmag) if np.isfinite(gmag) else None,
                       "pm_sig": float(pm_sig) if np.isfinite(pm_sig) else None,
                       "plx_sig": float(plx_sig) if np.isfinite(plx_sig) else None})

    def _xmatch(sub, guard=True):
        tbl = Table(rows=sub, names=["oid", "ra", "dec"])
        with _silence(quiet):
            res = _with_retries(XMatch.query, cat1=tbl, cat2="vizier:I/355/gaiadr3",
                                max_distance=radius_arcsec * u.arcsec,
                                colRA1="ra", colDec1="dec", _tries=2)
        if guard and len(sub) >= XMATCH_EMPTY_SUSPECT and (res is None or len(res) == 0):
            # Many of these objects always have a Gaia counterpart; an empty table would mark ALL
            # of them "not a star" for good, so treat it as a failed request (retried smaller).
            raise RuntimeError(f"X-Match returned no rows for {len(sub)} objects (service glitch?)")
        return res

    def _conn_error(exc):
        msg = f"{type(exc).__name__} {exc}".lower()
        return (isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))
                or any(k in msg for k in ("connecttimeout", "max retries exceeded", "connection refused",
                                          "name or service not known", "temporary failure in name")))

    def _give_up(sub, why):
        for (o, _ra, _dec) in sub:
            out[o] = (False, {"gaia_query_failed": True})
        return len(sub)

    n_chunks, n_failed, bad_in_a_row, tripped = (len(rows) + chunk - 1) // chunk, 0, 0, False
    for i in range(0, len(rows), chunk):
        sub, t0 = rows[i:i + chunk], time.monotonic()
        if tripped:
            n_failed += _give_up(sub, "breaker")
            continue
        try:
            _parse(_xmatch(sub), sub)
            bad_in_a_row = 0
            _stage(f"   gaia x-match chunk {i // chunk + 1}/{n_chunks}: {len(sub)} objects, "
                   f"{time.monotonic() - t0:.0f} s")
            continue
        except Exception as exc:
            print(f"    [bulk Gaia X-Match failed (chunk {i // chunk + 1}/{n_chunks}, "
                  f"{time.monotonic() - t0:.0f} s): {str(exc)[:200]}]")
            if _conn_error(exc):
                _stage("   !! CDS X-Match is unreachable: skipping the bulk star check for this run")
                tripped = True
                n_failed += _give_up(sub, "unreachable")
                continue
            print("    [retrying as 4 smaller requests]")
        step, ok_parts = max(1, (len(sub) + 3) // 4), 0
        answers = []
        for j in range(0, len(sub), step):
            part = sub[j:j + step]
            try:
                answers.append((part, _xmatch(part, guard=False)))
            except Exception as exc:
                n_failed += _give_up(part, "failed")
                print(f"    [Gaia X-Match failed for {len(part)} objects: {str(exc)[:200]}; keeping them "
                      f"unchecked (re-checked on a later run)]")
        if len(sub) >= XMATCH_EMPTY_SUSPECT and answers and all(r is None or len(r) == 0 for _p, r in answers):
            # same glitch test as in _xmatch, on the whole chunk: do not record "no Gaia source" for everyone
            print(f"    [X-Match answered {len(sub)} objects with no rows at all (service glitch?); "
                  f"keeping them unchecked (re-checked on a later run)]")
            for part, _r in answers:
                n_failed += _give_up(part, "empty")
            answers = []
        for part, res in answers:
            _parse(res, part)
            ok_parts += 1
        bad_in_a_row = 0 if ok_parts else bad_in_a_row + 1
        if bad_in_a_row >= 2 and not tripped:
            _stage("   !! two X-Match chunks failed in a row: skipping the bulk star check for this run")
            tripped = True
    if n_failed:
        _stage(f"   !! Gaia X-Match left {n_failed} of {len(rows)} objects unchecked")
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


def _detection_positions(ras, decs):
    """Finite float arrays of the per-detection positions, or None if there are < 2 (the
    positional covariance in check_nuclear needs at least two)."""
    if ras is None or decs is None:
        return None
    ras, decs = np.asarray(ras, dtype=float), np.asarray(decs, dtype=float)
    ok = np.isfinite(ras) & np.isfinite(decs)
    return (ras[ok], decs[ok]) if ok.sum() >= 2 else None


def is_object_nuclear(identifier, quiet=True, _retry=True, _mute=True):
    """Return (is_nuclear, p_value, sigma); all None if no usable host catalog or
    on error. plot=False skips iinuclear's (unused) image download. A 'used_catalog'
    crash usually means a stale/empty cached catalog -> clear it and retry once.

    This is iinuclear's OWN pipeline (it fetches the detections from ALeRCE itself and uses its
    SDSS-then-PS1 host). The scan does not use it any more: see nuclear_check(), which is given
    the positions Stage 2 already has and picks the host reference itself. It remains for
    objects with < 2 usable detections positions and for iinuclear-only comparisons."""
    try:
        with _silence(quiet and _mute):
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
            return is_object_nuclear(identifier, quiet=quiet, _retry=False, _mute=_mute)
        no_host = ("isfinite" in msg or "weights sum to zero" in msg
                   or "zero-size array" in msg or "used_catalog" in msg)
        if not (quiet and no_host):
            print(f"    [nuclear check failed for {identifier}: {exc}]")
        return None, None, None

# iinuclear swallows its own network errors and just prints them, so a catalog or ALeRCE
# outage looks exactly like "this galaxy has no host". These are the messages that mean
# the SERVICE failed. SDSS is handled separately (_sdss_service_error): its "no SDSS objects
# here" case prints an "Error querying SDSS: ... rename_column" line that is NOT a failure.
_NUCLEAR_SERVICE_MARKERS = ("error querying panstarrs", "error querying alerce",
                            "nuclear check failed")


def _sdss_service_error(text):
    """True if iinuclear's captured output holds an SDSS error other than the benign
    'no SDSS objects in this field' one (None.rename_column). A SkyServer outage looks like
    the former and, if PS1 then finds nothing, would be cached as 'no host'."""
    return any(ln.strip().startswith("error querying sdss") and "rename_column" not in ln
               for ln in text.splitlines())


def _catalog_reference(kind, ra_c, dec_c):
    """Host reference from SDSS or PS1 via iinuclear's own query functions (same 3" cone, same
    catalogue columns, same get_galaxy_center), but with the failure mode made explicit:
    returns (status, ref) with status 'ok' | 'no_host' | 'service_error'. iinuclear's query
    functions swallow their errors into a printed line, so the captured text is what tells
    "the server refused / is down" (SDSS has been answering with HTTP 403 pages) from
    "nothing there"."""
    with _capture_thread() as buf:
        try:
            res = (query_panstarrs if kind == "PS1" else query_sdss)(ra_c, dec_c)
        except Exception as exc:
            print(f"Error querying {'PanSTARRS' if kind == 'PS1' else 'SDSS'}: {exc}")
            res = None
    text = "".join(buf).lower()
    if res is None or len(res) == 0:
        failed = ("error querying panstarrs" in text) if kind == "PS1" else _sdss_service_error(text)
        return ("service_error" if failed else "no_host"), None
    if kind == "PS1":
        i = int(np.argmin(np.asarray(res["distance"], dtype=float)))
    else:
        i = int(np.argmin(np.asarray(calc_separations(res["ra"], res["dec"], ra_c, dec_c), dtype=float)))
    try:
        gc = _astuple(get_galaxy_center(Table(res[i:i + 1])))
        ra_g, dec_g, err = float(gc[GALCENTER_RA_IDX]), float(gc[GALCENTER_DEC_IDX]), float(gc[GALCENTER_ERR_IDX])
    except Exception:
        return "no_host", None          # catalogue row without usable positions
    if not (np.isfinite(ra_g) and np.isfinite(dec_g) and np.isfinite(err)):
        return "no_host", None
    sep = float(np.hypot((ra_g - ra_c) * np.cos(np.radians(dec_c)), dec_g - dec_c) * 3600.0)
    return "ok", {"ref": kind, "ra": ra_g, "dec": dec_g, "err": err, "type": "", "sep": sep}


def _host_unclear(ref):
    """A reference so far off and so uncertain that the sigma test against it means nothing."""
    return (ref is not None and ref.get("sep") is not None and ref.get("err") is not None
            and ref["sep"] > HOST_UNCLEAR_SEP_ARCSEC and ref["err"] >= HOST_UNCLEAR_ERR_ARCSEC)


def find_host_reference(ras, decs, use_legacy=True, notes=None):
    """Where is the host, and how well do we know it?  Tries the Legacy Survey Tractor
    catalogue first (where there is coverage), then PS1, then SDSS as the last resort; the first
    that has a source within 3" of the median detection position wins.
    Returns (ref_dict_or_None, status, failed): status 'ok' (a host), 'no_host' (EVERY
    reference answered and there is nothing), or 'service_error' (nothing found but at least one
    reference could not be asked: unknown, so no verdict). `failed` lists the references that
    errored; a host found by a later one is still used (and its name recorded). A host that is
    "unclear" (centre > HOST_UNCLEAR_SEP_ARCSEC from the ZTF position with an error >=
    HOST_UNCLEAR_ERR_ARCSEC) counts as no host from that reference; `notes` (a list) gets a line for each."""
    ra_c, dec_c = float(np.median(ras)), float(np.median(decs))
    failed = []

    def _usable(ref):
        if _host_unclear(ref):
            if notes is not None:
                notes.append(f"{ref['ref']} host rejected: centre {ref['sep']:.1f}\" from the ZTF position, "
                             f"error {ref['err']:.1f}\"")
            return False
        return True

    if use_legacy and _LS is not None:
        info, status = _LS.tractor_reference(ras, decs)
        if status == "ok":
            ref = {"ref": "LS", "ra": info["ra"], "dec": info["dec"], "err": info["err_arcsec"],
                   "type": info["type"], "sep": info["sep_arcsec"]}
            if _usable(ref):
                return ref, "ok", failed
        if status == "service_error":
            failed.append("LS")
    for kind in ("PS1", "SDSS"):
        status, ref = _catalog_reference(kind, ra_c, dec_c)
        if status == "ok" and _usable(ref):
            return ref, "ok", failed
        if status == "service_error":
            failed.append(kind)
    return None, ("service_error" if failed else "no_host"), failed


def nuclear_check(identifier, ras=None, decs=None, info=None, use_legacy=True):
    """Nuclear test for one object. Returns (is_nuclear, p_value, sigma, status) with status
    'ok' (sigma measured), 'no_host' (every reference answered; nothing to measure against) or
    'service_error' (unknown: try again later -- never record a verdict).

    With the detection positions (ras, decs; the scan has them from Stage 2) the host comes
    from find_host_reference() and NOTHING is fetched from ALeRCE -- iinuclear's get_data() would
    download the very same detections again. Without them (or with < 2), iinuclear's own
    pipeline runs as before. `info` (dict) is filled with ref ('LS'/'PS1'/'SDSS'), host_type,
    host_sep, host_err and failed (references that errored)."""
    known = _detection_positions(ras, decs)
    if known is not None:
        ras_k, decs_k = known
        notes = []
        ref, status, failed = find_host_reference(ras_k, decs_k, use_legacy=use_legacy, notes=notes)
        if info is not None:
            info.update(failed=failed, rejected=notes)
        if ref is None:
            return (None, None, None, status)
        sig, _chi, p, flag = _astuple(check_nuclear(ras_k, decs_k, ref["ra"], ref["dec"], ref["err"]))
        if info is not None:
            info.update(ref=ref["ref"], host_type=ref["type"], host_sep=ref["sep"], host_err=ref["err"])
        if not np.isfinite(sig):          # degenerate positions (singular covariance): nothing to say
            return (None, None, None, "no_host")
        return (bool(flag), float(p), float(sig), "ok")
    with _capture_thread() as buf:
        res = is_object_nuclear(identifier, quiet=True, _mute=False)
    if res[2] is not None:
        return (*res, "ok")
    text = "".join(buf).lower()
    if any(m in text for m in _NUCLEAR_SERVICE_MARKERS) or _sdss_service_error(text):
        return (None, None, None, "service_error")
    return (None, None, None, "no_host")


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
            res = _with_retries(lambda: _gaia_call(lambda: Gaia.launch_job(adql).get_results(), tries=1), _tries=2, _quiet=quiet)
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
#  Stage 6: archival host spectra (DESI DR1 AGN/Galaxy VAC)
# --------------------------------------------------------------------------- #
# Wraps spectra_finding/desi_agngal_crossmatch.py. Informational only: it never
# removes a candidate, and any failure (no catalog file, no fitsio, ...) leaves
# the columns blank with spec_checked=False, so the review page can tell "not
# checked" apart from "checked, nothing found".
#
# Caveat: this is a match against the AGN/Galaxy *value-added catalog*, which is
# a subset of DESI DR1. "No match" therefore means "not in that VAC", not "DESI
# never observed it".

SPEC_COLUMNS = ["spec_checked", "has_archival_spec", "spec_survey", "spec_type",
                "spec_z", "spec_zwarn", "spec_sep_arcsec", "spec_targetid",
                "spec_n_matches", "spec_url", "spec_agn", "spec_agn_why"]


def _import_desi_crossmatch():
    if SPEC_DIR not in sys.path:
        sys.path.insert(0, SPEC_DIR)
    import desi_agngal_crossmatch as xm      # imports fitsio at module level
    return xm


def _unrounded_z(vac_path, vac_row, fallback):
    """desi_agngal_crossmatch rounds Z to 3 dp; re-read that one row for the
    full-precision value (same pattern as its own line_measurements())."""
    try:
        import fitsio
        z = float(fitsio.read(vac_path, rows=[int(vac_row)], columns=["Z"], ext=1)[0]["Z"])
        return z if np.isfinite(z) else fallback
    except Exception:
        return fallback


def _desi_agn_why(m):
    """Why a DESI VAC row counts as an AGN, or None. SPECTYPE == QSO (the Redrock fit) or the VAC's own
    AGN_ANY bit (AGN_MASKBITS bit 0: QuasarNet / MgII / broad lines / BPT / UV / WISE ... any AGN diagnostic)."""
    why = []
    if str(m.get("SPECTYPE") or "").strip().upper() == "QSO":
        why.append("SPECTYPE=QSO")
    try:
        if int(m.get("AGN_MASKBITS") or 0) & 1:
            why.append(f"AGN_ANY (AGN_MASKBITS={int(m['AGN_MASKBITS'])})")
    except (TypeError, ValueError):
        pass
    return " + ".join(why) or None


def desi_archival_spectrum(ra, dec, xm, vac_path, radius_arcsec=SPEC_MATCH_RADIUS_ARCSEC):
    """Look up one position; returns a dict keyed by SPEC_COLUMNS.

    The whole VAC is searched (select=None), so SV spectra, repeat observations
    and ZWARN != 0 rows still count as "has a spectrum". The redshift, though, is
    only reported from a clean row (ZWARN == 0, finite Z, SPECTYPE != STAR), the
    nearest one if there are several; otherwise spec_z stays blank.
    """
    out = dict.fromkeys(SPEC_COLUMNS)
    out["spec_checked"] = True
    matches = xm.crossmatch_desi(ra, dec, radius_arcsec, select=None,
                                 verbose=False, path=vac_path)
    if not matches:
        out["has_archival_spec"] = False
        out["spec_agn"] = False
        return out
    clean = [m for m in matches
             if m["ZWARN"] == 0 and m["Z"] is not None and m["SPECTYPE"] != "STAR"]
    best = (clean or matches)[0]             # matches come back sorted by separation
    whys = [w for w in (_desi_agn_why(m) for m in (clean or matches)) if w]
    out["spec_agn"], out["spec_agn_why"] = bool(whys), (whys[0] if whys else None)
    out.update(has_archival_spec=True, spec_survey="DESI",
               spec_type=best["SPECTYPE"], spec_zwarn=best["ZWARN"],
               spec_sep_arcsec=round(best["separation_arcsec"], 3),
               spec_targetid=best["TARGETID"], spec_n_matches=len(matches),
               spec_url=best["SPECTRUM_URL"])
    if clean:
        out["spec_z"] = round(_unrounded_z(vac_path, best["VAC_ROW"], best["Z"]), 5)
    return out


def annotate_archival_spectra(finalists, vac_path=None, radius_arcsec=SPEC_MATCH_RADIUS_ARCSEC):
    """Stage 6: add the SPEC_COLUMNS to each finalist, in place. Sequential on
    purpose -- the crossmatch module keeps its catalog in module globals."""
    for s in finalists:
        for c in SPEC_COLUMNS:
            s.setdefault(c, None)
        s["spec_checked"] = False
    if not finalists:
        return

    vac_path = vac_path or os.environ.get("DESI_AGNGAL_VAC") or SPEC_VAC_DEFAULT
    try:
        if not os.path.exists(vac_path):
            raise FileNotFoundError(f"DESI VAC not found at {vac_path}")
        xm = _import_desi_crossmatch()
        xm.load_agngal(path=vac_path, select=None, verbose=False)
    except Exception as exc:
        _stage(f"Stage 6  spectra : SKIPPED ({type(exc).__name__}: {exc}); "
              f"spec_* columns left blank")
        return

    n_spec = n_z = n_fail = n_agn_spec = 0
    for s in finalists:
        if not (np.isfinite(s["ra"]) and np.isfinite(s["dec"])):
            n_fail += 1
            continue
        try:
            s.update(desi_archival_spectrum(s["ra"], s["dec"], xm, vac_path, radius_arcsec))
        except Exception as exc:
            n_fail += 1
            print(f"    [archival-spectrum check failed for {s['oid']}: {exc}]")
            continue
        n_spec += bool(s["has_archival_spec"])
        n_z += s["spec_z"] is not None
        n_agn_spec += bool(s.get("spec_agn"))
    _stage(f"Stage 6  spectra : {n_spec}/{len(finalists)} have a DESI VAC spectrum within "
          f"{radius_arcsec:g}\" ({n_z} with a clean redshift, {n_agn_spec} classified QSO/AGN -> 'In Likely AGN', {n_fail} not checked)")


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
#  Persistent per-object state: each daily run only does NEW work
# --------------------------------------------------------------------------- #
# One record per ALeRCE object, kept in a small JSON file that the workflow
# commits back to the repo (so it survives between runs):
#   "few"    fewer than MIN_EPOCHS matched g/r epochs. Re-screened only when ndet grows.
#   "red"    not blue (or stopped being blue). Never re-screened.
#   "nuc"    blue and iinuclear has been run; the sigma is stored, so the cut is applied
#            when the file is READ. Beyond sigma_max -> never re-screened. Within it ->
#            a tracked candidate: only its photometry is refreshed each day (colour,
#            rise, latest mag); nuclear / Gaia-AGN / spectrum results are reused.
#   "nohost" iinuclear returned nothing (no host catalogue, or an error); retried up to
#            NUC_MAX_ATTEMPTS times, then parked.
# Records are dropped once their first detection is older than STATE_KEEP_DAYS.
STATE_VERSION = 4
STATE_DEFAULT = os.path.join(_HERE, "state", "scan_state.json")
NUC_MAX_ATTEMPTS = 3
NUC_BREAKER = 20           # consecutive iinuclear service failures that stop the nuclear stage
NUC_RETRY_SLEEP = 15       # seconds before the 1st re-try of a failed lookup (x3 before the 2nd)
STATE_KEEP_DAYS = MAX_AGE_DAYS + 10
CACHED_KEYS = (("is_nuclear", "nuclear_pvalue", "nuclear_sigma", "nuclear_ref", "host_type", "host_err", "host_sep",
                "host_unclear", "host_note", "gaia_agn", "gaia_agn_grade",
                "gaia_agn_prob", "gaia_agn_sep_arcsec", "gaia_agn_z") + tuple(SPEC_COLUMNS))


def _state_params(use_forced):
    """Settings a cached verdict depends on. If any differ, the file is not reused."""
    return {"blue": COLOR_BLUE_THRESHOLD, "min_epochs": MIN_EPOCHS, "n_color": N_COLOR_EPOCHS,
            "match_days": GR_MATCH_WINDOW_DAYS, "forced": bool(use_forced), "positive_diff": REQUIRE_POSITIVE_DIFF,
            "gaia": [GAIA_MATCH_RADIUS_ARCSEC, GAIA_PM_SIG_THRESH, GAIA_PLX_SIG_THRESH,
                     GAIA_PROXIMITY_ARCSEC, GAIA_PROXIMITY_MAXMAG]}


def _clean(v):
    """JSON-safe copy: numpy scalars -> Python, NaN/inf/NA -> None. Python ints stay
    exact (DESI TARGETIDs are > 2**53, so they must never pass through a float)."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v if isinstance(v, str) else str(v)


class ScanState:
    def __init__(self, path, params):
        self.path, self.params = path, params
        self.rec = {}
        self.gaia = {}               # oid -> [is_star(0/1), firstmjd, pm_sig, plx_sig]; positions never move
        self.readonly = False        # True: use nothing from disk AND never overwrite it

    @classmethod
    def load(cls, path, params, reset=False):
        st = cls(path, params)
        if not path:
            return st
        if reset or not os.path.exists(path):
            _stage("State    : " + ("reset requested; starting empty" if reset
                                    else f"no state file yet ({path}); this run screens everything"))
            return st
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except Exception as exc:
            _stage(f"State    : {path} unreadable ({type(exc).__name__}: {exc}); starting empty "
                   f"(it will be rewritten)")
            return st
        migrated = 0
        if doc.get("version") == 2 and doc.get("params") == params:
            # v2 recorded iinuclear PanSTARRS/ALeRCE outages as "nohost". Those verdicts are
            # unreliable, so drop them (the objects are simply screened again).
            recs = doc.get("records", {})
            keep = {o: r for o, r in recs.items() if r.get("s") != "nohost"}
            migrated = len(recs) - len(keep)
            doc["records"], doc["version"] = keep, STATE_VERSION
        if doc.get("version") != STATE_VERSION or doc.get("params") != params:
            st.readonly = True
            _stage(f"State    : {path} was built with different settings "
                   f"({doc.get('params')} vs {params}); ignoring it and NOT overwriting it")
            return st
        st.rec = doc.get("records", {})
        st.gaia = doc.get("gaia", {})
        _stage(f"State    : loaded {len(st.rec)} object verdicts and {len(st.gaia)} Gaia star checks from {path}"
               + (f" (dropped {migrated} old 'no host' verdicts that may have been service errors)" if migrated else ""))
        return st

    def save(self, now_mjd):
        if not self.path or self.readonly:
            return
        cutoff = now_mjd - STATE_KEEP_DAYS
        self.rec = {o: r for o, r in self.rec.items()
                    if r.get("fm") is None or r["fm"] >= cutoff}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.gaia = {o: g for o, g in self.gaia.items() if g[1] is None or g[1] >= cutoff}
        lines = [f'{json.dumps(o)}:{json.dumps(self.rec[o], separators=(",", ":"), sort_keys=True)}'
                 for o in sorted(self.rec)]
        glines = [f'{json.dumps(o)}:{json.dumps(self.gaia[o], separators=(",", ":"))}'
                  for o in sorted(self.gaia)]
        head = '{"version":%d,"params":%s,"records":{\n' % (
            STATE_VERSION, json.dumps(self.params, sort_keys=True))
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(head + ",\n".join(lines) + '\n},"gaia":{\n' + ",\n".join(glines) + "\n}}\n")
        os.replace(tmp, self.path)

    def mark(self, oid, status, ndet, firstmjd, now_mjd, **extra):
        r = {"s": status, "nd": _clean(ndet), "fm": _clean(firstmjd), "chk": round(float(now_mjd), 2)}
        r.update({k: _clean(v) for k, v in extra.items()})
        self.rec[oid] = r

    def put_candidate(self, s, now_mjd, sigma_max):
        """Record a blue object whose nuclear test has been run. Beyond sigma_max only
        the nuclear numbers are kept; within it, the full output row."""
        row = {k: v for k, v in s.items() if not k.startswith("_")}
        if not s.get("host_unclear") and (s.get("nuclear_sigma") is None or s["nuclear_sigma"] > sigma_max):
            row = {k: row.get(k) for k in ("is_nuclear", "nuclear_pvalue", "nuclear_sigma", "nuclear_ref")}
        self.mark(s["oid"], "nuc", s.get("ndet"), s.get("firstmjd"), now_mjd, c=_clean(row))

    def plan(self, oids, meta, now_mjd, sigma_max, recheck_not_blue_days=0):
        """Split today's survivors into `work` (never decided, or worth another look)
        and `track` (candidates whose photometry is refreshed). Everything else is
        skipped. `meta[oid] = (firstmjd, ndet)`. Returns (work, track, skipped_counts)."""
        work, track, skipped = [], [], {}

        def skip(why):
            skipped[why] = skipped.get(why, 0) + 1

        for oid in oids:
            r = self.rec.get(oid)
            fm, nd = meta.get(oid, (np.nan, np.nan))
            if r is None:
                work.append(oid)
                continue
            grew = bool(np.isfinite(nd)) and nd > (r.get("nd") or 0)
            st = r.get("s")
            if st == "few":
                work.append(oid) if grew else skip("too few epochs, no new detections")
            elif st == "red":
                young = bool(np.isfinite(fm)) and (now_mjd - fm) < recheck_not_blue_days
                work.append(oid) if (recheck_not_blue_days > 0 and young and grew) else skip("not blue")
            elif st == "nohost":
                work.append(oid) if r.get("n", 0) < NUC_MAX_ATTEMPTS else skip("no host (parked)")
            elif st == "nuc":
                c = r.get("c") or {}
                sg = c.get("nuclear_sigma")
                track.append(oid) if (c.get("host_unclear") or (sg is not None and sg <= sigma_max)) else skip("not nuclear")
            else:
                work.append(oid)
        return work, track, skipped


# --------------------------------------------------------------------------- #
#  Full scan
def run_scan(days=MAX_AGE_DAYS, min_ndet=MIN_DETECTIONS, max_objects=None,
             require_rising=False, reject_stars=True, reject_cvnova=True,
             cvnova_classifier=CVNOVA_CLASSIFIER, cvnova_prob=CVNOVA_PROB,
             use_forced=True, sigma_max=NUCLEAR_SIGMA_MAX, workers=1, rate=5.0,
             flag_agn=True, agn_radius=GAIA_AGN_MATCH_RADIUS_ARCSEC,
             agn_min_prob=GAIA_AGN_MIN_PROB,
             check_spectra=True, spec_vac=None, spec_radius=SPEC_MATCH_RADIUS_ARCSEC,
             max_minutes=0, plotdir=None, client=None,
             state_path=None, reset_state=False, recheck_not_blue_days=0, legacy_survey=True,
             recheck_stars=False):
    """recheck_stars: also re-run the one-by-one Gaia archive star check on EVERY candidate that is about
    to be published, ignoring the cached verdicts (default off; see Stage 4b).
    legacy_survey: use the Legacy Survey Tractor host position first (then PS1, then SDSS as the
    last resort); False = PS1 then SDSS only.
    state_path: JSON file of per-object verdicts (see ScanState). With it, only
    objects not yet decided are screened and tracked candidates just get fresh
    photometry. None = stateless (screen everything, as before).

    max_minutes: wall-clock limit for the whole per-object work (0 = none). It exists
    so a cold start with a big backlog stops cleanly, SAVES its progress and lets the
    next run continue -- not to hide slow runs. Newest objects are screened first."""
    if client is None:
        client = Alerce()
    global _RATE_LIMITER
    _RATE_LIMITER = _RateLimiter(rate)
    t_start = time.monotonic()

    def bud(reserve_frac):
        """Minutes this stage may still use, keeping `reserve_frac` of the total for
        the stages after it. Time a stage doesn't use rolls on to the next one."""
        if not max_minutes:
            return None
        used = (time.monotonic() - t_start) / 60
        return max(max_minutes * (1 - reserve_frac) - used, 0.05)

    now_mjd = Time.now().mjd
    state = ScanState.load(state_path, _state_params(use_forced), reset=reset_state)

    _stage("Stage 1  recency : querying ALeRCE...")
    recent = query_recent_discoveries(client, days, min_ndet, max_objects)
    _stage(f"Stage 1  recency : {len(recent)} transients found in last {days:.0f} d "
           f"(ndet>={min_ndet})")
    if workers > 1:
        print(f"         (using {workers} worker threads for photometry/nuclear)")

    # Stage 1a: drop likely CVs/Novae (one bulk classifier query).
    if reject_cvnova:
        cvnova = query_cvnova_oids(client, days, min_ndet,
                                   classifier=cvnova_classifier, probability=cvnova_prob)
        before = len(recent)
        recent = recent[~recent["oid"].isin(cvnova)].reset_index(drop=True)
        _stage(f"Stage 1a CV/Nova : removed {before - len(recent)} CV/Nova "
               f"(p>={cvnova_prob}); {len(recent)} remain")

    meta = {str(r.oid): (r.firstmjd, r.ndet) for r in recent.itertuples()}

    # Stage 1b: reject stars before any photometry (bulk Gaia X-Match). A star's Gaia
    # verdict never changes, so only objects not checked on an earlier run are sent.
    positions = list(zip(recent["oid"], recent["meanra"], recent["meandec"]))
    if reject_stars:
        todo = [p for p in positions if p[0] not in state.gaia]
        _stage(f"Stage 1b stars   : {len(positions) - len(todo)} already checked on earlier runs; "
               f"X-matching {len(todo)} new against Gaia DR3...")
        fresh = gaia_star_mask(todo) if todo else {}
        star_map = {}
        for (o, _ra, _dec) in positions:
            if o in fresh:
                star_map[o] = fresh[o]
                if not fresh[o][1].get("gaia_query_failed"):     # unchecked ones are retried next run
                    info = fresh[o][1]
                    rnd = lambda v: None if v is None else round(float(v), 2)
                    state.gaia[o] = [int(fresh[o][0]), _clean(meta[o][0]),
                                     rnd(info.get("pm_sig")), rnd(info.get("plx_sig"))]
            elif o in state.gaia:
                flag, _fm, pm, plx = state.gaia[o]
                star_map[o] = (bool(flag), {"pm_sig": pm, "plx_sig": plx})
        oids = [o for (o, _ra, _dec) in positions if not star_map.get(o, (False, {}))[0]]
        _stage(f"Stage 1b stars   : removed {len(positions) - len(oids)} Gaia stars "
               f"before photometry; {len(oids)} remain")
        default_star = False
    else:
        star_map = {}
        oids = [o for (o, _ra, _dec) in positions]
        default_star = None
    state.save(now_mjd)          # checkpoint: Gaia star verdicts are banked

    def newest_first(o):
        fm = meta.get(o, (np.nan, np.nan))[0]
        return -float(fm) if np.isfinite(fm) else 0.0

    # What actually needs doing today (everything, if there is no state file yet).
    work, track, skipped = state.plan(oids, meta, now_mjd, sigma_max, recheck_not_blue_days)
    track_set = set(track)
    why = "; ".join(f"{n} {w}" for w, n in sorted(skipped.items())) or "none"
    _stage(f"Cache    : {len(work)} to screen (new / retry), {len(track)} tracked candidates to "
           f"refresh, {sum(skipped.values())} already decided and skipped ({why})")
    photo_oids = sorted(work + track, key=newest_first)

    # Stage 2: photometry (alert + forced) + epochs, only where needed.
    _stage(f"Stage 2  epochs  : pulling photometry for {len(photo_oids)} objects...")
    photo = _parallel_map(
        lambda oid: screen_photometry(oid, _get_client(), use_forced=use_forced), photo_oids, workers,
        on_error=lambda oid, exc: print(f"    [photometry failed for {oid}: {exc}]"),
        label="photometry", budget_min=bud(0.35))
    photo_pass, n_dropped = [], 0
    for oid, s in photo:
        fm, nd = meta.get(oid, (np.nan, np.nan))
        if s is None:                                   # too few matched g/r epochs
            if oid not in track_set:
                state.mark(oid, "few", nd, fm, now_mjd)
            continue
        star, ginfo = star_map.get(oid, (default_star, {}))
        s["is_star"] = None if ginfo.get("gaia_query_failed") else star
        s["gaia_pm_sig"] = ginfo.get("pm_sig")
        s["gaia_plx_sig"] = ginfo.get("plx_sig")
        s["firstmjd"], s["ndet"] = fm, nd
        photo_pass.append(s)
        if not s["is_blue"]:
            n_dropped += oid in track_set
            state.mark(oid, "red", nd, fm, now_mjd, drop=int(oid in track_set))
    _stage(f"Stage 2  epochs  : {len(photo_pass)} have >= {MIN_EPOCHS} matched g/r epochs")
    state.save(now_mjd)          # checkpoint: not-blue / too-few-epochs verdicts are banked

    blue = [s for s in photo_pass if s["is_blue"]]
    tracked_blue = [s for s in blue if s["oid"] in track_set]
    blue_new = sorted((s for s in blue if s["oid"] not in track_set), key=lambda s: newest_first(s["oid"]))
    for s in tracked_blue:                              # reuse nuclear / AGN / spectrum results
        cached = state.rec[s["oid"]].get("c") or {}
        for k in CACHED_KEYS:
            if k in cached:
                s[k] = cached[k]
    _stage(f"Stage 3  blue    : {len(blue)} are currently blue (g-r < {COLOR_BLUE_THRESHOLD}): "
           f"{len(tracked_blue)} tracked candidates, {len(blue_new)} newly blue; "
           f"{n_dropped} tracked candidates dropped (no longer blue)")

    # Stage 4: nuclear test -- only on objects never tested before. iinuclear hides its own
    # network failures, so nuclear_check() says whether "no answer" means "no host" (a real
    # verdict, recorded) or "PanSTARRS/ALeRCE errored" (retried here, then left undecided so the
    # next run tries again). A run of consecutive service failures stops the stage early.
    _stage(f"Stage 4  nuclear : testing {len(blue_new)} newly blue objects (host reference: "
           f"{'Legacy Survey, ' if legacy_survey and _LS is not None else ''}PS1, then SDSS as last resort; newest first)...")
    nuc_lock, nuc_fail = threading.Lock(), {"run": 0, "tripped": False}

    def _nuclear_task(s):
        if nuc_fail["tripped"]:
            return (None, None, None, "skipped")
        info = {}
        for attempt in range(3):
            info = {}
            res = nuclear_check(s["oid"], s.get("_ras"), s.get("_decs"), info=info, use_legacy=legacy_survey)
            if res[3] != "service_error":
                break
            if attempt < 2:
                time.sleep(NUC_RETRY_SLEEP * (3 ** attempt))
        if res[3] == "ok":
            s["nuclear_ref"], s["host_type"] = info.get("ref"), info.get("host_type") or None
            s["host_err"], s["host_sep"] = info.get("host_err"), info.get("host_sep")
            s["_ref_failed"] = info.get("failed") or []
        elif res[3] == "no_host":
            s["host_note"] = "; ".join(info.get("rejected") or []) or "no source within 3\" in any reference catalogue"
        with nuc_lock:
            if res[3] == "service_error":
                nuc_fail["run"] += 1
                if nuc_fail["run"] >= NUC_BREAKER and not nuc_fail["tripped"]:
                    nuc_fail["tripped"] = True
                    _stage(f"   !! {nuc_fail['run']} host lookups in a row failed (Legacy Survey / PanSTARRS / "
                           f"SDSS / ALeRCE errors): stopping the nuclear stage; the rest is retried next run")
            else:
                nuc_fail["run"] = 0
        return res

    nuc = _parallel_map(_nuclear_task, blue_new, workers,
                        on_error=lambda s, exc: print(f"    [nuclear failed for {s['oid']}: {exc}]"),
                        label="nuclear", budget_min=bud(0.06))
    tested, undetermined, n_svc = set(), [], 0
    for s, (is_nuc, p_val, sigma, status) in nuc:
        if status in ("service_error", "skipped"):
            n_svc += 1                                   # no verdict recorded: it will be retried
            continue
        s["is_nuclear"], s["nuclear_pvalue"], s["nuclear_sigma"] = is_nuc, p_val, sigma
        tested.add(s["oid"])
        if sigma is None:                                # every catalogue answered; none gives a usable host
            s["host_unclear"] = True                     # published as a candidate, in its own "Host Unclear" bin
            undetermined.append(s["oid"])
        state.put_candidate(s, now_mjd, sigma_max)
    for s in tracked_blue:                              # keep their stored row current
        state.put_candidate(s, now_mjd, sigma_max)
    newly_ok = [s for s in blue_new if s["oid"] in tested
                and (s.get("host_unclear") or (s.get("nuclear_sigma") is not None and s["nuclear_sigma"] <= sigma_max))]
    n_far = sum(1 for s in blue_new if s["oid"] in tested
                and s.get("nuclear_sigma") is not None and s["nuclear_sigma"] > sigma_max)
    n_unreached = len(blue_new) - len(tested) - n_svc
    refs_used = collections.Counter(s.get("nuclear_ref") for s in blue_new
                                    if s["oid"] in tested and s.get("nuclear_sigma") is not None)
    if refs_used:
        n_fb = sum(1 for s in blue_new if s.get("_ref_failed"))
        _stage("Stage 4  nuclear : host reference used: "
               + ", ".join(f"{k} {v}" for k, v in sorted(refs_used.items(), key=lambda kv: -kv[1]))
               + (f"; {n_fb} used a later reference because an earlier one errored" if n_fb else ""))
    finalists = tracked_blue + newly_ok
    # Tracked candidates whose refresh failed or ran out of time: publish the last known row.
    fresh = {s["oid"] for s in photo_pass}
    stale = [dict((state.rec[o].get("c") or {})) for o in track if o not in fresh]
    stale = [c for c in stale if "oid" in c]
    finalists += stale
    _stage(f"Stage 4  nuclear : {len(finalists)} candidates within {sigma_max:g}sigma of host or with an unclear host "
           f"({len(tracked_blue)} refreshed, {len(stale)} last-known, {len(newly_ok)} new; "
           f"of the new tests {n_far} farther, {len(undetermined)} Host Unclear (no usable host in any reference)"
           + (f"; {n_svc} lookups failed on service errors -> retried next run" if n_svc else "")
           + (f"; {n_unreached} not reached yet -> next run" if n_unreached else "") + ")")
    if undetermined:
        shown = ", ".join(undetermined[:25]) + (f", ... (+{len(undetermined) - 25} more)" if len(undetermined) > 25 else "")
        print(f"         Host Unclear (published in their own bin): {shown}")
    state.save(now_mjd)          # checkpoint: the expensive part is now banked

    # Candidates the bulk X-Match never answered for (service down, or an earlier run): one
    # Gaia archive cone search each. A few hundred at most, instead of all ~17k objects.
    if reject_stars:
        if recheck_stars:
            unchecked = list(finalists)
        else:
            unchecked = [s for s in finalists if s["oid"] not in state.gaia]
        if unchecked:
            _stage(f"Stage 4b stars  : "
                   + (f"re-checking all {len(unchecked)} candidates (--recheck-stars; cached verdicts ignored)"
                      if recheck_stars else f"{len(unchecked)} candidates were not star-checked in bulk")
                   + "; one by one (Gaia archive)...")
            res = _parallel_map(
                lambda s: is_gaia_star(s["ra"], s["dec"]), unchecked, workers,
                on_error=lambda s, exc: print(f"    [Gaia star check failed for {s['oid']}: {exc}]"),
                label="gaia-star", budget_min=bud(0.12))
            n_star = n_flipped = n_nomatch = 0
            for s, (is_star, info) in res:
                if info.get("gaia_query_failed"):
                    continue                                   # unknown: kept, retried next run
                if recheck_stars:
                    before = state.gaia.get(s["oid"])
                    n_flipped += bool(is_star) and not (before and before[0])
                    n_nomatch += not info.get("gaia_match")
                    what = ("no Gaia source within %.1f\"" % GAIA_MATCH_RADIUS_ARCSEC if not info.get("gaia_match") else
                            "Gaia source %.2f\" away, G=%s, pm_sig=%s, plx_sig=%s" % (
                                info.get("gaia_sep_arcsec") or float("nan"), info.get("gaia_gmag"),
                                None if info.get("pm_sig") is None else round(info["pm_sig"], 1),
                                None if info.get("plx_sig") is None else round(info["plx_sig"], 1)))
                    print(f"         recheck {s['oid']}: {'STAR' if is_star else 'not a star'} ({what})"
                          + ("  <-- newly rejected" if is_star and not (before and before[0]) else ""))
                rnd = lambda v: None if v is None else round(float(v), 2)
                state.gaia[s["oid"]] = [int(is_star), _clean(s.get("firstmjd")),
                                        rnd(info.get("pm_sig")), rnd(info.get("plx_sig"))]
                s["is_star"], s["gaia_pm_sig"], s["gaia_plx_sig"] = is_star, info.get("pm_sig"), info.get("plx_sig")
                n_star += bool(is_star)
            finalists = [s for s in finalists if not state.gaia.get(s["oid"], [0])[0]]
            _stage(f"Stage 4b stars  : {n_star} of the {len(unchecked)} are Gaia stars and were removed; "
                   f"{len(res) - n_star} confirmed non-stellar; {len(unchecked) - len(res)} not checked"
                   + (f" ({n_flipped} newly rejected by the re-check; {n_nomatch} have no Gaia source "
                      f"within {GAIA_MATCH_RADIUS_ARCSEC:g}\")" if recheck_stars else ""))

    if require_rising:
        finalists = [s for s in finalists if s["is_rising"]]
        print(f"         rising  : {len(finalists)} of those are also rising")

    # Stage 5: Gaia AGN annotation -- informational only; reuse cached results.
    for s in finalists:
        s.setdefault("gaia_agn", None)
    need_agn = [s for s in finalists if s.get("gaia_agn") is None]
    if flag_agn and need_agn:
        agn_res = _parallel_map(
            lambda s: gaia_agn_flag(s["ra"], s["dec"], agn_radius, agn_min_prob),
            need_agn, workers,
            on_error=lambda s, exc: print(f"    [AGN check failed for {s['oid']}: {exc}]"),
            label="gaia-agn", budget_min=bud(0.0))
        for s, (is_agn, info) in agn_res:
            s["gaia_agn"] = is_agn
            s["gaia_agn_grade"] = info.get("grade")
            s["gaia_agn_prob"] = info.get("qso_prob")
            s["gaia_agn_sep_arcsec"] = info.get("sep_arcsec")
            s["gaia_agn_z"] = info.get("redshift_qsoc")
    if flag_agn and finalists:
        n_agn = sum(1 for s in finalists if s.get("gaia_agn") is True)
        n_likely = sum(1 for s in finalists if s.get("gaia_agn_grade") == "likely")
        n_unk = sum(1 for s in finalists if s.get("gaia_agn") is None)
        _stage(f"Stage 5  AGN flag: {n_agn}/{len(finalists)} match a Gaia DR3 QSO "
               f"candidate within {agn_radius:g}\" ({n_likely} graded 'likely', "
               f"{n_unk} undetermined; {len(need_agn)} newly checked)")

    # Stage 6: archival host spectra -- informational only. Runs last, so it only
    # ever sees candidates that survived every cut above (incl. --rising). Candidates
    # whose spectrum was already checked keep that result.
    if check_spectra:
        annotate_archival_spectra([s for s in finalists if not s.get("spec_checked")],
                                  spec_vac, spec_radius)

    for s in finalists:                                # bank the AGN / spectrum results
        if s.get("nuclear_sigma") is not None or s.get("host_unclear"):
            state.put_candidate(s, now_mjd, sigma_max)
    state.save(now_mjd)

    if plotdir and finalists:
        os.makedirs(plotdir, exist_ok=True)
        for s in finalists:
            if "_epochs" in s:                          # last-known rows have no light curve
                plot_lightcurve(s, savepath=os.path.join(plotdir, f"{s['oid']}.png"))
        print(f"         plots   : wrote figure(s) to {plotdir}/")

    return finalists


def results_table(finalists):
    # latest_mag = combined g/r *difference-image* magnitude of the latest matched
    # epoch (the review page's "latest epoch brighter than" cut).
    cols = ["oid", "ra", "dec", "firstmjd", "ndet", "n_matched_epochs",
            "n_alert_g", "n_alert_r", "n_alert_i", "n_forced_gr", "n_negative_gr", "gr_color", "is_blue",
            "is_rising", "delta_mag", "latest_mag", "is_star", "gaia_pm_sig", "gaia_plx_sig",
            "nuclear_sigma", "nuclear_pvalue", "is_nuclear", "nuclear_ref", "host_type", "host_err", "host_sep",
            "host_unclear", "host_note",
            "gaia_agn", "gaia_agn_grade", "gaia_agn_prob",
            "gaia_agn_sep_arcsec", "gaia_agn_z"] + SPEC_COLUMNS
    rows = [{c: s.get(c) for c in cols} for s in finalists]
    df = pd.DataFrame(rows, columns=cols)
    # DESI TARGETIDs are ~4e16 (> 2**53): pandas would turn an int column with
    # blanks into float64 and silently corrupt them. Keep these as nullable ints.
    for c in ("spec_targetid", "spec_zwarn", "spec_n_matches"):
        df[c] = pd.array([r.get(c) for r in rows], dtype="Int64")
    return df


# --------------------------------------------------------------------------- #
#  Single-object path (debug)
# --------------------------------------------------------------------------- #
def classify_one(name, client=None, want_plot=False, reject_stars=True, use_forced=True,
                 check_spectra=True, spec_vac=None, spec_radius=SPEC_MATCH_RADIUS_ARCSEC):
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
        print(f"    Gaia star check: {ginfo if ginfo else 'no position'}")
    else:
        s["is_star"] = None
    _info = {}
    (s["is_nuclear"], s["nuclear_pvalue"], s["nuclear_sigma"], _status) = nuclear_check(
        name, s.get("_ras"), s.get("_decs"), info=_info)
    s["nuclear_ref"], s["host_type"] = _info.get("ref"), _info.get("host_type") or None
    s["host_err"], s["host_sep"] = _info.get("host_err"), _info.get("host_sep")
    if _status != "ok":
        print(f"{name}: nuclear test gave no result ({_status.replace('_', ' ')}"
              + (f"; failed: {', '.join(_info['failed'])}" if _info.get("failed") else "") + ")")
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
          f"is_nuclear={s['is_nuclear']}, {agn}, {s['n_matched_epochs']} epochs, "
          f"latest m={s['latest_mag']:.2f}")
    if check_spectra:                     # same Stage 6 the full scan runs, for one object
        annotate_archival_spectra([s], spec_vac, spec_radius)
        if s.get("spec_checked"):
            where = f"{s['ra']:.5f}, {s['dec']:.5f}"
            if s["has_archival_spec"]:
                print(f"    DESI VAC: {s['spec_type']}  z={s['spec_z']}  ZWARN={s['spec_zwarn']}  "
                      f"sep={s['spec_sep_arcsec']}\"  ({s['spec_n_matches']} row(s))  {s['spec_url']}"
                      + (f"  -> AGN ({s['spec_agn_why']})" if s.get("spec_agn") else ""))
            else:
                print(f"    DESI VAC: no row within {spec_radius:g}\" of ({where}) "
                      f"[this position was searched]")
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
    p.add_argument("--sigma-max", type=float, default=NUCLEAR_SIGMA_MAX, help="keep transients within this many sigma of host (default 2.4477, i.e. iinuclear's p > 0.05)")
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
    p.add_argument("--no-spec-check", action="store_true",
                   help="skip the Stage 6 archival-spectrum check (DESI VAC) of final candidates")
    p.add_argument("--spec-vac", default=None, metavar="FITS",
                   help="path to agngal_desi.fits (default: $DESI_AGNGAL_VAC, else "
                        "spectra_finding/data/agngal_desi.fits)")
    p.add_argument("--spec-radius", type=float, default=SPEC_MATCH_RADIUS_ARCSEC,
                   help="match radius in arcsec for the archival-spectrum check (default 2.0)")
    p.add_argument("--max-minutes", type=float, default=0,
                   help="wall-clock limit in minutes for the per-object work (default 0 = none). On a "
                        "cold start with a big backlog the scan stops cleanly, saves its progress to "
                        "the state file and the next run carries on (newest objects first).")
    p.add_argument("--state", default=STATE_DEFAULT, metavar="JSON",
                   help="per-object verdict cache so each run only screens NEW objects "
                        "(default: state/scan_state.json next to this script)")
    p.add_argument("--no-state", action="store_true", help="ignore and do not write the state file (screen everything)")
    p.add_argument("--reset-state", action="store_true", help="discard the state file and rebuild it from scratch")
    p.add_argument("--recheck-stars", action="store_true",
                   help="re-run the one-by-one Gaia archive star check on every candidate about to be published, "
                        "ignoring cached verdicts (default off; for a manual run after Gaia errors)")
    p.add_argument("--no-legacy-survey", action="store_true",
                   help="do not use Legacy Survey Tractor host positions (PS1, then SDSS only)")
    p.add_argument("--recheck-not-blue-days", type=float, default=0,
                   help="also re-screen objects judged 'not blue' while they are younger than this many "
                        "days and have gained detections (default 0 = never re-screen them)")

    args = p.parse_args()

    if args.target:
        classify_one(args.target, want_plot=args.plot, reject_stars=not args.keep_stars,
                     use_forced=not args.no_forced, check_spectra=not args.no_spec_check,
                     spec_vac=args.spec_vac, spec_radius=args.spec_radius)
    else:
        if os.path.exists("scan_partial.txt"):
            os.remove("scan_partial.txt")
        if args.max_minutes:
            # Last-resort watchdog: if ANYTHING (a stage the budget doesn't cover, a hung
            # library call) is still running 25 min after the budget, print every thread's
            # stack -- so the log shows exactly where it is stuck -- and exit non-zero.
            # That lets the workflow still commit the saved state before its 240 min limit.
            import faulthandler
            faulthandler.dump_traceback_later((args.max_minutes + 25) * 60, exit=True)
        finalists = run_scan(days=args.days, min_ndet=args.min_ndet, max_objects=args.max_objects,
                             require_rising=args.rising, reject_stars=not args.keep_stars,
                             reject_cvnova=not args.keep_cvnova, cvnova_classifier=args.cvnova_classifier,
                             cvnova_prob=args.cvnova_prob, use_forced=not args.no_forced,
                             sigma_max=args.sigma_max, workers=args.workers, rate=args.rate,
                             flag_agn=not args.no_agn_flag, agn_radius=args.agn_radius,
                             agn_min_prob=args.agn_min_prob,
                             check_spectra=not args.no_spec_check, spec_vac=args.spec_vac,
                             spec_radius=args.spec_radius, max_minutes=args.max_minutes,
                             plotdir=args.plotdir if args.plot else None,
                             state_path=None if args.no_state else args.state,
                             reset_state=args.reset_state,
                             recheck_not_blue_days=args.recheck_not_blue_days,
                             legacy_survey=not args.no_legacy_survey,
                             recheck_stars=args.recheck_stars)
        if args.max_minutes:
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
        table = results_table(finalists)
        print("\n=== Final candidates (blue + non-stellar + non-CV + nuclear) ===")
        print(table.to_string(index=False) if len(table) else "(none)")
        exit_code = 0
        if args.out:
            if _PARTIAL and len(table) == 0:
                # Never replace the published list with an empty one from a run that
                # ran out of time: fail loudly and keep the previous CSV.
                print("\nERROR: time budget hit and nothing finished; NOT writing " + args.out,
                      file=sys.stderr)
                exit_code = 1
            else:
                table.to_csv(args.out, index=False)
                print(f"\nwrote {len(table)} row(s) -> {args.out}")
                if _PARTIAL:
                    with open("scan_partial.txt", "w") as fh:
                        fh.write("PARTIAL SCAN (time budget): " + "; ".join(_PARTIAL) + "\n")
        if _PARTIAL:
            # Abandoned worker threads may still be blocked in a network call and
            # would keep the interpreter from exiting; leave without joining them.
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(exit_code)
        sys.exit(exit_code)
