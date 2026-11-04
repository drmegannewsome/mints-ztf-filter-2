"""Positional crossmatch against the DESI DR1 AGN/Galaxy Classification VAC.

Give it an (RA, Dec) and it returns *every* agngal_desi row whose
TARGET_RA / TARGET_DEC falls within ``radius_arcsec`` (default 1"), with the
redshift, SPECTYPE, the raw AGN_MASKBITS / OPT_UV_TYPE / IR_TYPE values, and
the spectrum URL for its TARGETID.

Typical use::

    from desi_agngal_crossmatch import crossmatch_desi

    matches = crossmatch_desi(55.5673516, -4.7308623)        # 1" default
    matches = crossmatch_desi(55.5673516, -4.7308623, 3.0)   # 3"
    matches = crossmatch_desi(55.5673516, -4.7308623, select=None)  # whole VAC

By default the catalog is cut to the clean main-survey sample
(SURVEY=='main', MAIN_PRIMARY, ZWARN==0), exactly as in your selection cell.
Pass ``select=None`` to search every row of the VAC instead, including SV
spectra and repeat observations of the same TARGETID.
"""

import os

import numpy as np
import fitsio
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky


# ============================================================
# Configuration
# ============================================================

VAC_FILE = os.environ.get(
    "DESI_AGNGAL_VAC",
    "/Users/yaeldgany/Desktop/PhD/DESI/agngal_desi.fits",   # <- edit me
)

LS_VIEWER = "https://www.legacysurvey.org/viewer"

# HDU1 columns kept for the crossmatch
EXT1_COLS = [
    'TARGETID', 'SURVEY', 'PROGRAM', 'Z', 'ZWARN', 'SPECTYPE',
    'TARGET_RA', 'TARGET_DEC', 'AGN_MASKBITS', 'OPT_UV_TYPE', 'IR_TYPE',
    'MEAN_MJD', 'HEALPIX', 'COADD_EXPTIME', 'MAIN_PRIMARY',
]

# Module-level catalog cache
_CAT = None               # structured array of the selected rows
_CAT_ROWS = None          # global row indices into the VAC file
_CAT_COORDS = None        # SkyCoord of the finite TARGET_RA/DEC rows
_CAT_ROWIDX = None        # coord index -> index into _CAT
_CAT_PATH = None
_CAT_SELECT = None


# ============================================================
# Small utilities
# ============================================================

def _value(x):
    """Convert Astropy/Numpy/FITS values into clean Python values."""
    if x is None or np.ma.is_masked(x):
        return None
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode(errors="ignore").strip()
    if isinstance(x, str):
        x = x.strip()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _round(x, ndigits=3):
    """Round a float for display/output; pass everything else through."""
    x = _value(x)
    return None if x is None else round(float(x), ndigits)


# ============================================================
# DESI spectrum URL
# ============================================================

def desi_spectrum_url(targetid):
    """Interactive spectrum in the Legacy Survey viewer (as in the inspector)."""
    targetid = _value(targetid)
    if targetid is None:
        return None
    return f"{LS_VIEWER}/desi-spectrum/dr1/targetid{int(targetid)}"


# ============================================================
# Loading
# ============================================================

def _selection_mask(path, select):
    """Row indices passing ``select``; None means every row."""
    if select is None:
        return None

    cols = ['SURVEY', 'MAIN_PRIMARY', 'ZWARN']
    sel = fitsio.read(path, columns=cols, ext=1)

    survey = np.char.strip(np.asarray(sel['SURVEY']).astype(str))
    keep = np.ones(len(sel), dtype=bool)

    if "main" in select:
        keep &= (survey == 'main')
    if "primary" in select:
        keep &= np.asarray(sel['MAIN_PRIMARY'], dtype=bool)
    if "good" in select:
        keep &= (np.asarray(sel['ZWARN']) == 0)

    return np.flatnonzero(keep)


def load_agngal(path=None, columns=EXT1_COLS,
                select="main_primary_good", force=False, verbose=True):
    """Read the VAC once and cache it with its sky coordinates.

    ``select`` is a string containing any of 'main', 'primary', 'good'
    (the default applies all three, reproducing your selection cell), or
    None to keep every row of the VAC.
    """
    global _CAT, _CAT_ROWS, _CAT_COORDS, _CAT_ROWIDX, _CAT_PATH, _CAT_SELECT

    path = path or VAC_FILE

    if (_CAT is not None and not force
            and path == _CAT_PATH and select == _CAT_SELECT):
        return _CAT

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Point DESI_AGNGAL_VAC at your local copy "
            "of agngal_desi.fits."
        )

    rows = _selection_mask(path, select)

    if rows is None:
        cat = fitsio.read(path, columns=columns, ext=1)
        rows = np.arange(len(cat))
    else:
        cat = fitsio.read(path, rows=rows, columns=columns, ext=1)

    ra = np.asarray(cat['TARGET_RA'], dtype=float)
    dec = np.asarray(cat['TARGET_DEC'], dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec)

    _CAT = cat
    _CAT_ROWS = rows
    _CAT_ROWIDX = np.flatnonzero(good)
    _CAT_COORDS = SkyCoord(ra[good] * u.deg, dec[good] * u.deg)
    _CAT_PATH = path
    _CAT_SELECT = select

    if verbose:
        print(f"keep {len(cat)} DESI rows  (select={select!r})")

    return _CAT


def line_measurements(match, path=None, ext=2):
    """HDU2 emission-line row for one match, read on demand.

    ``match`` is an entry returned by crossmatch_desi(); only the single
    row it points at is read, so HDU2 never has to be held in memory.
    """
    path = path or _CAT_PATH or VAC_FILE
    row = fitsio.read(path, rows=[int(match["VAC_ROW"])], ext=ext)[0]
    return {name: _value(row[name]) for name in row.dtype.names}


# ============================================================
# Crossmatch
# ============================================================

def crossmatch_desi(ra, dec, radius_arcsec=1.0, select="main_primary_good",
                    spectype=None, verbose=True, path=None):
    """All DESI VAC spectra within ``radius_arcsec`` of (ra, dec).

    Parameters
    ----------
    ra, dec : float
        Search position, degrees (ICRS).
    radius_arcsec : float
        Match radius against TARGET_RA / TARGET_DEC. Default 1.0.
    select : str or None
        Catalog cut applied at load time (see load_agngal).
    spectype : str, iterable or None
        Optionally keep only e.g. 'QSO', or ('QSO', 'GALAXY').

    Returns
    -------
    list of dict, sorted by separation, one entry per matched spectrum.
    """
    load_agngal(path=path, select=select, verbose=verbose)

    ra = float(ra)
    dec = float(dec)
    radius_arcsec = float(radius_arcsec)

    target = SkyCoord([ra] * u.deg, [dec] * u.deg)

    _, idx, sep, _ = search_around_sky(
        target, _CAT_COORDS, radius_arcsec * u.arcsec,
    )

    order = np.argsort(sep.arcsec)

    if verbose:
        print(f'Input: RA={ra}, Dec={dec}, radius={radius_arcsec}"')

    if len(order) == 0:
        if verbose:
            print(f"No DESI spectrum found within {radius_arcsec} arcsec "
                  "at this position.")
        return []

    if isinstance(spectype, str):
        spectype = (spectype,)
    if spectype is not None:
        spectype = {s.strip().upper() for s in spectype}

    results = []

    for k in order:
        i = _CAT_ROWIDX[idx[k]]
        row = _CAT[i]

        stype = str(_value(row['SPECTYPE']) or "").upper()
        if spectype is not None and stype not in spectype:
            continue

        targetid = int(row['TARGETID'])
        survey = _value(row['SURVEY'])
        program = _value(row['PROGRAM'])
        healpix = _value(row['HEALPIX'])

        entry = {
            "separation_arcsec": float(sep[k].arcsec),
            "TARGETID": targetid,
            "VAC_ROW": int(_CAT_ROWS[i]),      # row index in the VAC file
            "SURVEY": survey,
            "PROGRAM": program,
            "SPECTYPE": stype,
            "Z": _round(row['Z']),
            "ZWARN": _value(row['ZWARN']),
            "MAIN_PRIMARY": bool(_value(row['MAIN_PRIMARY'])),
            "TARGET_RA": _value(row['TARGET_RA']),
            "TARGET_DEC": _value(row['TARGET_DEC']),
            "MEAN_MJD": _value(row['MEAN_MJD']),
            "COADD_EXPTIME": _value(row['COADD_EXPTIME']),
            "HEALPIX": int(healpix) if healpix is not None else None,
            "AGN_MASKBITS": _value(row['AGN_MASKBITS']),
            "OPT_UV_TYPE": _value(row['OPT_UV_TYPE']),
            "IR_TYPE": _value(row['IR_TYPE']),
            "SPECTRUM_URL": desi_spectrum_url(targetid),
        }
        results.append(entry)

    if verbose:
        _print_matches(results)

    return results


def desi_by_targetid(targetid, select=None, verbose=True, path=None):
    """Every VAC row for one TARGETID (all surveys / programs).

    Run with ``select=None`` so repeat and SV spectra are not cut away.
    """
    load_agngal(path=path, select=select, verbose=verbose)

    targetid = int(targetid)
    hits = np.flatnonzero(np.asarray(_CAT['TARGETID']) == targetid)

    results = []
    for i in hits:
        row = _CAT[i]
        survey, program = _value(row['SURVEY']), _value(row['PROGRAM'])
        healpix = _value(row['HEALPIX'])
        results.append({
            "separation_arcsec": None,
            "TARGETID": targetid,
            "VAC_ROW": int(_CAT_ROWS[i]),
            "SURVEY": survey,
            "PROGRAM": program,
            "SPECTYPE": str(_value(row['SPECTYPE']) or "").upper(),
            "Z": _round(row['Z']),
            "ZWARN": _value(row['ZWARN']),
            "MAIN_PRIMARY": bool(_value(row['MAIN_PRIMARY'])),
            "TARGET_RA": _value(row['TARGET_RA']),
            "TARGET_DEC": _value(row['TARGET_DEC']),
            "MEAN_MJD": _value(row['MEAN_MJD']),
            "COADD_EXPTIME": _value(row['COADD_EXPTIME']),
            "HEALPIX": int(healpix) if healpix is not None else None,
            "AGN_MASKBITS": _value(row['AGN_MASKBITS']),
            "OPT_UV_TYPE": _value(row['OPT_UV_TYPE']),
            "IR_TYPE": _value(row['IR_TYPE']),
            "SPECTRUM_URL": desi_spectrum_url(targetid),
        })

    results.sort(key=lambda x: (str(x["SURVEY"]), str(x["PROGRAM"])))

    if verbose:
        _print_matches(results)

    return results


# ============================================================
# Printing
# ============================================================

def _print_matches(results):
    if not results:
        print("No matches.")
        return

    by_targetid = {}
    for r in results:
        by_targetid.setdefault(r["TARGETID"], []).append(r)

    n_t = len(by_targetid)
    print(f"Found {len(results)} match(es), {n_t} distinct TARGETID"
          f"{'s' if n_t != 1 else ''}")

    for targetid, spectra in by_targetid.items():
        print("\n" + "-" * 60)
        print(f"TARGETID={targetid}   {len(spectra)} spectrum/spectra")

        for n, e in enumerate(spectra, 1):
            prefix = f"  [{n}] " if len(spectra) > 1 else "  "
            sep_text = (
                "" if e["separation_arcsec"] is None
                else f'  sep={e["separation_arcsec"]:.3f}"'
            )
            z_text = "None" if e["Z"] is None else f'{e["Z"]:.3f}'
            print(f'{prefix}SPECTYPE={e["SPECTYPE"]}  Z={z_text}  '
                  f'ZWARN={e["ZWARN"]}{sep_text}')
            print(f'      SURVEY={e["SURVEY"]}  PROGRAM={e["PROGRAM"]}  '
                  f'HEALPIX={e["HEALPIX"]}  MEAN_MJD={e["MEAN_MJD"]}  '
                  f'EXPTIME={e["COADD_EXPTIME"]}')
            print(f'      AGN_MASKBITS={e["AGN_MASKBITS"]}  '
                  f'OPT_UV_TYPE={e["OPT_UV_TYPE"]}  IR_TYPE={e["IR_TYPE"]}')
            print(f'      Spectrum: {e["SPECTRUM_URL"]}')


__all__ = [
    "crossmatch_desi",
    "desi_by_targetid",
    "load_agngal",
    "line_measurements",
    "desi_spectrum_url",
    "VAC_FILE",
    "EXT1_COLS",
]
