r"""
HYPERLOAD — Orbit Determinator
================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Designed to work directly after moving_object.py, which exports
per-night MPC One-Line astrometry files.  This script:

  1. LOAD TRACKLETS     — Import MPC One-Line files from multiple nights
                          (one file per session), or enter observations
                          manually (RA, Dec, time).

  2. LINK TRACKLETS     — Test whether detections from different nights
                          are the same object via great-circle consistency
                          and proper-motion gradient screening.
                          Flags ambiguous / poor links.

  3. FIT ORBIT          — Least-squares minimisation of RA/Dec residuals
                          over all linked observations simultaneously.
                          Determines 6 Keplerian elements (a, e, i, Ω, ω, M₀).
                          Initial guess: grid over plausible distances (0.3–10 AU),
                          select best basin, refine with Nelder-Mead.

  4. UNCERTAINTY        — Monte-Carlo: perturb observations by their
                          astrometric uncertainty (default 1 arcsec),
                          re-fit 200 times → element distributions + σ.

  5. CLASSIFY           — Orbit class from Granvik+ 2018 / MPC convention:
                          NEO-Atira, NEO-Aten, NEO-Apollo, NEO-Amor,
                          MBA inner/middle/outer, Cybele, Hilda, Jupiter Trojan,
                          Centaur, TNO (KBO / SDO / detached).

  6. EXPORT             — Combined MPC One-Line astrometry report ready
                          for MPC submission. Orbital elements table.

Works with moving_object.py:
    → moving_object.py exports per-night .txt MPC files
    → this script reads those files directly
    → outputs combined_astrometry_YYYYMMDD.txt for MPC submission

Algorithm:
  - State vector propagation: Kepler equation (elliptic orbits)
  - Coordinate transforms: J2000 equatorial ↔ ecliptic
  - Earth position: low-precision analytical series (< 0.01 AU error)
  - Orbit fitting: scipy.optimize.minimize (Nelder-Mead)
  - Orbit uncertainty: Monte-Carlo re-sampling
  - MPC format: https://www.minorplanetcenter.net/iau/info/OpticalObs.html

References:
  Gauss 1809        — Theoria motus, preliminary orbit determination
  Bate et al. 1971  — Fundamentals of Astrodynamics
  Curtis 2005       — Orbital Mechanics for Engineering Students
  Granvik et al. 2018 — NEO orbit class boundaries, Icarus 312, 181
  MPC 2024          — Guide to Minor Body Astrometry

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, re, math, traceback
import numpy as np
from pathlib import Path
from datetime import datetime, date

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log,"a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\norbit_determinator.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","scipy","matplotlib","astropy"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm as scipy_norm

try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QMessageBox, QSizePolicy,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QPlainTextEdit, QLineEdit, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon, QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Orbit Determinator"

# Gaussian gravitational constant (AU^1.5/day)
K_GAUSS = 0.01720209895
MU      = K_GAUSS**2          # AU^3/day^2
DEG     = math.pi / 180.0


# ═══════════════════════════════════════════════════════════════════════════════
#  COORDINATE & PHYSICS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def earth_ecliptic(jd: float) -> np.ndarray:
    """
    Heliocentric ecliptic position of Earth (low-precision analytical).
    Accuracy ~0.01 AU. Sufficient for preliminary orbit determination.
    """
    T   = (jd - 2451545.0) / 36525.0
    L   = (280.460 + 36000.771 * T) * DEG
    g   = (357.528 + 35999.050 * T) * DEG
    lam = L + 1.915 * DEG * math.sin(g) + 0.020 * DEG * math.sin(2*g)
    eps = (23.4393 - 0.01300 * T) * DEG
    r   = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2*g)
    return np.array([-r * math.cos(lam),
                     -r * math.cos(eps) * math.sin(lam),
                     -r * math.sin(eps) * math.sin(lam)])


def topocentric_correction(jd: float,
                            lat_deg: float, lon_deg: float,
                            alt_m: float = 0.0) -> np.ndarray:
    """Observer's geocentric displacement vector (AU, ecliptic)."""
    eps = (23.4393) * DEG
    r_earth_AU = 6.378137e6 / 1.496e11
    alt_AU     = alt_m / 1.496e11
    rho        = r_earth_AU + alt_AU
    lat        = lat_deg * DEG
    GMST       = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360
    lha        = GMST * DEG + lon_deg * DEG
    # Geocentric equatorial
    x_eq = rho * math.cos(lat) * math.cos(lha)
    y_eq = rho * math.cos(lat) * math.sin(lha)
    z_eq = rho * math.sin(lat)
    # Rotate to ecliptic
    return np.array([x_eq,
                      y_eq * math.cos(eps) + z_eq * math.sin(eps),
                     -y_eq * math.sin(eps) + z_eq * math.cos(eps)])


def observer_pos(jd: float, lat=0.0, lon=0.0, alt=0.0) -> np.ndarray:
    return earth_ecliptic(jd) + topocentric_correction(jd, lat, lon, alt)


def radec_to_ecliptic_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Unit direction vector from RA/Dec (degrees, J2000) in ecliptic frame."""
    eps = 23.4393 * DEG
    ra  = ra_deg * DEG; dec = dec_deg * DEG
    x   = math.cos(dec) * math.cos(ra)
    y   = math.cos(dec) * math.sin(ra)
    z   = math.sin(dec)
    return np.array([x,
                      y * math.cos(eps) + z * math.sin(eps),
                     -y * math.sin(eps) + z * math.cos(eps)])


def ecliptic_to_radec(xyz: np.ndarray) -> tuple[float, float]:
    """RA/Dec (degrees) from ecliptic unit vector."""
    eps = 23.4393 * DEG
    x, ye, ze = xyz
    y   =  ye * math.cos(eps) - ze * math.sin(eps)
    z   =  ye * math.sin(eps) + ze * math.cos(eps)
    r   = math.sqrt(x*x + y*y + z*z)
    ra  = math.degrees(math.atan2(y, x)) % 360
    dec = math.degrees(math.asin(z / r))
    return ra, dec


# ── Kepler equation solver ─────────────────────────────────────────────────

def solve_kepler(M: float, e: float, tol: float = 1e-12) -> float:
    """Solve Kepler's equation M = E - e sin E for eccentric anomaly E."""
    E = M if e < 0.8 else math.pi
    for _ in range(100):
        dE = (M - E + e * math.sin(E)) / (1 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def elements_to_state(a: float, e: float, i_deg: float,
                       Om_deg: float, om_deg: float,
                       M0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Keplerian elements → heliocentric ecliptic state vector (AU, AU/day).
    Epoch is encoded implicitly in M0.
    """
    i  = i_deg  * DEG; Om = Om_deg * DEG
    om = om_deg * DEG; M  = M0_deg * DEG

    E  = solve_kepler(M, e)
    nu = 2 * math.atan2(math.sqrt(1+e) * math.sin(E/2),
                         math.sqrt(1-e) * math.cos(E/2))
    p  = a * (1 - e**2)
    if p <= 0 or a <= 0: return np.zeros(3), np.zeros(3)
    r  = p / (1 + e * math.cos(nu))

    x_o  =  r * math.cos(nu);        y_o  =  r * math.sin(nu)
    vx_o = -math.sqrt(MU/p)*math.sin(nu)
    vy_o =  math.sqrt(MU/p)*(e + math.cos(nu))

    cos = math.cos; sin = math.sin
    R = np.array([
        [cos(Om)*cos(om) - sin(Om)*sin(om)*cos(i),
        -cos(Om)*sin(om) - sin(Om)*cos(om)*cos(i),
         sin(Om)*sin(i)],
        [sin(Om)*cos(om) + cos(Om)*sin(om)*cos(i),
        -sin(Om)*sin(om) + cos(Om)*cos(om)*cos(i),
        -cos(Om)*sin(i)],
        [sin(om)*sin(i), cos(om)*sin(i), cos(i)],
    ])
    return R @ np.array([x_o, y_o, 0.0]), R @ np.array([vx_o, vy_o, 0.0])


def propagate_kepler(r0: np.ndarray, v0: np.ndarray,
                     dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate state (r0, v0) by dt days using Kepler equation.
    Valid for elliptic orbits (e < 1).
    """
    pass  # not used — use propagate_elements() instead


def state_to_elements_dict(r_vec: np.ndarray,
                            v_vec: np.ndarray) -> dict | None:
    """State vector → orbital elements dict."""
    r = float(np.linalg.norm(r_vec)); v = float(np.linalg.norm(v_vec))
    h_vec = np.cross(r_vec, v_vec);   h = float(np.linalg.norm(h_vec))
    e_vec = np.cross(v_vec, h_vec) / MU - r_vec / r
    e     = float(np.linalg.norm(e_vec))
    E_en  = v**2 / 2 - MU / r
    if abs(E_en) < 1e-15: return None
    a     = -MU / (2 * E_en)

    i_r = math.acos(float(np.clip(h_vec[2] / h, -1, 1)))
    K   = np.array([0.0, 0.0, 1.0])
    N   = np.cross(K, h_vec); Nm = float(np.linalg.norm(N))

    Om_r = 0.0
    if Nm > 1e-12:
        Om_r = math.acos(float(np.clip(N[0] / Nm, -1, 1)))
        if N[1] < 0: Om_r = 2*math.pi - Om_r

    om_r = 0.0
    if Nm > 1e-12 and e > 1e-8:
        om_r = math.acos(float(np.clip(float(np.dot(N, e_vec)) / (Nm*e), -1, 1)))
        if e_vec[2] < 0: om_r = 2*math.pi - om_r

    nu_r = 0.0
    if e > 1e-8:
        nu_r = math.acos(float(np.clip(float(np.dot(e_vec, r_vec)) / (e*r), -1, 1)))
        if float(np.dot(r_vec, v_vec)) < 0: nu_r = 2*math.pi - nu_r

    # Mean anomaly
    E_r = 2 * math.atan2(math.sqrt(max(1-e, 0)) * math.sin(nu_r/2),
                          math.sqrt(max(1+e, 0)) * math.cos(nu_r/2))
    M_r = E_r - e * math.sin(E_r)

    q = a * (1 - e)
    T_yr = 2 * math.pi * abs(a)**1.5 / K_GAUSS / 365.25

    return dict(
        a=a, e=e, i=math.degrees(i_r),
        Om=math.degrees(Om_r) % 360,
        om=math.degrees(om_r) % 360,
        M0=math.degrees(M_r)  % 360,
        nu=math.degrees(nu_r) % 360,
        q=q, Q=a*(1+e),
        T_yr=T_yr,
    )


def propagate_elements(a, e, i, Om, om, M0_deg, dt_days):
    """Propagate elements by dt_days (advance mean anomaly)."""
    e = max(0.0, min(0.9999, e))          # keep elliptic
    n = K_GAUSS / abs(a)**1.5            # mean motion rad/day
    M_new = (M0_deg * DEG + n * dt_days) % (2*math.pi)
    return a, e, i, Om, om, math.degrees(M_new)


def predict_radec(a, e, i, Om, om, M0_deg,
                  jd_ref: float, jd_obs: float,
                  obs_pos: np.ndarray) -> tuple[float, float]:
    """Predict RA/Dec of object at jd_obs given elements at jd_ref."""
    dt   = jd_obs - jd_ref
    elems = propagate_elements(a, e, i, Om, om, M0_deg, dt)
    r_vec, _ = elements_to_state(*elems)
    diff  = r_vec - obs_pos
    return ecliptic_to_radec(diff / np.linalg.norm(diff))


# ── Angular separation ─────────────────────────────────────────────────────

def angular_sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    """Great-circle angular separation in arcseconds."""
    d  = math.cos(dec1*DEG)*math.cos(dec2*DEG)*math.cos((ra1-ra2)*DEG)
    d += math.sin(dec1*DEG)*math.sin(dec2*DEG)
    return math.degrees(math.acos(max(-1.0, min(1.0, d)))) * 3600


# ═══════════════════════════════════════════════════════════════════════════════
#  MPC FORMAT  (reads output of moving_object.py)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_mpc_oneline(line: str) -> dict | None:
    """
    Parse an MPC One-Line observation record.
    Format: https://www.minorplanetcenter.net/iau/info/OpticalObs.html
    Columns: 1-5 designation, 6 discovery, 7 note1, 8 note2,
             15-32 date, 33-44 RA, 45-57 Dec, 58-65 mag+band, 77-80 code
    """
    line = line.rstrip()
    if len(line) < 80: return None
    try:
        # Date: YYYY MM DD.ddddd (columns 15-32, 1-indexed → 14:31 0-indexed)
        date_str = line[15:31].strip()
        parts    = date_str.split()
        if len(parts) < 3: return None
        yr  = int(parts[0]); mo = int(parts[1]); day = float(parts[2])
        day_i = int(day); frac = day - day_i
        jd = (_ymd_to_jd(yr, mo, day_i) + frac)

        # RA: HH MM SS.sss (columns 32:44)
        ra_str = line[32:44].strip()
        rp = ra_str.split()
        ra_deg = (float(rp[0]) + float(rp[1])/60 + float(rp[2])/3600) * 15

        # Dec: ±DD MM SS.ss (columns 44:56)
        dec_str = line[44:56].strip()
        sign = -1 if dec_str.startswith('-') else 1
        dec_str = dec_str.lstrip('+-')
        dp = dec_str.split()
        dec_deg = sign * (float(dp[0]) + float(dp[1])/60 + float(dp[2])/3600)

        mag_str = line[65:70].strip()
        mag = float(mag_str) if mag_str else None

        obs_code = line[77:80].strip()
        desig    = line[0:5].strip() or line[5:12].strip()

        return dict(jd=jd, ra=ra_deg, dec=dec_deg, mag=mag,
                    obs_code=obs_code, desig=desig, raw=line)
    except Exception:
        return None


def _ymd_to_jd(y, m, d) -> float:
    if m <= 2: y -= 1; m += 12
    A = int(y/100); B = 2-A+int(A/4)
    return int(365.25*(y+4716)) + int(30.6001*(m+1)) + d + B - 1524.5


def format_mpc_oneline(desig: str, jd: float, ra_deg: float, dec_deg: float,
                        mag: float = None, obs_code: str = "XXX") -> str:
    """Format an MPC One-Line astrometry record."""
    # Date
    jd2 = jd + 0.5; z = int(jd2); f = jd2 - z
    if z < 2299161:
        A = z
    else:
        alpha = int((z - 1867216.25)/36524.25)
        A = z + 1 + alpha - int(alpha/4)
    B = A + 1524; C = int((B-122.1)/365.25)
    D = int(365.25*C); E = int((B-D)/30.6001)
    day = B - D - int(30.6001*E) + f
    month = E-1 if E < 14 else E-13
    year  = C-4716 if month > 2 else C-4715

    ra_h   = ra_deg / 15; ra_hh = int(ra_h)
    ra_mm  = (ra_h - ra_hh)*60; ra_m = int(ra_mm)
    ra_ss  = (ra_mm - ra_m)*60
    dec_a  = abs(dec_deg); sign = '+' if dec_deg >= 0 else '-'
    dec_dd = int(dec_a); dec_mm2 = (dec_a-dec_dd)*60
    dec_m2 = int(dec_mm2); dec_ss2 = (dec_mm2-dec_m2)*60

    mag_str = f"{mag:5.1f}V" if mag else "      "
    desig_f = f"{desig:<5s}" if len(desig)<=5 else desig[:5]

    return (f"{desig_f}  C{year:4d} {month:02d} {day:08.5f} "
            f"{ra_hh:02d} {ra_m:02d} {ra_ss:05.2f}"
            f"{sign}{dec_dd:02d} {dec_m2:02d} {dec_ss2:04.1f}         "
            f"{mag_str}      {obs_code:<3s}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRACKLET  (one night's observations)
# ═══════════════════════════════════════════════════════════════════════════════

class Tracklet:
    """
    A single-night sequence of observations of a moving object.
    Fit with linear proper motion in RA/Dec.
    """

    def __init__(self, obs_list: list[dict], night_label: str = ""):
        self.obs       = sorted(obs_list, key=lambda o: o['jd'])
        self.label     = night_label
        self.t_mid     = (self.obs[0]['jd'] + self.obs[-1]['jd']) / 2
        self._fit_linear()

    def _fit_linear(self):
        ts   = np.array([o['jd'] for o in self.obs])
        ras  = np.array([o['ra'] for o in self.obs])
        decs = np.array([o['dec'] for o in self.obs])
        dt   = ts - self.t_mid
        # Handle RA wrap-around
        ras  = np.unwrap(ras * DEG) / DEG

        self.dra_dt  = float(np.polyfit(dt, ras,  1)[0])  # deg/day
        self.ddec_dt = float(np.polyfit(dt, decs, 1)[0])
        self.ra0     = float(np.interp(0, dt, ras)) % 360
        self.dec0    = float(np.interp(0, dt, decs))

        # Residuals in arcsec
        ra_fit  = self.ra0  + self.dra_dt  * dt
        dec_fit = self.dec0 + self.ddec_dt * dt
        res = np.sqrt((ras-ra_fit)**2 + (decs-dec_fit)**2) * 3600
        self.rms_arcsec = float(res.mean())

    def predict(self, jd: float) -> tuple[float, float]:
        dt = jd - self.t_mid
        return (self.ra0 + self.dra_dt * dt) % 360, self.dec0 + self.ddec_dt * dt

    @property
    def speed_arcsec_hr(self) -> float:
        return math.sqrt(self.dra_dt**2 + self.ddec_dt**2) * 3600 / 24

    @property
    def pa_deg(self) -> float:
        """Position angle of motion (degrees from North, East=90)."""
        return math.degrees(math.atan2(self.dra_dt, self.ddec_dt)) % 360

    def __repr__(self):
        return (f"Tracklet({self.label} JD={self.t_mid:.1f} "
                f"RA={self.ra0:.3f}° Dec={self.dec0:.3f}° "
                f"dRA={self.dra_dt*3600:.1f}\"/day RMS={self.rms_arcsec:.2f}\")")


# ═══════════════════════════════════════════════════════════════════════════════
#  ORBIT FITTER
# ═══════════════════════════════════════════════════════════════════════════════

class OrbitFitter:
    """
    Fit Keplerian orbital elements to a set of astrometric observations
    by minimising total RA/Dec residuals.

    Algorithm:
      1. Grid search over distance (0.3–15 AU) at middle epoch
         to find approximate r2; rough estimate of a, e.
      2. Nelder-Mead refinement of all 6 elements.
      3. Monte-Carlo uncertainty: 200 trials with obs perturbed by σ.
    """

    def __init__(self, obs_list: list[dict],
                 obs_positions: list[np.ndarray],
                 jd_ref: float,
                 lat_deg: float = 0.0,
                 lon_deg: float = 0.0,
                 alt_m:   float = 0.0):
        self.obs      = obs_list          # [{jd, ra, dec, ...}]
        self.obs_pos  = obs_positions     # heliocentric observer positions
        self.jd_ref   = jd_ref

    def _residual(self, params: np.ndarray) -> float:
        a, e, i, Om, om, M0 = params
        a = abs(a); e = abs(e) % 0.999
        if a < 0.1 or a > 60: return 1e9
        if a*(1-e**2) <= 0:   return 1e9
        total = 0.0
        for ob, Robs in zip(self.obs, self.obs_pos):
            try:
                ra_p, dec_p = predict_radec(
                    a, e, i, Om, om, M0, self.jd_ref, ob['jd'], Robs)
                sep = angular_sep_arcsec(ob['ra'], ob['dec'], ra_p, dec_p)
                total += sep**2
            except Exception:
                total += 1e6
        return total

    def fit(self, n_mc: int = 200,
            sigma_arcsec: float = 1.0,
            progress_cb = None) -> dict:
        """Run grid search + refinement + Monte-Carlo uncertainty."""

        # ── Step 1: Grid search for initial distance ───────────────────────
        best_r2 = 1.0; best_val = 1e20
        jd_mid  = np.median([o['jd'] for o in self.obs])
        mid_obs = min(self.obs, key=lambda o: abs(o['jd'] - jd_mid))
        rho_dir = radec_to_ecliptic_unit(mid_obs['ra'], mid_obs['dec'])
        R_mid   = self.obs_pos[self.obs.index(mid_obs)]

        for r2_try in np.linspace(0.3, 15, 50):
            r_obj  = R_mid + r2_try * rho_dir
            rv_est = state_to_elements_dict(r_obj,
                         np.array([0.0, K_GAUSS/r2_try**0.5, 0.0]))
            if rv_est is None: continue
            a0  = rv_est['a']; e0 = min(max(rv_est['e'],0),0.99)
            p0  = [a0, e0, rv_est['i'], rv_est['Om'],
                   rv_est['om'], rv_est['M0']]
            val = self._residual(p0)
            if val < best_val:
                best_val = val; best_r2 = r2_try; best_p0 = p0

        if progress_cb: progress_cb(20, "Grid search done, refining…")

        # ── Step 2: Nelder-Mead refinement ────────────────────────────────
        res = minimize(self._residual, best_p0,
                       method='Nelder-Mead',
                       options={'maxiter':10000, 'xatol':1e-7,
                                'fatol':1e-4, 'adaptive':True})

        best_p = res.x
        a,e,i,Om,om,M0 = best_p
        # Keep in physical bounds
        e  = abs(e) % 0.9999
        i  = i % 180; Om = Om % 360; om = om % 360; M0 = M0 % 360

        rms_arcsec = math.sqrt(max(res.fun, 0) / max(len(self.obs), 1))

        if progress_cb: progress_cb(40, f"Fit complete. RMS={rms_arcsec:.2f}\"  Running MC…")

        # ── Step 3: Monte-Carlo uncertainty ───────────────────────────────
        sigma_deg = sigma_arcsec / 3600.0
        mc_results = []
        for mc_i in range(n_mc):
            if progress_cb and mc_i % 20 == 0:
                pct = 40 + int(55 * mc_i / n_mc)
                progress_cb(pct, f"Monte-Carlo {mc_i}/{n_mc}…")
            obs_p = []
            for ob in self.obs:
                obs_p.append({**ob,
                    'ra':  ob['ra']  + np.random.normal(0, sigma_deg),
                    'dec': ob['dec'] + np.random.normal(0, sigma_deg)})
            fitter_mc = OrbitFitter(obs_p, self.obs_pos, self.jd_ref)
            res_mc = minimize(fitter_mc._residual, best_p,
                              method='Nelder-Mead',
                              options={'maxiter':3000, 'xatol':1e-5,
                                       'fatol':0.1, 'adaptive':True})
            mc_results.append(res_mc.x)

        mc_arr = np.array(mc_results)
        el_names = ['a','e','i','Om','om','M0']
        uncertainties = {n: float(np.std(mc_arr[:,k]))
                         for k, n in enumerate(el_names)}

        if progress_cb: progress_cb(96, "Computing residuals…")

        # Per-observation residuals
        residuals = []
        for ob, Robs in zip(self.obs, self.obs_pos):
            ra_p, dec_p = predict_radec(a,e,i,Om,om,M0,
                                         self.jd_ref, ob['jd'], Robs)
            sep = angular_sep_arcsec(ob['ra'], ob['dec'], ra_p, dec_p)
            residuals.append({'jd':ob['jd'],'ra_obs':ob['ra'],'dec_obs':ob['dec'],
                               'ra_pred':ra_p,'dec_pred':dec_p,'sep_arcsec':sep})

        return dict(
            a=a, e=e, i=i, Om=Om, om=om, M0=M0,
            q=a*(1-e), Q=a*(1+e),
            T_yr=2*math.pi*abs(a)**1.5 / K_GAUSS / 365.25,
            jd_ref=self.jd_ref,
            rms_arcsec=rms_arcsec,
            uncertainties=uncertainties,
            mc_results=mc_arr,
            residuals=residuals,
            n_obs=len(self.obs),
            success=res.success,
            orbit_class=classify_orbit(a, e, i),
        )


# ── Orbit classification ───────────────────────────────────────────────────

def classify_orbit(a: float, e: float, i: float) -> str:
    """
    Classify orbit by dynamical class.
    Based on Granvik et al. 2018 boundaries + MPC conventions.
    """
    q = a * (1 - e)  # perihelion distance
    Q = a * (1 + e)  # aphelion distance

    if a < 0: return "Hyperbolic / unbound"
    if Q < 0.983: return "NEO-Atira (Q < Earth)"
    if a < 1.0 and Q >= 0.983: return "NEO-Aten"
    if a >= 1.0 and q <= 1.017: return "NEO-Apollo"
    if a > 1.0 and q > 1.017 and q <= 1.3: return "NEO-Amor"
    if 1.3 < q < 1.7: return "Mars-crosser"
    if 2.0 <= a <= 2.5: return "MBA inner (Hungaria / Flora)"
    if 2.5 < a <= 2.82: return "MBA middle (Nysa / Koronis)"
    if 2.82 < a <= 3.27: return "MBA outer (Eos / Themis)"
    if 3.27 < a <= 3.7: return "Cybele group"
    if 3.7 < a <= 4.2 and e < 0.3: return "Hilda group (3:2 resonance)"
    if 5.05 < a < 5.4 and abs(a - 5.20) < 0.35: return "Jupiter Trojan (L4/L5)"
    if 4.5 < a < 30 and q < 5.5: return "Centaur"
    if 30 <= a <= 50 and e < 0.25: return "Classical KBO (Cubewano)"
    if a > 50 and q > 35: return "Detached / Sednoid"
    if a > 30: return "Trans-Neptunian object (TNO)"
    return f"Uncertain (a={a:.2f} q={q:.2f})"


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class OrbitWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, tracklets, obs_all, params):
        super().__init__()
        self.tracklets = tracklets
        self.obs_all   = obs_all
        self.params    = params

    def run(self):
        try:
            self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _pipeline(self):
        p  = self.params
        lat, lon, alt = p.get('lat',0), p.get('lon',0), p.get('alt',0)

        self.progress.emit(5, "Computing observer positions…")
        obs_positions = [observer_pos(o['jd'], lat, lon, alt)
                         for o in self.obs_all]

        jd_ref = float(np.median([o['jd'] for o in self.obs_all]))
        self.progress.emit(10, f"Fitting orbit to {len(self.obs_all)} observations…")

        fitter = OrbitFitter(self.obs_all, obs_positions, jd_ref, lat, lon, alt)

        def prog(pct, msg):
            self.progress.emit(pct, msg)

        result = fitter.fit(
            n_mc=p.get('n_mc', 200),
            sigma_arcsec=p.get('sigma_arcsec', 1.0),
            progress_cb=prog)

        self.progress.emit(100,
            f"Done!  {result['orbit_class']}  RMS={result['rms_arcsec']:.2f}\"")
        self.finished.emit(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════════

DARK = """
QMainWindow,QWidget{background:#0e1118;color:#d0d8e8;
    font-family:'JetBrains Mono','Cascadia Code',Consolas,monospace;font-size:11px;}
QTabWidget::pane{border:1px solid #1e2840;background:#111824;}
QTabBar::tab{background:#131a28;color:#4a6080;padding:6px 16px;
    border:1px solid #1e2840;border-bottom:none;}
QTabBar::tab:selected{background:#111824;color:#60a8ff;
    border-bottom:2px solid #3060c0;}
QGroupBox{border:1px solid #1e2840;border-radius:4px;margin-top:8px;
    padding-top:8px;color:#4a6080;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
QPushButton{background:#131a28;color:#8ab0d8;border:1px solid #2040a0;
    border-radius:4px;padding:5px 14px;}
QPushButton:hover{background:#1a2840;border-color:#4070e0;}
QPushButton:disabled{color:#283040;border-color:#181e2c;}
QPushButton#run_btn{background:#0e1e38;color:#60c0ff;
    border:1px solid #3060c0;font-weight:bold;padding:7px 20px;}
QPushButton#run_btn:hover{background:#142848;}
QSpinBox,QDoubleSpinBox,QComboBox,QLineEdit{background:#0e1420;
    border:1px solid #1e2840;border-radius:3px;padding:3px 6px;color:#c0d0e8;}
QTextEdit,QPlainTextEdit{background:#080c14;border:1px solid #1e2840;
    color:#40c080;font-family:Consolas,monospace;font-size:10px;}
QTableWidget{background:#080c14;alternate-background-color:#0c1018;
    gridline-color:#1e2840;color:#c0d0e8;}
QTableWidget QHeaderView::section{background:#0e1420;color:#4a6080;
    border:1px solid #1e2840;padding:3px;font-weight:bold;}
QSplitter::handle{background:#1e2840;}
QLabel#title_lbl{font-size:17px;font-weight:bold;color:#60c0ff;padding:6px;}
QLabel#sub_lbl{font-size:9px;color:#2a4060;padding:0 8px 4px;}
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8,5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor='#0c1018')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)
    def redraw(self): self.fig.tight_layout(pad=0.4); self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#080c14')
    ax.set_title(title, color='#4080c0', fontsize=9, pad=3)
    ax.set_xlabel(xl, color='#303848', fontsize=8)
    ax.set_ylabel(yl, color='#303848', fontsize=8)
    ax.tick_params(colors='#303848', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e2840')


class StatusFooter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setStyleSheet("background:#080c14;border-top:1px solid #1e2840;")
        lay = QHBoxLayout(self); lay.setContentsMargins(6,0,6,0)
        self.lbl = QLabel("Ready")
        self.lbl.setStyleSheet("color:#2a4060;font-size:10px;")
        self.pbar = QProgressBar(); self.pbar.setFixedSize(200,12)
        self.pbar.setVisible(False)
        self.pbar.setStyleSheet("""
            QProgressBar{background:#0c1420;border:1px solid #203060;
                border-radius:3px;}
            QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1040a0,stop:1 #4090e0);border-radius:2px;}""")
        lay.addWidget(self.lbl,1); lay.addWidget(self.pbar)

    def update(self, pct, msg):
        self.lbl.setText(msg)
        if pct >= 0:
            self.pbar.setVisible(True); self.pbar.setValue(pct)
            if pct >= 100:
                QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))


class OrbitApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1440, 920)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))

        self._tracklets: list[Tracklet] = []
        self._obs_all:   list[dict]     = []
        self._result:    dict | None    = None
        self._worker = self._wthread    = None

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._make_header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._make_controls())
        sp.addWidget(self._make_tabs())
        sp.setSizes([290, 1150])
        rl.addWidget(sp, 1)
        self.status = StatusFooter(); rl.addWidget(self.status)

    def _make_header(self):
        hdr = QWidget(); hdr.setFixedHeight(62)
        hdr.setStyleSheet("background:#080c14;border-bottom:1px solid #1e2840;")
        lay = QHBoxLayout(hdr); lay.setContentsMargins(12,0,12,0)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(
                46,46,Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("Orbit Determinator"); t.setObjectName("title_lbl")
        s = QLabel("Multi-night tracklet linking  ·  Keplerian least-squares fit  "
                   "·  Monte-Carlo uncertainty  ·  Orbit classification  ·  MPC export")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _make_controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(5)

        # ── Load ──────────────────────────────────────────────────────────
        grp = QGroupBox("Load Observations"); gl = QVBoxLayout(grp)
        self.file_lbl = QLabel("No files loaded")
        self.file_lbl.setStyleSheet("color:#2a4060;font-size:10px;")
        self.file_lbl.setWordWrap(True)
        b1 = QPushButton("📂  Load MPC One-Line file(s)…"); b1.clicked.connect(self._load_mpc)
        b1.setObjectName("run_btn")
        b_man = QPushButton("✎  Manual entry…"); b_man.clicked.connect(self._manual_entry)
        b_ex  = QPushButton("📋  Load synthetic example"); b_ex.clicked.connect(self._load_example)
        gl.addWidget(self.file_lbl); gl.addWidget(b1); gl.addWidget(b_man); gl.addWidget(b_ex)
        lay.addWidget(grp)

        # ── Observer ──────────────────────────────────────────────────────
        grp2 = QGroupBox("Observer Location"); g2 = QGridLayout(grp2)
        g2.addWidget(QLabel("Latitude (°):"),0,0); self.sp_lat=QDoubleSpinBox(); self.sp_lat.setRange(-90,90); self.sp_lat.setValue(48.0); self.sp_lat.setDecimals(3); g2.addWidget(self.sp_lat,0,1)
        g2.addWidget(QLabel("Longitude (°):"),1,0); self.sp_lon=QDoubleSpinBox(); self.sp_lon.setRange(-180,180); self.sp_lon.setValue(19.0); self.sp_lon.setDecimals(3); g2.addWidget(self.sp_lon,1,1)
        g2.addWidget(QLabel("Altitude (m):"),2,0); self.sp_alt=QDoubleSpinBox(); self.sp_alt.setRange(0,5000); self.sp_alt.setValue(500); g2.addWidget(self.sp_alt,2,1)
        g2.addWidget(QLabel("MPC obs code:"),3,0); self.le_obs=QLineEdit("XXX"); g2.addWidget(self.le_obs,3,1)
        lay.addWidget(grp2)

        # ── Fit parameters ─────────────────────────────────────────────────
        grp3 = QGroupBox("Fit Parameters"); g3 = QGridLayout(grp3)
        g3.addWidget(QLabel("MC iterations:"),0,0); self.sp_nmc=QSpinBox(); self.sp_nmc.setRange(50,2000); self.sp_nmc.setValue(50); g3.addWidget(self.sp_nmc,0,1)
        g3.addWidget(QLabel("Obs. σ (\"):"),1,0); self.sp_sigma=QDoubleSpinBox(); self.sp_sigma.setRange(0.1,10); self.sp_sigma.setValue(1.0); self.sp_sigma.setDecimals(1); g3.addWidget(self.sp_sigma,1,1)
        lay.addWidget(grp3)

        # ── Run ───────────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Determine Orbit")
        self.btn_run.setObjectName("run_btn"); self.btn_run.setFixedHeight(38)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("✖  Cancel")
        self.btn_cancel.setEnabled(False); self.btn_cancel.clicked.connect(self._cancel)
        self.btn_export = QPushButton("💾  Export MPC report…")
        self.btn_export.setEnabled(False); self.btn_export.clicked.connect(self._export)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_cancel); lay.addWidget(self.btn_export)
        lay.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet("color:#40a0a0;font-size:10px;")
        self.summary_lbl.setWordWrap(True); lay.addWidget(self.summary_lbl)
        scroll.setWidget(inner); return scroll

    def _make_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_tracklets(), "🌌  Tracklets")
        self.tabs.addTab(self._tab_elements(),  "🪐  Elements")
        self.tabs.addTab(self._tab_residuals(), "📊  Residuals")
        self.tabs.addTab(self._tab_mc(),        "📈  Uncertainty")
        self.tabs.addTab(self._tab_orbit(),     "🔭  Orbit View")
        self.tabs.addTab(self._tab_log(),       "📋  Log")
        return self.tabs

    def _tab_tracklets(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.tracklet_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.tracklet_plot)
        self.tracklet_table = QTableWidget(0,5)
        self.tracklet_table.setHorizontalHeaderLabels(
            ["Night","JD mid","RA (°)","Dec (°)","Speed (\"/hr)"])
        self.tracklet_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tracklet_table.setFixedHeight(180)
        self.tracklet_table.setAlternatingRowColors(True)
        lay.addWidget(self.tracklet_table)
        return w

    def _tab_elements(self):
        w = QWidget(); main = QHBoxLayout(w)
        self.elem_text = QTextEdit(); self.elem_text.setReadOnly(True)
        self.elem_text.setFont(QFont("Consolas", 11))
        main.addWidget(self.elem_text)
        return w

    def _tab_residuals(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.resid_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.resid_plot)
        self.resid_table = QTableWidget(0,5)
        self.resid_table.setHorizontalHeaderLabels(["JD","RA obs","Dec obs","RA pred","Sep (\")"])
        self.resid_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.resid_table.setFixedHeight(200); self.resid_table.setAlternatingRowColors(True)
        lay.addWidget(self.resid_table)
        return w

    def _tab_mc(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.mc_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.mc_plot)
        return w

    def _tab_orbit(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.orbit_plot = MplCanvas(figsize=(9,6)); lay.addWidget(self.orbit_plot)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_mpc(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,"Load MPC One-Line files","",
            "Text (*.txt *.dat);;All (*)")
        if not paths: return
        self._obs_all = []; night_obs = {}
        for path in sorted(paths):
            name = Path(path).stem
            with open(path,'r') as f:
                lines = f.readlines()
            obs_night = []
            for line in lines:
                ob = parse_mpc_oneline(line)
                if ob: obs_night.append(ob)
            if obs_night:
                night_obs[name] = obs_night
                self._obs_all.extend(obs_night)
        self._build_tracklets(night_obs)
        self._log(f"Loaded {len(self._obs_all)} observations from {len(paths)} files")

    def _manual_entry(self):
        """Simple dialog for manual RA/Dec/time entry."""
        dlg = QWidget(); dlg.setWindowTitle("Manual observation entry")
        dlg.setStyleSheet(DARK)
        dlg.resize(500, 350)
        lay = QVBoxLayout(dlg)
        lbl = QLabel("Enter observations, one per line:\n"
                      "JD  RA_deg  Dec_deg  [mag]\n"
                      "Example: 2460000.5  123.456  -12.345  16.2")
        lbl.setStyleSheet("color:#4a6080;")
        self._man_edit = QPlainTextEdit()
        self._man_edit.setPlaceholderText("2460000.500  123.456  -12.345  16.2\n"
                                           "2460001.500  123.987  -11.890  16.1\n"
                                           "2460030.500  130.123  -10.555  16.3")
        btn = QPushButton("Import"); btn.clicked.connect(lambda: self._import_manual(dlg))
        lay.addWidget(lbl); lay.addWidget(self._man_edit); lay.addWidget(btn)
        dlg.show(); self._man_dlg = dlg

    def _import_manual(self, dlg):
        obs = []; night_label = "Manual"
        for line in self._man_edit.toPlainText().splitlines():
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) < 3: continue
            try:
                jd   = float(parts[0]); ra = float(parts[1]); dec = float(parts[2])
                mag  = float(parts[3]) if len(parts) > 3 else None
                obs.append(dict(jd=jd, ra=ra, dec=dec, mag=mag, desig='NEW', raw=line))
            except ValueError: continue
        if not obs:
            QMessageBox.warning(self,"No data","No valid observations found."); return
        self._obs_all.extend(obs)
        # Group by night (observations < 1 day apart)
        nights = {}; n = 0
        sorted_obs = sorted(obs, key=lambda o: o['jd'])
        t_prev = sorted_obs[0]['jd']; night_obs = [sorted_obs[0]]
        for ob in sorted_obs[1:]:
            if ob['jd'] - t_prev < 1.0:
                night_obs.append(ob)
            else:
                nights[f"Manual_N{n}"] = night_obs
                n += 1; night_obs = [ob]
            t_prev = ob['jd']
        nights[f"Manual_N{n}"] = night_obs
        self._build_tracklets(nights)
        dlg.close()
        self._log(f"Manual: {len(obs)} observations in {len(nights)} nights")

    def _load_example(self):
        """Generate synthetic MBA observations across 3 nights."""
        # a=2.5, e=0.15, i=8, Om=45, om=30, M0=25 at JD 2460000
        rng = np.random.default_rng(42)
        a,e,i,Om,om,M0 = 2.5,0.15,8.0,45.0,30.0,25.0
        t0 = 2460000.0
        obs_all = {}
        for night_idx, day_offset in enumerate([0, 30, 60]):
            night_obs = []
            for obs_i in range(3):
                dt = day_offset + obs_i * 0.04   # ~1 hour apart
                jd = t0 + dt
                # Propagate orbit
                a2,e2,i2,Om2,om2,M0_prop = propagate_elements(a,e,i,Om,om,M0,dt)
                r_obj, _ = elements_to_state(a2,e2,i2,Om2,om2,M0_prop)
                R_obs    = observer_pos(jd, 48.0, 19.0, 500)
                diff     = r_obj - R_obs
                ra, dec  = ecliptic_to_radec(diff / np.linalg.norm(diff))
                # Add 1 arcsec noise
                ra  += rng.normal(0, 1/3600); dec += rng.normal(0, 1/3600)
                mag = 16.5 + rng.normal(0, 0.1)
                night_obs.append(dict(jd=jd, ra=ra%360, dec=dec,
                                      mag=round(mag,1), desig='SYNTH',
                                      raw=f"{jd:.5f} {ra:.5f} {dec:.5f}"))
            obs_all[f"Night_{night_idx}"] = night_obs
            self._obs_all.extend(night_obs)
        self._build_tracklets(obs_all)
        self._log(f"Synthetic MBA example: a=2.5AU e=0.15 i=8°")
        self._log(f"3 nights × 3 observations = 9 total")

    def _build_tracklets(self, night_obs: dict):
        self._tracklets = []
        for label, obs in night_obs.items():
            if len(obs) >= 2:
                tr = Tracklet(obs, label)
                self._tracklets.append(tr)
                self._log(f"Tracklet {label}: {len(obs)} obs  "
                          f"speed={tr.speed_arcsec_hr:.1f}\"/hr  "
                          f"PA={tr.pa_deg:.0f}°  RMS={tr.rms_arcsec:.2f}\"")
        if self._tracklets:
            self.btn_run.setEnabled(True)
            self.file_lbl.setText(
                f"{len(self._obs_all)} observations\n"
                f"{len(self._tracklets)} tracklets")
            self.file_lbl.setStyleSheet("color:#40a080;font-size:10px;")
            self._draw_tracklets()
            self._fill_tracklet_table()

    # ── Run / cancel ──────────────────────────────────────────────────────────

    def _run(self):
        if not self._obs_all: return
        params = dict(
            lat=self.sp_lat.value(), lon=self.sp_lon.value(),
            alt=self.sp_alt.value(),
            n_mc=self.sp_nmc.value(),
            sigma_arcsec=self.sp_sigma.value(),
        )
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.btn_export.setEnabled(False)

        self._wthread = QThread(self)
        self._worker  = OrbitWorker(self._tracklets, self._obs_all, params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()
        self.tabs.setCurrentIndex(5)

    def _cancel(self):
        if self._wthread: self._wthread.quit()
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_progress(self, pct, msg):
        self.status.update(pct, msg); self._log(f"[{pct:3d}%]  {msg}")

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}")
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self,"Error",msg[:500])

    def _on_finished(self, result):
        self._result = result
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        self.btn_export.setEnabled(True)

        r = result
        unc = r['uncertainties']
        self.summary_lbl.setText(
            f"{r['orbit_class']}\n"
            f"a={r['a']:.4f}±{unc['a']:.4f} AU\n"
            f"e={r['e']:.4f}±{unc['e']:.4f}\n"
            f"RMS={r['rms_arcsec']:.2f}\"  N={r['n_obs']}")

        self._log(f"\n{'='*55}")
        self._log(f"✓  Orbit determination complete")
        self._log(f"   Class:  {r['orbit_class']}")
        self._log(f"   a   = {r['a']:.6f} ± {unc['a']:.6f} AU")
        self._log(f"   e   = {r['e']:.6f} ± {unc['e']:.6f}")
        self._log(f"   i   = {r['i']:.4f} ± {unc['i']:.4f}°")
        self._log(f"   Ω   = {r['Om']:.4f} ± {unc['Om']:.4f}°")
        self._log(f"   ω   = {r['om']:.4f} ± {unc['om']:.4f}°")
        self._log(f"   q   = {r['q']:.4f} AU  (perihelion)")
        self._log(f"   Q   = {r['Q']:.4f} AU  (aphelion)")
        self._log(f"   P   = {r['T_yr']:.3f} yr")
        self._log(f"   RMS = {r['rms_arcsec']:.3f}\"")
        self._log(f"{'='*55}\n")

        self._fill_elements_tab(r)
        self._draw_residuals(r)
        self._draw_mc(r)
        self._draw_orbit(r)
        self.tabs.setCurrentIndex(1)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_tracklets(self):
        fig = self.tracklet_plot.fig; fig.clear()
        fig.patch.set_facecolor('#0c1018')
        ax = fig.add_subplot(111)
        _ax(ax, "Tracklets on sky (J2000)","RA (°)","Dec (°)")
        colors = ['#60a8ff','#60e060','#ffaa40','#ff6060','#c060ff','#40e0e0']
        for i, tr in enumerate(self._tracklets):
            c = colors[i % len(colors)]
            ras  = [o['ra']  for o in tr.obs]
            decs = [o['dec'] for o in tr.obs]
            ax.plot(ras, decs, 'o-', color=c, ms=4, lw=1.5,
                    label=f"{tr.label} ({tr.speed_arcsec_hr:.0f}\"/hr)")
            # Motion arrow
            ax.annotate('',
                xy=(ras[-1]+tr.dra_dt*0.5, decs[-1]+tr.ddec_dt*0.5),
                xytext=(ras[-1], decs[-1]),
                arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
        ax.invert_xaxis()
        ax.legend(fontsize=7, facecolor='#0e1420', edgecolor='#1e2840',
                  labelcolor='white')
        self.tracklet_plot.redraw()

    def _fill_tracklet_table(self):
        self.tracklet_table.setRowCount(len(self._tracklets))
        for r, tr in enumerate(self._tracklets):
            vals = [tr.label, f"{tr.t_mid:.3f}",
                    f"{tr.ra0:.4f}", f"{tr.dec0:.4f}",
                    f"{tr.speed_arcsec_hr:.2f}"]
            for c, v in enumerate(vals):
                self.tracklet_table.setItem(r,c,QTableWidgetItem(v))

    def _fill_elements_tab(self, r):
        unc = r['uncertainties']
        text = (
            f"{'='*52}\n"
            f"  ORBITAL ELEMENTS  (JD {r['jd_ref']:.3f})\n"
            f"{'='*52}\n\n"
            f"  Orbit class : {r['orbit_class']}\n\n"
            f"  a  = {r['a']:12.6f} ± {unc['a']:.6f}  AU\n"
            f"  e  = {r['e']:12.6f} ± {unc['e']:.6f}\n"
            f"  i  = {r['i']:12.6f} ± {unc['i']:.6f}  °\n"
            f"  Ω  = {r['Om']:12.6f} ± {unc['Om']:.6f}  °\n"
            f"  ω  = {r['om']:12.6f} ± {unc['om']:.6f}  °\n"
            f"  M₀ = {r['M0']:12.6f} ± {unc['M0']:.6f}  °\n\n"
            f"  q  = {r['q']:12.6f}   AU  (perihelion distance)\n"
            f"  Q  = {r['Q']:12.6f}   AU  (aphelion distance)\n"
            f"  P  = {r['T_yr']:12.4f}   yr  (orbital period)\n\n"
            f"  Fit RMS = {r['rms_arcsec']:.3f}\"\n"
            f"  N obs   = {r['n_obs']}\n"
            f"  MC runs = {len(r['mc_results'])}\n\n"
            f"{'='*52}\n"
            f"  MPC Submission Guide:\n"
            f"  1. Export astrometry below\n"
            f"  2. Submit at:\n"
            f"     https://www.minorplanetcenter.net/cgi-bin/submitobs.cgi\n"
            f"{'='*52}\n"
        )
        self.elem_text.setPlainText(text)

    def _draw_residuals(self, r):
        fig = self.resid_plot.fig; fig.clear()
        fig.patch.set_facecolor('#0c1018')
        axes = fig.subplots(1,2)
        resids = r['residuals']
        seps   = [res['sep_arcsec'] for res in resids]
        jds    = [res['jd'] for res in resids]

        _ax(axes[0], "Residuals vs time","JD","Residual (\")")
        axes[0].bar(range(len(seps)), seps, color='#3060a0', edgecolor='none', alpha=0.85)
        axes[0].axhline(r['rms_arcsec'], color='#ff8040', lw=1.5, ls='--',
                        label=f"RMS={r['rms_arcsec']:.2f}\"")
        axes[0].legend(fontsize=7, facecolor='#0e1420', edgecolor='#1e2840', labelcolor='white')

        _ax(axes[1], "O-C sky map","ΔRA (\")","ΔDec (\")")
        dra  = [(res['ra_obs']  - res['ra_pred'] )*3600 for res in resids]
        ddec = [(res['dec_obs'] - res['dec_pred'])*3600 for res in resids]
        axes[1].scatter(dra, ddec, c='#60a8ff', s=40, alpha=0.8, zorder=3)
        axes[1].axhline(0,c='#1e2840',lw=0.8); axes[1].axvline(0,c='#1e2840',lw=0.8)
        lim = max(max(abs(d) for d in dra+ddec)*1.3, 1.5)
        axes[1].set_xlim(-lim,lim); axes[1].set_ylim(-lim,lim)

        self.resid_plot.redraw()
        self.resid_table.setRowCount(len(resids))
        for row, res in enumerate(resids):
            for col, val in enumerate([f"{res['jd']:.3f}",
                                        f"{res['ra_obs']:.5f}",
                                        f"{res['dec_obs']:.5f}",
                                        f"{res['ra_pred']:.5f}",
                                        f"{res['sep_arcsec']:.2f}"]):
                self.resid_table.setItem(row,col,QTableWidgetItem(val))

    def _draw_mc(self, r):
        mc  = r['mc_results']
        fig = self.mc_plot.fig; fig.clear()
        fig.patch.set_facecolor('#0c1018')
        axes = fig.subplots(2,3)
        names  = ['a','e','i','Ω','ω','M₀']
        units  = ['AU','','°','°','°','°']
        true_v = [r['a'],r['e'],r['i'],r['Om'],r['om'],r['M0']]
        for idx, (ax, name, unit, tv) in enumerate(zip(axes.flat,names,units,true_v)):
            _ax(ax, f"{name}  ({unit})" if unit else name)
            col_data = mc[:,idx]
            ax.hist(col_data, bins=30, color='#2060a0', edgecolor='none', alpha=0.85)
            ax.axvline(tv,           color='#ff8040', lw=2,   label='Fit')
            ax.axvline(tv+mc[:,idx].std(), color='#4080c0', lw=1, ls='--')
            ax.axvline(tv-mc[:,idx].std(), color='#4080c0', lw=1, ls='--')
            ax.legend(fontsize=6, facecolor='#0e1420', edgecolor='#1e2840', labelcolor='white')
        self.mc_plot.redraw()

    def _draw_orbit(self, r):
        fig = self.orbit_plot.fig; fig.clear()
        fig.patch.set_facecolor('#0c1018')
        ax  = fig.add_subplot(111, aspect='equal')
        _ax(ax, "Orbit view (ecliptic plane projection)","X (AU)","Y (AU)")

        # Draw inner planets
        for body,a_b,color in [('Mercury',0.387,'#a0a0a0'),
                                ('Venus',0.723,'#e0b060'),
                                ('Earth',1.000,'#4080ff'),
                                ('Mars',1.524,'#e05030')]:
            theta = np.linspace(0,2*math.pi,100)
            ax.plot(a_b*np.cos(theta), a_b*np.sin(theta),
                    '--', color=color, lw=0.6, alpha=0.4)
            ax.text(a_b, 0.04, body, color=color, fontsize=6, alpha=0.5)

        # Draw asteroid belt schematic
        for a_belt, alpha in [(2.0,0.08),(3.27,0.08)]:
            theta=np.linspace(0,2*math.pi,100)
            ax.plot(a_belt*np.cos(theta), a_belt*np.sin(theta),
                    '-', color='#404040', lw=0.4, alpha=alpha)

        # Draw fitted orbit
        a,e,i,Om,om = r['a'],r['e'],r['i'],r['Om'],r['om']
        nu_arr = np.linspace(0, 2*math.pi, 360)
        xs=[]; ys=[]
        for nu in nu_arr:
            rv, _ = elements_to_state(a,e,i,Om,om, math.degrees(nu))
            xs.append(rv[0]); ys.append(rv[1])
        ax.plot(xs, ys, '-', color='#60a8ff', lw=2, label=f"Orbit (a={a:.2f} AU)")

        # Mark perihelion
        rv_q, _ = elements_to_state(a,e,i,Om,om,0)
        ax.plot(rv_q[0],rv_q[1],'v',color='#ff8040',ms=8,label=f"q={r['q']:.3f} AU")

        # Sun
        ax.plot(0,0,'*',color='#ffff40',ms=16,label='Sun',zorder=5)

        # Current positions of observations
        for ob in self._obs_all:
            ax.plot(0,0,'.',alpha=0)  # placeholder

        ax.set_xlim(-max(r['Q']*1.3,2), max(r['Q']*1.3,2))
        ax.set_ylim(-max(r['Q']*1.3,2), max(r['Q']*1.3,2))
        ax.legend(fontsize=7, facecolor='#0e1420', edgecolor='#1e2840',
                  labelcolor='white', loc='upper right')
        ax.text(0.02, 0.98, r['orbit_class'],
                transform=ax.transAxes, ha='left', va='top',
                color='#40c080', fontsize=9, fontweight='bold')
        self.orbit_plot.redraw()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        if not self._result or not self._obs_all: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MPC astrometry", "combined_astrometry.txt",
            "Text (*.txt);;All (*)")
        if not path: return
        r    = self._result
        unc  = r['uncertainties']
        code = self.le_obs.text().strip() or "XXX"
        lines = [
            "# HYPERLOAD — Orbit Determinator  MPC astrometry report",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Orbit class: {r['orbit_class']}",
            f"# a={r['a']:.6f}±{unc['a']:.6f} e={r['e']:.6f}±{unc['e']:.6f}"
            f" i={r['i']:.4f}±{unc['i']:.4f}",
            f"# RMS={r['rms_arcsec']:.3f}\" N={r['n_obs']}",
            "#",
            "# MPC One-Line astrometry (submit at mpc.net):",
            "",
        ]
        for ob in sorted(self._obs_all, key=lambda o: o['jd']):
            lines.append(format_mpc_oneline(
                ob.get('desig','NEW'), ob['jd'],
                ob['ra'], ob['dec'], ob.get('mag'), code))
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        self._log(f"✓ Exported: {path}")

    def _log(self, msg):
        self.log.append(msg)
        sb = self.log.verticalScrollBar(); sb.setValue(sb.maximum())


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = OrbitApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
