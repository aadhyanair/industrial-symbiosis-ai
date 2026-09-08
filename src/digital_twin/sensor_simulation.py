"""
NIR / moisture / chemical multi-sensor simulation for recovered-material streams.

The simulator produces the kind of signal a real sorting line would see:

    * a near-infra-red reflectance spectrum (1000-2500 nm) built from
      material-characteristic overtone / combination absorption bands,
    * a moisture reading that also injects O-H water bands into the spectrum,
    * a small bank of scalar "chemical" sensors (density, pH, conductivity,
      VOC, chlorine index, ...) with material-dependent baselines.

Contamination is modelled explicitly: a sample of nominal purity ``p`` is a
convex blend of the target material with a random contaminant material, plus
extra broadband noise that scales with ``(1 - p)``.  This gives a continuous,
physically-motivated purity label for the regression head.

Everything is deterministic given a ``numpy`` ``Generator`` / seed so datasets
are reproducible.
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------
# Spectral grid
# ------------------------------------------------------------------

WL_MIN_NM = 1000.0
WL_MAX_NM = 2500.0
N_WAVELENGTHS = 256

WAVELENGTHS = np.linspace(WL_MIN_NM, WL_MAX_NM, N_WAVELENGTHS)

# ------------------------------------------------------------------
# Scalar chemical-sensor channels (order matters - used everywhere)
# ------------------------------------------------------------------

SENSOR_CHANNELS = [
    "moisture_pct",       # %  water by weight
    "density_kg_m3",      # bulk density
    "ph",                 # aqueous extract pH
    "conductivity_ms_cm", # electrical conductivity of extract
    "voc_ppm",            # head-space volatile organics
    "chlorine_index",     # surrogate XRF/spectral Cl response (PVC marker)
    "ir_reflectance",     # broadband IR reflectance (metal / glass marker)
    "bulk_temp_c",        # stream temperature
]
N_SENSORS = len(SENSOR_CHANNELS)

# ------------------------------------------------------------------
# Material library
# ------------------------------------------------------------------
# For every material we store:
#   bands      : list of (centre_nm, width_nm, amplitude) NIR absorption features
#   baseline   : (offset, slope_per_1000nm, curvature) scatter baseline
#   moisture   : (mean_pct, sd_pct) intrinsic moisture
#   density    : (mean, sd) kg/m3
#   ph         : (mean, sd)
#   conductivity : (mean, sd) mS/cm
#   voc        : (mean, sd) ppm
#   chlorine   : (mean, sd) unitless index 0-1
#   ir_reflect : (mean, sd) 0-1
# Values are illustrative but chosen to be internally separable and to line up
# with well-known NIR marker bands (C-H ~1200/1700/2300 nm, O-H ~1450/1940 nm,
# aromatic C-H ~1680 nm, C-Cl influence near 2200-2500 nm, etc.).

MATERIALS: dict[str, dict] = {
    "PET": dict(
        bands=[(1130, 45, 0.30), (1410, 40, 0.35), (1660, 35, 0.55),
               (1900, 55, 0.40), (2130, 45, 0.60), (2300, 50, 0.45)],
        baseline=(0.35, 0.05, -0.02), moisture=(0.4, 0.3),
        density=(1350, 40), ph=(6.8, 0.3), conductivity=(0.05, 0.03),
        voc=(15, 8), chlorine=(0.03, 0.02), ir_reflect=(0.30, 0.05),
    ),
    "HDPE": dict(
        bands=[(1210, 45, 0.70), (1400, 40, 0.45), (1730, 40, 0.85),
               (2310, 45, 0.75), (2350, 45, 0.60)],
        baseline=(0.55, 0.02, -0.01), moisture=(0.2, 0.2),
        density=(950, 25), ph=(7.0, 0.2), conductivity=(0.02, 0.02),
        voc=(8, 5), chlorine=(0.02, 0.02), ir_reflect=(0.45, 0.05),
    ),
    "LDPE": dict(
        bands=[(1215, 48, 0.65), (1395, 42, 0.42), (1725, 42, 0.80),
               (2305, 48, 0.70), (2345, 48, 0.55)],
        baseline=(0.52, 0.02, -0.01), moisture=(0.2, 0.2),
        density=(920, 20), ph=(7.0, 0.2), conductivity=(0.02, 0.02),
        voc=(9, 5), chlorine=(0.02, 0.02), ir_reflect=(0.44, 0.05),
    ),
    "PP": dict(
        bands=[(1195, 42, 0.60), (1360, 35, 0.40), (1690, 35, 0.70),
               (2270, 42, 0.65), (2340, 45, 0.55)],
        baseline=(0.50, 0.03, -0.01), moisture=(0.2, 0.2),
        density=(905, 20), ph=(7.0, 0.2), conductivity=(0.02, 0.02),
        voc=(10, 6), chlorine=(0.02, 0.02), ir_reflect=(0.43, 0.05),
    ),
    "PS": dict(
        bands=[(1145, 35, 0.45), (1680, 30, 0.90), (2140, 40, 0.55),
               (2350, 45, 0.50)],
        baseline=(0.42, 0.04, -0.02), moisture=(0.3, 0.2),
        density=(1050, 25), ph=(6.9, 0.3), conductivity=(0.03, 0.02),
        voc=(25, 12), chlorine=(0.02, 0.02), ir_reflect=(0.36, 0.05),
    ),
    "PVC": dict(
        bands=[(1180, 40, 0.40), (1420, 40, 0.45), (1700, 35, 0.50),
               (2210, 45, 0.75), (2380, 55, 0.85), (2460, 55, 0.70)],
        baseline=(0.40, 0.05, -0.02), moisture=(0.4, 0.3),
        density=(1400, 45), ph=(6.2, 0.4), conductivity=(0.08, 0.04),
        voc=(30, 15), chlorine=(0.85, 0.08), ir_reflect=(0.33, 0.05),
    ),
    "PAPER": dict(
        bands=[(1450, 60, 0.85), (1490, 55, 0.70), (1930, 70, 0.95),
               (2100, 55, 0.65), (2270, 50, 0.55), (2330, 50, 0.50)],
        baseline=(0.30, 0.06, -0.03), moisture=(6.0, 2.0),
        density=(700, 120), ph=(6.5, 0.5), conductivity=(0.15, 0.06),
        voc=(5, 4), chlorine=(0.04, 0.03), ir_reflect=(0.25, 0.05),
    ),
    "CARDBOARD": dict(
        bands=[(1455, 62, 0.80), (1500, 55, 0.68), (1935, 72, 0.90),
               (2105, 55, 0.62), (2280, 52, 0.58)],
        baseline=(0.28, 0.06, -0.03), moisture=(8.0, 2.5),
        density=(650, 110), ph=(6.6, 0.5), conductivity=(0.18, 0.07),
        voc=(5, 4), chlorine=(0.04, 0.03), ir_reflect=(0.23, 0.05),
    ),
    "WOOD": dict(
        bands=[(1440, 65, 0.75), (1670, 45, 0.45), (1930, 75, 0.80),
               (2270, 55, 0.60), (2480, 60, 0.55)],
        baseline=(0.26, 0.07, -0.03), moisture=(12.0, 4.0),
        density=(550, 90), ph=(5.5, 0.6), conductivity=(0.20, 0.08),
        voc=(12, 7), chlorine=(0.03, 0.02), ir_reflect=(0.22, 0.05),
    ),
    "TEXTILE": dict(
        bands=[(1490, 55, 0.60), (1720, 40, 0.55), (1940, 70, 0.70),
               (2180, 50, 0.50), (2300, 50, 0.55)],
        baseline=(0.34, 0.05, -0.02), moisture=(7.0, 3.0),
        density=(400, 120), ph=(6.8, 0.5), conductivity=(0.12, 0.06),
        voc=(18, 10), chlorine=(0.05, 0.03), ir_reflect=(0.28, 0.05),
    ),
    "GLASS": dict(
        bands=[(1150, 80, 0.10), (2000, 120, 0.15)],
        baseline=(0.15, 0.02, 0.00), moisture=(0.3, 0.3),
        density=(2500, 60), ph=(8.5, 0.6), conductivity=(0.10, 0.05),
        voc=(2, 2), chlorine=(0.03, 0.02), ir_reflect=(0.70, 0.06),
    ),
    "FERROUS_METAL": dict(
        bands=[(1200, 150, 0.05)],
        baseline=(0.10, 0.01, 0.00), moisture=(0.5, 0.4),
        density=(7800, 120), ph=(7.5, 0.5), conductivity=(45.0, 12.0),
        voc=(3, 3), chlorine=(0.04, 0.03), ir_reflect=(0.82, 0.05),
    ),
    "ALUMINIUM": dict(
        bands=[(1200, 150, 0.04)],
        baseline=(0.09, 0.01, 0.00), moisture=(0.4, 0.3),
        density=(2700, 80), ph=(7.8, 0.5), conductivity=(60.0, 15.0),
        voc=(3, 3), chlorine=(0.04, 0.03), ir_reflect=(0.88, 0.04),
    ),
    "ORGANIC": dict(
        bands=[(1430, 70, 0.70), (1690, 45, 0.40), (1935, 80, 1.00),
               (2050, 55, 0.55), (2300, 55, 0.60)],
        baseline=(0.24, 0.05, -0.02), moisture=(55.0, 12.0),
        density=(600, 150), ph=(4.8, 0.8), conductivity=(2.5, 1.0),
        voc=(120, 45), chlorine=(0.05, 0.03), ir_reflect=(0.20, 0.05),
    ),
    "SOLVENT_WASTE": dict(
        bands=[(1150, 40, 0.55), (1400, 45, 0.50), (1690, 40, 0.65),
               (2250, 50, 0.70), (2450, 55, 0.60)],
        baseline=(0.30, 0.04, -0.02), moisture=(2.0, 1.5),
        density=(850, 90), ph=(5.5, 1.2), conductivity=(0.5, 0.4),
        voc=(600, 180), chlorine=(0.25, 0.15), ir_reflect=(0.32, 0.06),
    ),
}

MATERIAL_NAMES = list(MATERIALS.keys())
N_MATERIALS = len(MATERIAL_NAMES)


# ------------------------------------------------------------------
# Spectral building blocks
# ------------------------------------------------------------------

def _gaussian_band(centre, width, amplitude):
    """A single absorption feature evaluated on the global wavelength grid."""
    return amplitude * np.exp(-0.5 * ((WAVELENGTHS - centre) / width) ** 2)


def _clean_absorbance(material, rng, moisture_pct):
    """Noise-free absorbance spectrum for one material at a given moisture."""
    spec = MATERIALS[material]

    offset, slope, curv = spec["baseline"]
    x = (WAVELENGTHS - WL_MIN_NM) / 1000.0
    absorbance = offset + slope * x + curv * x ** 2

    for centre, width, amp in spec["bands"]:
        jitter_c = centre + rng.normal(0, 3.0)
        jitter_a = amp * (1.0 + rng.normal(0, 0.05))
        absorbance = absorbance + _gaussian_band(jitter_c, width, jitter_a)

    # water bands grow with moisture content
    m = max(moisture_pct, 0.0) / 100.0
    absorbance = absorbance + _gaussian_band(1450, 45, 1.10 * m)
    absorbance = absorbance + _gaussian_band(1940, 60, 1.40 * m)

    return absorbance


def _scalar_sensors(material, rng, purity, moisture_pct, contaminant):
    """Draw the scalar chemical-sensor bank for one sample."""
    spec = MATERIALS[material]
    cont = MATERIALS[contaminant]
    w = 1.0 - purity  # contaminant weight

    def blend(key):
        mu = (1 - w) * spec[key][0] + w * cont[key][0]
        sd = (1 - w) * spec[key][1] + w * cont[key][1]
        return mu, sd

    dens_mu, dens_sd = blend("density")
    ph_mu, ph_sd = blend("ph")
    cond_mu, cond_sd = blend("conductivity")
    voc_mu, voc_sd = blend("voc")
    cl_mu, cl_sd = blend("chlorine")
    ir_mu, ir_sd = blend("ir_reflect")

    # extra measurement noise grows as purity falls
    k = 1.0 + 2.0 * w

    values = np.array([
        moisture_pct,
        rng.normal(dens_mu, dens_sd * k),
        rng.normal(ph_mu, ph_sd * k),
        max(rng.normal(cond_mu, cond_sd * k), 0.0),
        max(rng.normal(voc_mu, voc_sd * k), 0.0),
        float(np.clip(rng.normal(cl_mu, cl_sd * k), 0.0, 1.0)),
        float(np.clip(rng.normal(ir_mu, ir_sd * k), 0.0, 1.0)),
        rng.normal(25.0, 4.0),
    ], dtype=np.float64)
    return values


# ------------------------------------------------------------------
# Public sampling API
# ------------------------------------------------------------------

def simulate_sample(material, rng, purity=None, moisture_pct=None,
                    contaminant=None, snr_db=35.0):
    """
    Simulate one multi-sensor observation.

    Returns
    -------
    dict with keys ``spectrum`` (N_WAVELENGTHS,), ``sensors`` (N_SENSORS,),
    ``material``, ``purity``, ``moisture_pct``, ``contaminant``.
    """
    if material not in MATERIALS:
        raise KeyError(f"unknown material {material!r}")

    if purity is None:
        # skew toward high purity, with a contaminated tail
        purity = float(np.clip(rng.beta(6.0, 1.6), 0.05, 1.0))

    if contaminant is None:
        others = [m for m in MATERIAL_NAMES if m != material]
        contaminant = others[rng.integers(len(others))]

    if moisture_pct is None:
        m_mu, m_sd = MATERIALS[material]["moisture"]
        cm_mu, cm_sd = MATERIALS[contaminant]["moisture"]
        w = 1.0 - purity
        moisture_pct = max(
            rng.normal((1 - w) * m_mu + w * cm_mu, (1 - w) * m_sd + w * cm_sd),
            0.0,
        )

    target = _clean_absorbance(material, rng, moisture_pct)
    other = _clean_absorbance(contaminant, rng, moisture_pct)
    mixed = purity * target + (1.0 - purity) * other

    # scatter / baseline drift (multiplicative + additive)
    mixed = mixed * (1.0 + rng.normal(0, 0.02))
    mixed = mixed + rng.normal(0, 0.01) + rng.normal(0, 0.004) * (
        (WAVELENGTHS - WL_MIN_NM) / 1000.0
    )

    # detector noise: fixed SNR plus a purity-dependent term
    sig_power = float(np.mean(mixed ** 2))
    noise_sd = np.sqrt(sig_power / (10.0 ** (snr_db / 10.0)))
    noise_sd *= 1.0 + 1.5 * (1.0 - purity)
    mixed = mixed + rng.normal(0, noise_sd, size=mixed.shape)

    # reflectance-style spectrum in a sane range
    spectrum = np.clip(mixed, 1e-3, None)

    sensors = _scalar_sensors(material, rng, purity, moisture_pct, contaminant)

    return dict(
        spectrum=spectrum.astype(np.float64),
        sensors=sensors,
        material=material,
        purity=float(purity),
        moisture_pct=float(moisture_pct),
        contaminant=contaminant,
    )


def simulate_batch(n, rng, materials=None, class_weights=None):
    """
    Vectorised convenience wrapper.

    Returns
    -------
    X_spec  : (n, N_WAVELENGTHS) float64
    X_sens  : (n, N_SENSORS)     float64
    y_mat   : (n,) int   material index into MATERIAL_NAMES
    y_pur   : (n,) float purity in [0, 1]
    """
    materials = materials or MATERIAL_NAMES
    idx_pool = np.array([MATERIAL_NAMES.index(m) for m in materials])

    if class_weights is not None:
        p = np.asarray(class_weights, dtype=float)
        p = p / p.sum()
    else:
        p = None

    X_spec = np.empty((n, N_WAVELENGTHS), dtype=np.float64)
    X_sens = np.empty((n, N_SENSORS), dtype=np.float64)
    y_mat = np.empty(n, dtype=np.int64)
    y_pur = np.empty(n, dtype=np.float64)

    for i in range(n):
        mat_idx = int(rng.choice(idx_pool, p=p))
        s = simulate_sample(MATERIAL_NAMES[mat_idx], rng)
        X_spec[i] = s["spectrum"]
        X_sens[i] = s["sensors"]
        y_mat[i] = mat_idx
        y_pur[i] = s["purity"]

    return X_spec, X_sens, y_mat, y_pur


if __name__ == "__main__":
    g = np.random.default_rng(0)
    demo = simulate_sample("PVC", g, purity=0.7)
    print("material          :", demo["material"])
    print("contaminant       :", demo["contaminant"])
    print("purity            :", round(demo["purity"], 3))
    print("moisture_pct      :", round(demo["moisture_pct"], 2))
    print("spectrum shape    :", demo["spectrum"].shape)
    for name, val in zip(SENSOR_CHANNELS, demo["sensors"]):
        print(f"  {name:20s} {val:12.4f}")
