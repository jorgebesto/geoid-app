"""
=============================================================================
  Modelo Geoidal EGM96 - Backend Principal
  Geodesia Física - Universidad Distrital Francisco José de Caldas
  
  Descripción:
    API REST para el cálculo de ondulación geoidal usando el modelo EGM96
    truncado, modelo de comparación con datos de referencia ICGEM, y
    modelo híbrido con correcciones topográficas.
=============================================================================
"""

import os
import io
import json
import logging
import warnings
import tempfile
import traceback
from pathlib import Path

import numpy as np
import scipy.special as sp
from scipy.interpolate import RegularGridInterpolator, griddata
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para generación de imágenes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import requests

from flask import Flask, request, jsonify, send_file, render_template

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Directorio de datos
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MAPS_DIR = Path(__file__).parent / "static" / "maps"
MAPS_DIR.mkdir(exist_ok=True)

# Constantes del elipsoide GRS80 / WGS84
A_ELLIPSOID = 6378137.0          # semieje mayor [m]
F_ELLIPSOID = 1.0 / 298.257223563  # aplanamiento
E2 = F_ELLIPSOID * (2.0 - F_ELLIPSOID)  # excentricidad al cuadrado
GM = 3.986004418e14              # parámetro gravitacional [m³/s²]
GAMMA_E = 9.7803253359           # gravedad normal en el ecuador [m/s²]
K_SOMA = 0.00193185265241        # constante de Somigliana


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE COEFICIENTES EGM96
# ─────────────────────────────────────────────────────────────────────────────

# Cache en memoria para los coeficientes (evitar recargar en cada request)
_COEF_CACHE: dict = {}


def cargar_coeficientes(gfc_path: Path, L_max: int = 170) -> tuple:
    """
    Lee los coeficientes Cnm y Snm desde un archivo .gfc (formato ICGEM).
    """
    mtime = gfc_path.stat().st_mtime if gfc_path.exists() else 0
    cache_key = f"{gfc_path}_{L_max}_{mtime}"
    
    if cache_key in _COEF_CACHE:
        logger.info("Coeficientes cargados desde caché.")
        return _COEF_CACHE[cache_key]

    logger.info(f"Leyendo coeficientes desde {gfc_path} hasta grado L={L_max}...")

    size = L_max + 1
    C = np.zeros((size, size), dtype=np.float64)
    S = np.zeros((size, size), dtype=np.float64)

    count = 0
    with open(gfc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.lower().startswith('end_of_head'):
                continue
            if line.lower().startswith('begin_of_head'):
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 4:
                continue
            
            idx_offset = 0
            if not parts[0].lstrip('-+').replace('.','',1).isdigit():
                idx_offset = 1
                
            if len(parts) < idx_offset + 4:
                continue
                
            try:
                n = int(parts[idx_offset])
                m = int(parts[idx_offset + 1])
                cnm = float(parts[idx_offset + 2])
                snm = float(parts[idx_offset + 3])
            except (ValueError, IndexError):
                continue

            if n > L_max:
                continue

            if n < size and m < size:
                C[n, m] = cnm
                S[n, m] = snm
                count += 1

    logger.info(f"  → {count} coeficientes cargados (n ≤ {L_max}).")
    if count == 0:
        logger.warning(f"ADVERTENCIA: No se pudo leer ningún coeficiente del archivo {gfc_path}.")
    _COEF_CACHE[cache_key] = (C, S)
    return C, S


def descargar_egm96_gfc() -> Path:
    """Descarga el archivo EGM96.gfc desde ICGEM si no existe localmente."""
    gfc_path = DATA_DIR / "EGM96.gfc"
    
    if gfc_path.exists() and gfc_path.stat().st_size > 1000:
        logger.info("Archivo EGM96.gfc ya existe localmente.")
        return gfc_path

    url = "http://icgem.gfz-potsdam.de/getmodel/gfc/7fd8fe44aa1518cd79ca84300aef4b41ddb2364aef9e82b7cdaabdb60a9053f1/EGM96.gfc"
    
    logger.info(f"Descargando EGM96.gfc desde ICGEM...")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(gfc_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        logger.info(f"Descarga completada: {gfc_path.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        logger.warning(f"No se pudo descargar EGM96.gfc: {e}")
        logger.info("Generando coeficientes sintéticos para modo demo...")
        _generar_coeficientes_sinteticos(gfc_path, L_max=170)

    return gfc_path


def _generar_coeficientes_sinteticos(gfc_path: Path, L_max: int = 170):
    """Genera un archivo .gfc sintético para modo demo."""
    logger.info("Generando coeficientes EGM96 sintéticos (modo demo)...")
    
    rng = np.random.default_rng(42)
    
    with open(gfc_path, 'w') as f:
        f.write("begin_of_head ================================\n")
        f.write("product_type              gravity_field\n")
        f.write("modelname                 EGM96_SYNTHETIC_DEMO\n")
        f.write("max_degree                360\n")
        f.write("norm                      fully_normalized\n")
        f.write("end_of_head ===================================\n")
        f.write(f"gfc   2   0  -4.841651437e-04   0.000000000e+00\n")
        
        for n in range(2, L_max + 1):
            sigma = 1e-5 / (n ** 2)
            for m in range(0, n + 1):
                cnm = rng.normal(0, sigma)
                snm = rng.normal(0, sigma) if m > 0 else 0.0
                if n == 2 and m == 0:
                    cnm = -4.841651437e-04
                f.write(f"gfc  {n:4d}  {m:4d}  {cnm:+.9e}  {snm:+.9e}\n")
    
    logger.info(f"  → Coeficientes sintéticos escritos en {gfc_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORMACIONES GEODÉSICAS
# ─────────────────────────────────────────────────────────────────────────────

def geodesica_a_geocentrica(lat_gd_deg: np.ndarray) -> np.ndarray:
    """Convierte latitud geodésica a latitud geocéntrica."""
    lat_gd_rad = np.radians(lat_gd_deg)
    lat_gc_rad = np.arctan((1.0 - E2) * np.tan(lat_gd_rad))
    return np.degrees(lat_gc_rad)


def radio_geocentrico(lat_gd_deg: np.ndarray) -> np.ndarray:
    """Calcula el radio geocéntrico para una latitud geodésica."""
    lat_rad = np.radians(lat_gd_deg)
    num = A_ELLIPSOID * np.sqrt(1.0 - E2)
    den = np.sqrt(1.0 - E2 * np.cos(lat_rad) ** 2)
    return num / den


def gravedad_normal_somigliana(lat_gd_deg: np.ndarray) -> np.ndarray:
    """Fórmula de gravedad normal de Somigliana sobre el elipsoide."""
    lat_rad = np.radians(lat_gd_deg)
    sin2 = np.sin(lat_rad) ** 2
    return GAMMA_E * (1.0 + K_SOMA * sin2) / np.sqrt(1.0 - E2 * sin2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. POLINOMIOS DE LEGENDRE NORMALIZADOS
# ─────────────────────────────────────────────────────────────────────────────

def calcular_legendre(x: float, n_max: int) -> np.ndarray:
    """
    Calcula los polinomios de Legendre totalmente normalizados P̄_nm(x).
    """
    size = n_max + 1
    P = np.zeros((size, size), dtype=np.float64)
    
    u = np.sqrt(max(0.0, 1.0 - x * x))
    P[0, 0] = 1.0
    
    for m in range(0, n_max + 1):
        if m > 0:
            factor = np.sqrt((2.0 * m + 1.0) / (2.0 * m))
            if m == 1:
                factor *= np.sqrt(2.0)
            P[m, m] = factor * u * P[m - 1, m - 1]
        
        if m + 1 <= n_max:
            P[m + 1, m] = np.sqrt(2.0 * m + 3.0) * x * P[m, m]
        
        for n in range(m + 2, n_max + 1):
            a = np.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
            b = np.sqrt((2.0 * n + 1.0) * (n - 1.0 - m) * (n - 1.0 + m) /
                        ((2.0 * n - 3.0) * (n * n - m * m)))
            P[n, m] = a * x * P[n - 1, m] - b * P[n - 2, m]
    
    return P


# ─────────────────────────────────────────────────────────────────────────────
# 4. SÍNTESIS DE ONDULACIÓN GEOIDAL EGM TRUNCADO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_egm_truncado(lats: np.ndarray, lons: np.ndarray,
                           C: np.ndarray, S: np.ndarray,
                           L_used: int = 170) -> np.ndarray:
    """
    Calcula la ondulación geoidal N(φ,λ) usando el modelo EGM96 truncado.
    """
    n_lat = len(lats)
    n_lon = len(lons)
    N_grid = np.zeros((n_lat, n_lon), dtype=np.float64)

    lons_rad = np.radians(lons)

    cos_ml = np.zeros((L_used + 1, n_lon), dtype=np.float64)
    sin_ml = np.zeros((L_used + 1, n_lon), dtype=np.float64)
    for m in range(L_used + 1):
        cos_ml[m, :] = np.cos(m * lons_rad)
        sin_ml[m, :] = np.sin(m * lons_rad)

    for i, lat_gd in enumerate(lats):
        if (i + 1) % 10 == 0:
            logger.info(f"  → Procesando latitud {i+1}/{n_lat} ({lat_gd:.2f}°)")

        lat_gc = geodesica_a_geocentrica(lat_gd)
        r = radio_geocentrico(lat_gd)
        gamma = gravedad_normal_somigliana(lat_gd)
        
        x_leg = np.sin(np.radians(lat_gc))
        a_over_r = A_ELLIPSOID / r

        P = calcular_legendre(x_leg, L_used)

        C_ref = {
            2: -4.84166774985e-04,
            4:  7.90304054e-07,
            6: -1.687251e-09,
            8:  3.461e-12
        }

        N_lat = np.zeros(n_lon, dtype=np.float64)
        for n in range(2, L_used + 1):
            a_r_n = a_over_r ** n
            for m in range(0, n + 1):
                pnm = P[n, m]
                if pnm == 0.0:
                    continue
                
                delta_C = C[n, m]
                if m == 0 and n in C_ref:
                    delta_C -= C_ref[n]
                
                harmonic = delta_C * cos_ml[m, :] + S[n, m] * sin_ml[m, :]
                N_lat += a_r_n * pnm * harmonic

        factor_bruns = GM / (r * gamma)
        N_0 = -0.53
        N_grid[i, :] = factor_bruns * N_lat + N_0

    return N_grid


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATOS DE REFERENCIA (ICGEM / EGM96 precomputado)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo_referencia(lat_min: float, lat_max: float,
                              lon_min: float, lon_max: float) -> tuple:
    """Obtiene datos de ondulación geoidal de referencia de ICGEM o sintéticos."""
    step = 0.5
    lats_ref = np.arange(lat_min, lat_max + step, step)
    lons_ref = np.arange(lon_min, lon_max + step, step)
    
    try:
        logger.info("Intentando obtener datos de referencia de ICGEM...")
        N_ref = _descargar_icgem_grid(lats_ref, lons_ref)
        if N_ref is not None:
            logger.info("Datos ICGEM obtenidos exitosamente.")
            return lats_ref, lons_ref, N_ref
    except Exception as e:
        logger.warning(f"ICGEM no disponible: {e}")

    logger.info("Generando datos de referencia sintéticos...")
    N_ref = _referencia_sintetica(lats_ref, lons_ref)
    return lats_ref, lons_ref, N_ref


def _descargar_icgem_grid(lats: np.ndarray, lons: np.ndarray) -> np.ndarray | None:
    """Descarga la ondulación geoidal del servicio web ICGEM."""
    url = "http://icgem.gfz-potsdam.de/calcgrid"
    params = {
        'model': 'egm96',
        'lat1': lats[0],
        'lat2': lats[-1],
        'lon1': lons[0],
        'lon2': lons[-1],
        'step': round(lats[1] - lats[0], 4) if len(lats) > 1 else 0.5,
        'func': 'geoid_undulation',
        'format': 'gdf',
        'height': 0,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    
    lines = resp.text.strip().split('\n')
    data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
    
    values = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                values.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    
    if not values:
        return None
    
    n_lat = len(lats)
    n_lon = len(lons)
    N_grid = np.full((n_lat, n_lon), np.nan)
    
    for lat_v, lon_v, n_v in values:
        i = np.argmin(np.abs(lats - lat_v))
        j = np.argmin(np.abs(lons - lon_v))
        N_grid[i, j] = n_v
    
    return N_grid


def _referencia_sintetica(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Genera una superficie de referencia sintética realista para Colombia."""
    LAT_G, LON_G = np.meshgrid(lats, lons, indexing='ij')
    
    N = 5.0 + 8.0 * np.sin(np.radians(LAT_G + 10)) + \
        3.0 * np.cos(np.radians(LON_G + 72))
    
    dist_andes = np.sqrt((LAT_G - 4.0)**2 + (LON_G + 73.0)**2)
    N += 12.0 * np.exp(-dist_andes**2 / 20.0)
    
    dist_amaz = np.sqrt((LAT_G + 1.0)**2 + (LON_G + 71.0)**2)
    N -= 8.0 * np.exp(-dist_amaz**2 / 30.0)
    
    dist_pac = np.sqrt((LAT_G - 4.0)**2 + (LON_G + 77.0)**2)
    N -= 5.0 * np.exp(-dist_pac**2 / 15.0)
    
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 0.3, N.shape)
    from scipy.ndimage import gaussian_filter
    N += gaussian_filter(noise, sigma=2)
    
    return N


def interpolar_datos(lats_ref: np.ndarray, lons_ref: np.ndarray,
                     N_ref: np.ndarray,
                     lats_target: np.ndarray, lons_target: np.ndarray) -> np.ndarray:
    """Interpola el grid de referencia a las coordenadas del modelo calculado."""
    mask_nan = np.isnan(N_ref)
    if mask_nan.any():
        mean_val = np.nanmean(N_ref)
        N_ref = np.where(mask_nan, mean_val, N_ref)
    
    interpolador = RegularGridInterpolator(
        (lats_ref, lons_ref), N_ref,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    LAT_G, LON_G = np.meshgrid(lats_target, lons_target, indexing='ij')
    pts = np.column_stack([LAT_G.ravel(), LON_G.ravel()])
    N_interp = interpolador(pts).reshape(len(lats_target), len(lons_target))
    
    return N_interp


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODELO HÍBRIDO
# ─────────────────────────────────────────────────────────────────────────────

def generar_dem_sintetico(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Genera un DEM sintético geomorfológicamente realista para la región."""
    LAT_G, LON_G = np.meshgrid(lats, lons, indexing='ij')
    
    cord_occ = 2500 * np.exp(-((LON_G + 76.5)**2 + (LAT_G - 3)**2) / 4)
    cord_cen = 3500 * np.exp(-((LON_G + 75.5)**2 + (LAT_G - 4)**2) / 5)
    cord_ori = 2800 * np.exp(-((LON_G + 73.5)**2 + (LAT_G - 5)**2) / 6)
    valle_mag = -500 * np.exp(-((LON_G + 74.8)**2 + (LAT_G - 5)**2) / 3)
    amazonia = 200 * np.maximum(0, (LON_G + 73) / 3)
    costa_pac = 100 * np.maximum(0, -(LON_G + 77))
    
    H = np.maximum(0, cord_occ + cord_cen + cord_ori + valle_mag + amazonia + costa_pac)
    
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(42)
    rugosidad = rng.normal(0, 100, H.shape)
    H += gaussian_filter(rugosidad, sigma=1.5)
    H = np.maximum(0, H)
    
    return H


def calcular_hibrido(N_egmud: np.ndarray, H: np.ndarray,
                     lats: np.ndarray) -> np.ndarray:
    """Calcula el modelo geoidal híbrido con correcciones topográficas."""
    gamma_arr = gravedad_normal_somigliana(lats)
    gamma_2d = gamma_arr[:, np.newaxis]
    
    delta_N_FA = -0.001 * H
    
    rho_corteza = 2670.0
    G_grav = 6.674e-11
    delta_N_H = -(np.pi * G_grav * rho_corteza * H ** 2) / gamma_2d
    
    N_hibrido = N_egmud + delta_N_FA + delta_N_H
    
    return N_hibrido


# ─────────────────────────────────────────────────────────────────────────────
# 7. GENERACIÓN DE MAPAS  ← FUNCIÓN CORREGIDA
# ─────────────────────────────────────────────────────────────────────────────

def generar_mapa(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                 titulo: str, etiqueta_barra: str, nombre_archivo: str,
                 colormap: str = 'RdYlBu_r',
                 divergente: bool = False) -> str:
    """
    Genera y guarda un mapa de ondulación geoidal como imagen PNG.
    
    FIX: Se eliminó ax.set_aspect() que causaba espacios en blanco enormes
    al forzar un ratio de aspecto que desbordaba el contenedor de la figura.
    En su lugar se calcula figsize directamente desde el ratio geográfico
    para que la imagen llene el espacio sin márgenes muertos.
    """
    LON_G, LAT_G = np.meshgrid(lons, lats)

    # ── Ratio geográfico real de la región ──────────────────────────────────
    lon_range = float(lons[-1] - lons[0])
    lat_range = float(lats[-1] - lats[0])
    lat_mid   = float(np.mean(lats))
    cos_corr  = max(np.cos(np.radians(lat_mid)), 0.01)

    # Ancho visual real vs alto visual real (en "grados equivalentes")
    ancho_geo = lon_range * cos_corr
    alto_geo  = lat_range

    # Área de datos dentro de la figura (sin márgenes)
    DATA_W = 7.0  # pulgadas fijas para el ancho de datos
    DATA_H = DATA_W * (alto_geo / max(ancho_geo, 0.01))

    # Acotar para que no sea ni minúscula ni gigante
    DATA_H = max(3.0, min(8.0, DATA_H))

    # Márgenes adicionales para colorbar, ejes, título
    MARGIN_L = 0.7   # espacio para ylabel + tick labels
    MARGIN_R = 1.4   # espacio para colorbar
    MARGIN_B = 0.55  # espacio para xlabel + tick labels
    MARGIN_T = 0.5   # espacio para título

    fig_w = DATA_W + MARGIN_L + MARGIN_R
    fig_h = DATA_H + MARGIN_B + MARGIN_T

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=110)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Posicionar el axes para que ocupe exactamente el área de datos
    # [left, bottom, width, height] en fracción de la figura
    left   = MARGIN_L / fig_w
    bottom = MARGIN_B / fig_h
    width  = DATA_W   / fig_w
    height = DATA_H   / fig_h
    ax.set_position([left, bottom, width, height])

    # ── Datos ───────────────────────────────────────────────────────────────
    vmin, vmax = np.nanmin(data), np.nanmax(data)

    if divergente and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax.pcolormesh(LON_G, LAT_G, data, cmap=colormap, norm=norm, shading='auto')
    else:
        im = ax.pcolormesh(LON_G, LAT_G, data, cmap=colormap,
                           vmin=vmin, vmax=vmax, shading='auto')

    # SIN set_aspect() — el figsize ya garantiza la proporción correcta

    # ── Colorbar ─────────────────────────────────────────────────────────────
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.95)
    cbar.set_label(etiqueta_barra, color='white', fontsize=9, labelpad=6)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=7)
    cbar.outline.set_edgecolor('white')

    # ── Cuadrícula ───────────────────────────────────────────────────────────
    ax.grid(True, linestyle='--', alpha=0.25, color='white', linewidth=0.4)

    # ── Contornos ────────────────────────────────────────────────────────────
    try:
        contour = ax.contour(LON_G, LAT_G, data,
                             levels=6, colors='white', linewidths=0.4, alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=6, fmt='%.1f', colors='white')
    except Exception:
        pass

    # ── Ejes ─────────────────────────────────────────────────────────────────
    ax.set_xlabel('Longitud (°)', color='white', fontsize=9)
    ax.set_ylabel('Latitud (°)',  color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=7, length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # ── Título ───────────────────────────────────────────────────────────────
    ax.set_title(titulo, color='white', fontsize=10, fontweight='bold', pad=8)

    # ── Stats superpuestas ───────────────────────────────────────────────────
    stats_text = (f"min={vmin:.2f}  max={vmax:.2f}  "
                  f"μ={np.nanmean(data):.2f}  σ={np.nanstd(data):.2f}")
    ax.text(0.01, 0.01, stats_text, transform=ax.transAxes,
            color='lightgray', fontsize=6, va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # ── Guardar ──────────────────────────────────────────────────────────────
    # bbox_inches='tight' recorta espacios en blanco residuales automáticamente
    out_path = MAPS_DIR / f"{nombre_archivo}.png"
    plt.savefig(out_path, dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor(), pad_inches=0.08)
    plt.close(fig)

    logger.info(f"Mapa guardado: {out_path} ({fig_w:.1f}x{fig_h:.1f} in)")
    return f"/static/maps/{nombre_archivo}.png"


def calcular_metricas(N_ref: np.ndarray, N_calc: np.ndarray) -> dict:
    """Calcula métricas de error entre modelo calculado y referencia."""
    diff = N_ref - N_calc
    valid = ~np.isnan(diff)
    d = diff[valid]
    
    return {
        "error_medio": float(np.mean(d)),
        "rmse": float(np.sqrt(np.mean(d**2))),
        "error_max": float(np.max(d)),
        "error_min": float(np.min(d)),
        "std_error": float(np.std(d)),
        "n_puntos": int(valid.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENDPOINTS DE LA API REST
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def status():
    """Verifica que el servidor está activo."""
    gfc_path = DATA_DIR / "EGM96.gfc"
    return jsonify({
        "status": "ok",
        "egm96_disponible": gfc_path.exists(),
        "egm96_size_mb": round(gfc_path.stat().st_size / 1e6, 1) if gfc_path.exists() else 0
    })


@app.route('/api/compute-legendre', methods=['POST'])
def compute_legendre():
    """
    Calcula un polinomio de Legendre P_nm para una latitud dada.
    """
    try:
        data = request.json
        lat_gd = float(data.get('lat', 4.6097))
        n = int(data.get('n', 2))
        m = int(data.get('m', 0))

        if m > n:
            return jsonify({"error": "El orden m no puede ser mayor al grado n."}), 400
        if n < 0 or m < 0:
            return jsonify({"error": "Grado y orden deben ser positivos."}), 400

        # Transformación a latitud geocéntrica
        lat_gc = geodesica_a_geocentrica(np.array([lat_gd]))[0]
        x_leg = np.sin(np.radians(lat_gc))

        # Calcular todos los polinomios hasta n
        P = calcular_legendre(x_leg, n)
        valor_pnm = P[n, m]

        # Generar fórmula abstracta en LaTeX
        k = 1 if m == 0 else 2
        import math
        try:
            if n <= 20: # Evitar desbordamiento de factoriales
                norm_factor = math.sqrt(k * (2*n + 1) * math.factorial(n-m) / math.factorial(n+m))
                rod_factor = 1.0 / ((2**n) * math.factorial(n))
                C = norm_factor * rod_factor
                # Format C in scientific notation for LaTeX
                c_sci = f"{C:.4e}".split('e')
                base = c_sci[0]
                exp = int(c_sci[1])
                if exp != 0:
                    C_str = f"{base} \\times 10^{{{exp}}}"
                else:
                    C_str = f"{base}"
            else:
                C_str = f"N_{{{n},{m}}} \\cdot \\frac{{1}}{{2^{{{n}}} {n}!}}"
        except OverflowError:
            C_str = f"N_{{{n},{m}}} \\cdot \\frac{{1}}{{2^{{{n}}} {n}!}}"

        deriv_str = f"\\frac{{d^{{{n+m}}}}}{{dx^{{{n+m}}}}}" if (n+m) > 0 else ""
        m_str = f"(1-x^2)^{{{m}/2}}" if m > 0 else ""
        
        formula_latex = f"\\bar{{P}}_{{{n},{m}}}(x) = {C_str}"
        if m_str: formula_latex += f" \\cdot {m_str}"
        if deriv_str: formula_latex += f" \\cdot {deriv_str} (x^2 - 1)^{{{n}}}"

        return jsonify({
            "success": True,
            "lat_gd": lat_gd,
            "lat_gc": float(lat_gc),
            "x_leg": float(x_leg),
            "n": n,
            "m": m,
            "valor": float(valor_pnm),
            "formula": formula_latex
        })
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/compute-egmud', methods=['POST'])
def compute_egmud():
    """
    Calcula la ondulación geoidal con el modelo EGM96 truncado (EGMUD).
    """
    try:
        data = request.form
        lat_min = float(data.get('lat_min', -5.0))
        lat_max = float(data.get('lat_max', 15.0))
        lon_min = float(data.get('lon_min', -80.0))
        lon_max = float(data.get('lon_max', -65.0))
        resolucion = float(data.get('resolucion', 0.5))
        L_max = int(data.get('L_max', 170))
        
        file_gfc = request.files.get('file_gfc')
        if not file_gfc:
            return jsonify({"error": "No se subió archivo .gfc"}), 400
        
        gfc_path = DATA_DIR / "uploaded.gfc"
        file_gfc.save(gfc_path)
        
        if lat_max <= lat_min or lon_max <= lon_min:
            return jsonify({"error": "Límites geográficos inválidos"}), 400
        if resolucion < 0.05 or resolucion > 5.0:
            return jsonify({"error": "Resolución debe estar entre 0.05° y 5.0°"}), 400
        if L_max < 2 or L_max > 360:
            return jsonify({"error": "L_max debe estar entre 2 y 360"}), 400

        logger.info(f"EGMUD: región [{lat_min},{lat_max}]×[{lon_min},{lon_max}] "
                    f"res={resolucion}° L={L_max}")

        lats = np.arange(lat_min, lat_max + resolucion / 2, resolucion)
        lons = np.arange(lon_min, lon_max + resolucion / 2, resolucion)
        
        C, S = cargar_coeficientes(gfc_path, L_max)
        N_egmud = calcular_egm_truncado(lats, lons, C, S, L_used=L_max)
        
        np.save(DATA_DIR / "N_egmud_last.npy", N_egmud)
        np.save(DATA_DIR / "lats_last.npy", lats)
        np.save(DATA_DIR / "lons_last.npy", lons)
        
        url_mapa = generar_mapa(
            N_egmud, lats, lons,
            titulo=f"Ondulación Geoidal EGMUD — EGM96 (L={L_max})",
            etiqueta_barra="N (m)",
            nombre_archivo="egmud",
            colormap='terrain',
            divergente=False
        )
        
        return jsonify({
            "success": True,
            "mapa_url": url_mapa,
            "estadisticas": {
                "min_m": float(np.nanmin(N_egmud)),
                "max_m": float(np.nanmax(N_egmud)),
                "media_m": float(np.nanmean(N_egmud)),
                "std_m": float(np.nanstd(N_egmud)),
            },
            "n_lat": len(lats),
            "n_lon": len(lons),
            "grid_bounds": {
                "lat_min": float(lats[0]),
                "lat_max": float(lats[-1]),
                "lon_min": float(lons[0]),
                "lon_max": float(lons[-1]),
            },
            "grid_data": N_egmud.tolist(),
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/compute-comparison', methods=['POST'])
def compute_comparison():
    """
    Calcula el mapa de diferencias entre el modelo de referencia y EGMUD.
    """
    try:
        data = request.get_json()
        
        if not (DATA_DIR / "N_egmud_last.npy").exists():
            return jsonify({"error": "Debe calcular EGMUD primero"}), 400
        
        N_egmud = np.load(DATA_DIR / "N_egmud_last.npy")
        lats = np.load(DATA_DIR / "lats_last.npy")
        lons = np.load(DATA_DIR / "lons_last.npy")
        
        lat_min, lat_max = float(lats[0]), float(lats[-1])
        lon_min, lon_max = float(lons[0]), float(lons[-1])
        
        lats_ref, lons_ref, N_ref_raw = cargar_modelo_referencia(
            lat_min, lat_max, lon_min, lon_max
        )
        
        N_ref = interpolar_datos(lats_ref, lons_ref, N_ref_raw, lats, lons)
        diferencia = N_ref - N_egmud
        
        np.save(DATA_DIR / "N_ref_last.npy", N_ref)
        
        metricas = calcular_metricas(N_ref, N_egmud)
        
        url_mapa = generar_mapa(
            diferencia, lats, lons,
            titulo="Diferencia: Referencia EGM96 − EGMUD (L=170)",
            etiqueta_barra="ΔN (m)",
            nombre_archivo="comparacion",
            colormap='RdBu_r',
            divergente=True
        )
        
        return jsonify({
            "success": True,
            "mapa_url": url_mapa,
            "metricas": metricas,
            "grid_bounds": {
                "lat_min": float(lats[0]),
                "lat_max": float(lats[-1]),
                "lon_min": float(lons[0]),
                "lon_max": float(lons[-1]),
            },
            "grid_data": diferencia.tolist(),
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/compute-hybrid', methods=['POST'])
def compute_hybrid():
    """
    Calcula el modelo geoidal híbrido mediante Regresión Múltiple.
    """
    try:
        if not (DATA_DIR / "N_egmud_last.npy").exists():
            return jsonify({"error": "Debe calcular EGMUD primero"}), 400
        if not (DATA_DIR / "N_ref_last.npy").exists():
            return jsonify({"error": "Debe generar la Comparación primero"}), 400
            
        N_egmud = np.load(DATA_DIR / "N_egmud_last.npy")
        N_ref = np.load(DATA_DIR / "N_ref_last.npy")
        lats = np.load(DATA_DIR / "lats_last.npy")
        lons = np.load(DATA_DIR / "lons_last.npy")
        
        file_corr = request.files.get('file_corr')
        if not file_corr:
            return jsonify({"error": "No se subió archivo de correcciones"}), 400
        
        content = file_corr.read().decode('utf-8', errors='ignore')
        lines = content.strip().split('\n')
        data_lines = lines[1:]
        
        lon_pts, lat_pts, h_pts, corr_pts = [], [], [], []
        
        for line in data_lines:
            line = line.strip()
            if not line: continue
            
            if ';' in line:
                parts = line.split(';')
            elif '\t' in line:
                parts = line.split('\t')
            else:
                if line.count(',') >= 3:
                    parts = line.split(',')
                else:
                    parts = line.split()
            
            if len(parts) >= 4:
                try:
                    lon_pts.append(float(parts[0].strip().replace(',', '.')))
                    lat_pts.append(float(parts[1].strip().replace(',', '.')))
                    h_pts.append(float(parts[2].strip().replace(',', '.')))
                    corr_pts.append(float(parts[3].strip().replace(',', '.')))
                except ValueError:
                    continue
                    
        lon_pts  = np.array(lon_pts)
        lat_pts  = np.array(lat_pts)
        h_pts    = np.array(h_pts)
        corr_pts = np.array(corr_pts)
        
        if len(corr_pts) == 0:
            return jsonify({"error": "El archivo de correcciones no contiene datos válidos."}), 400
            
        # Regresión Múltiple (Mínimos Cuadrados)
        interp_egmud = RegularGridInterpolator(
            (lats, lons), N_egmud, method='linear',
            bounds_error=False, fill_value=np.nan)
        interp_ref = RegularGridInterpolator(
            (lats, lons), N_ref, method='linear',
            bounds_error=False, fill_value=np.nan)
        
        pts_coords    = np.column_stack((lat_pts, lon_pts))
        N_egmud_pts   = interp_egmud(pts_coords)
        N_ref_pts     = interp_ref(pts_coords)
        
        valid = ~(np.isnan(N_egmud_pts) | np.isnan(N_ref_pts))
        if np.sum(valid) < 3:
            return jsonify({"error": "No hay suficientes puntos dentro de la región."}), 400
            
        delta_N_pts = N_ref_pts[valid] - N_egmud_pts[valid]
        H_valid     = h_pts[valid]
        Corr_valid  = corr_pts[valid]
        
        A = np.column_stack((np.ones_like(H_valid), H_valid, Corr_valid))
        beta, _, _, _ = np.linalg.lstsq(A, delta_N_pts, rcond=None)
        beta_0, beta_1, beta_2 = beta
        logger.info(f"Regresión: offset={beta_0:.4f}, factor_H={beta_1:.6e}, factor_FA={beta_2:.6e}")
        
        LAT_G, LON_G = np.meshgrid(lats, lons, indexing='ij')
        pts_scatter  = np.column_stack((lon_pts, lat_pts))
        
        H_grid    = griddata(pts_scatter, h_pts,    (LON_G, LAT_G), method='linear')
        Corr_grid = griddata(pts_scatter, corr_pts, (LON_G, LAT_G), method='linear')
        
        mask_nan = np.isnan(H_grid)
        if np.any(mask_nan):
            H_nearest = griddata(pts_scatter, h_pts, (LON_G, LAT_G), method='nearest')
            H_grid[mask_nan] = H_nearest[mask_nan]
            
        mask_nan_c = np.isnan(Corr_grid)
        if np.any(mask_nan_c):
            C_nearest = griddata(pts_scatter, corr_pts, (LON_G, LAT_G), method='nearest')
            Corr_grid[mask_nan_c] = C_nearest[mask_nan_c]
            
        N_hibrido = N_egmud + beta_0 + (beta_1 * H_grid) + (beta_2 * Corr_grid)
        
        url_mapa = generar_mapa(
            N_hibrido, lats, lons,
            titulo="Modelo Híbrido (Regresión Múltiple)",
            etiqueta_barra="N_híbrido (m)",
            nombre_archivo="hibrido",
            colormap='RdYlBu_r',
            divergente=False
        )
        
        metricas_hibrido = None
        if (DATA_DIR / "N_ref_last.npy").exists():
            N_ref = np.load(DATA_DIR / "N_ref_last.npy")
            metricas_hibrido = calcular_metricas(N_ref, N_hibrido)
        
        return jsonify({
            "success": True,
            "mapa_url": url_mapa,
            "dem_url": None,
            "estadisticas": {
                "min_m": float(np.nanmin(N_hibrido)),
                "max_m": float(np.nanmax(N_hibrido)),
                "media_m": float(np.nanmean(N_hibrido)),
                "std_m": float(np.nanstd(N_hibrido)),
            },
            "metricas_vs_referencia": metricas_hibrido,
            "grid_bounds": {
                "lat_min": float(lats[0]),
                "lat_max": float(lats[-1]),
                "lon_min": float(lons[0]),
                "lon_max": float(lons[-1]),
            },
            "grid_data": N_hibrido.tolist(),
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/descargar-coeficientes', methods=['POST'])
def descargar_coeficientes_endpoint():
    """Descarga o verifica los coeficientes EGM96."""
    try:
        gfc_path = descargar_egm96_gfc()
        size_mb = gfc_path.stat().st_size / 1e6
        return jsonify({
            "success": True,
            "ruta": str(gfc_path),
            "size_mb": round(size_mb, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info("═" * 60)
    logger.info("  Modelo Geoidal EGM96 — Geodesia Física")
    logger.info("  Universidad Distrital Francisco José de Caldas")
    logger.info("═" * 60)
    logger.info(f"  Directorio de datos: {DATA_DIR}")
    logger.info(f"  Directorio de mapas: {MAPS_DIR}")
    logger.info("  Iniciando servidor en http://localhost:5000")
    logger.info("═" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)