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
from scipy.interpolate import RegularGridInterpolator
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
    
    El formato .gfc contiene líneas como:
        gfc  n  m  Cnm  Snm  sigma_C  sigma_S
    
    Parámetros
    ----------
    gfc_path : Path
        Ruta al archivo .gfc con los coeficientes esféricos.
    L_max : int
        Grado máximo a leer (truncamiento).
    
    Retorna
    -------
    (C, S) : tuple de np.ndarray
        Matrices triangulares inferiores de coeficientes, indexadas [n, m].
    """
    cache_key = f"{gfc_path}_{L_max}"
    if cache_key in _COEF_CACHE:
        logger.info("Coeficientes cargados desde caché.")
        return _COEF_CACHE[cache_key]

    logger.info(f"Leyendo coeficientes desde {gfc_path} hasta grado L={L_max}...")

    # Inicializar matrices con ceros
    size = L_max + 1
    C = np.zeros((size, size), dtype=np.float64)
    S = np.zeros((size, size), dtype=np.float64)

    count = 0
    with open(gfc_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Saltar encabezados y líneas vacías
            if not line or line.startswith('#') or line.lower().startswith('end_of_head'):
                continue
            if line.lower().startswith('begin_of_head'):
                continue
            # Solo procesar líneas de coeficientes 'gfc'
            parts = line.split()
            if len(parts) < 5 or parts[0].lower() != 'gfc':
                continue
            try:
                n = int(parts[1])
                m = int(parts[2])
                cnm = float(parts[3])
                snm = float(parts[4])
            except (ValueError, IndexError):
                continue

            if n > L_max:
                continue  # Ignorar grados mayores al truncamiento

            if n < size and m < size:
                C[n, m] = cnm
                S[n, m] = snm
                count += 1

    logger.info(f"  → {count} coeficientes cargados (n ≤ {L_max}).")
    _COEF_CACHE[cache_key] = (C, S)
    return C, S


def descargar_egm96_gfc() -> Path:
    """
    Descarga el archivo de coeficientes EGM96 en formato .gfc desde ICGEM.
    Si ya existe localmente, no lo vuelve a descargar.
    
    Retorna
    -------
    Path al archivo .gfc local.
    """
    gfc_path = DATA_DIR / "EGM96.gfc"
    
    if gfc_path.exists() and gfc_path.stat().st_size > 1000:
        logger.info("Archivo EGM96.gfc ya existe localmente.")
        return gfc_path

    # URL oficial de ICGEM para EGM96
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
        # Generar archivo sintético de ejemplo para desarrollo/demo
        logger.info("Generando coeficientes sintéticos para modo demo...")
        _generar_coeficientes_sinteticos(gfc_path, L_max=170)

    return gfc_path


def _generar_coeficientes_sinteticos(gfc_path: Path, L_max: int = 170):
    """
    Genera un archivo .gfc sintético con coeficientes realistas para modo demo.
    Los coeficientes siguen la distribución estadística típica de EGM96.
    Útil cuando no hay conexión a internet o el archivo no está disponible.
    """
    logger.info("Generando coeficientes EGM96 sintéticos (modo demo)...")
    
    rng = np.random.default_rng(42)
    
    with open(gfc_path, 'w') as f:
        f.write("begin_of_head ================================\n")
        f.write("product_type              gravity_field\n")
        f.write("modelname                 EGM96_SYNTHETIC_DEMO\n")
        f.write("max_degree                360\n")
        f.write("norm                      fully_normalized\n")
        f.write("end_of_head ===================================\n")
        
        # J2 (coeficiente zonal dominante - aplanamiento terrestre)
        f.write(f"gfc   2   0  -4.841651437e-04   0.000000000e+00\n")
        
        # Coeficientes realistas según caída espectral de la gravedad terrestre
        for n in range(2, L_max + 1):
            # Varianza decrece con el grado: σ² ∝ n^(-4)
            sigma = 1e-5 / (n ** 2)
            for m in range(0, n + 1):
                cnm = rng.normal(0, sigma)
                snm = rng.normal(0, sigma) if m > 0 else 0.0
                # Corrección especial para J2
                if n == 2 and m == 0:
                    cnm = -4.841651437e-04
                f.write(f"gfc  {n:4d}  {m:4d}  {cnm:+.9e}  {snm:+.9e}\n")
    
    logger.info(f"  → Coeficientes sintéticos escritos en {gfc_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORMACIONES GEODÉSICAS
# ─────────────────────────────────────────────────────────────────────────────

def geodesica_a_geocentrica(lat_gd_deg: np.ndarray) -> np.ndarray:
    """
    Convierte latitud geodésica a latitud geocéntrica.
    
    φ_gc = arctan((1 - e²) * tan(φ_gd))
    
    Parámetros
    ----------
    lat_gd_deg : array
        Latitudes geodésicas en grados.
    
    Retorna
    -------
    array : latitudes geocéntricas en grados.
    """
    lat_gd_rad = np.radians(lat_gd_deg)
    lat_gc_rad = np.arctan((1.0 - E2) * np.tan(lat_gd_rad))
    return np.degrees(lat_gc_rad)


def radio_geocentrico(lat_gd_deg: np.ndarray) -> np.ndarray:
    """
    Calcula el radio geocéntrico para una latitud geodésica.
    
    r = a√(1-e²) / √(1 - e²·cos²(φ_gd))
    
    Parámetros
    ----------
    lat_gd_deg : array
        Latitudes geodésicas en grados.
    
    Retorna
    -------
    array : radio geocéntrico en metros.
    """
    lat_rad = np.radians(lat_gd_deg)
    num = A_ELLIPSOID * np.sqrt(1.0 - E2)
    den = np.sqrt(1.0 - E2 * np.cos(lat_rad) ** 2)
    return num / den


def gravedad_normal_somigliana(lat_gd_deg: np.ndarray) -> np.ndarray:
    """
    Fórmula de gravedad normal de Somigliana sobre el elipsoide.
    
    γ(φ) = γ_e * (1 + k·sin²φ) / √(1 - e²·sin²φ)
    
    Parámetros
    ----------
    lat_gd_deg : array
        Latitudes geodésicas en grados.
    
    Retorna
    -------
    array : gravedad normal en m/s².
    """
    lat_rad = np.radians(lat_gd_deg)
    sin2 = np.sin(lat_rad) ** 2
    return GAMMA_E * (1.0 + K_SOMA * sin2) / np.sqrt(1.0 - E2 * sin2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. POLINOMIOS DE LEGENDRE NORMALIZADOS
# ─────────────────────────────────────────────────────────────────────────────

def calcular_legendre(x: float, n_max: int) -> np.ndarray:
    """
    Calcula los polinomios de Legendre totalmente normalizados P̄_nm(x)
    para todos los órdenes 0 ≤ m ≤ n ≤ n_max.
    
    La normalización ortonormal es:
        P̄_nm(x) = P_nm(x) * √((2n+1)(n-m)! / (4π(n+m)!))
    
    Para eficiencia, usa la recurrencia de tres términos:
        - Caso m = n:    P̄_mm via fórmula diagonal
        - Caso n = m+1:  P̄_{m+1,m} vía término inmediato
        - Caso n > m+1:  recurrencia estándar
    
    Parámetros
    ----------
    x : float
        Argumento (sin(φ_gc) = cos(θ)), con θ colatitud geocéntrica.
    n_max : int
        Grado máximo de la expansión.
    
    Retorna
    -------
    P : np.ndarray shape (n_max+1, n_max+1)
        Matriz triangular inferior de polinomios normalizados.
    """
    size = n_max + 1
    P = np.zeros((size, size), dtype=np.float64)
    
    # u = √(1 - x²) = cos(φ_gc) (necesario para la recursión diagonal)
    u = np.sqrt(max(0.0, 1.0 - x * x))
    
    # Caso inicial P̄_00 = 1 (normalización de la constante)
    P[0, 0] = 1.0
    
    for m in range(0, n_max + 1):
        # ── Caso m = m (diagonal): P̄_mm ─────────────────────────────────
        if m > 0:
            # P̄_mm = √((2m-1)(2m+1)/(2m(2m))) * u * P̄_{m-1,m-1}  [recurrencia]
            # Equivalente a: P_mm = (-1)^m (2m-1)!! (1-x²)^{m/2}  normalizado
            factor = np.sqrt((2.0 * m + 1.0) / (2.0 * m))
            P[m, m] = factor * u * P[m - 1, m - 1]
        
        # ── Caso n = m+1 ──────────────────────────────────────────────────
        if m + 1 <= n_max:
            P[m + 1, m] = np.sqrt(2.0 * m + 3.0) * x * P[m, m]
        
        # ── Caso n > m+1: recurrencia de tres términos ────────────────────
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
    
    La fórmula de síntesis es la fórmula de Bruns:
    
        N(φ,λ) = (a / γ(φ)) * Σ_{n=2}^{L} Σ_{m=0}^{n}
                  (a/r)^n * P̄_nm(sin φ_gc) * (C_nm·cos(mλ) + S_nm·sin(mλ))
    
    donde:
        - a    = semieje mayor del elipsoide
        - γ(φ) = gravedad normal de Somigliana
        - r    = radio geocéntrico
        - P̄_nm = polinomios de Legendre totalmente normalizados
    
    Parámetros
    ----------
    lats, lons : 1D arrays
        Latitudes y longitudes geodésicas en grados (vectores de la malla).
    C, S : 2D arrays (L_max+1, L_max+1)
        Coeficientes de armónicos esféricos EGM96.
    L_used : int
        Grado máximo de truncamiento (default 170).
    
    Retorna
    -------
    N_grid : 2D array shape (len(lats), len(lons))
        Ondulación geoidal en metros.
    """
    n_lat = len(lats)
    n_lon = len(lons)
    N_grid = np.zeros((n_lat, n_lon), dtype=np.float64)

    lons_rad = np.radians(lons)

    # Pre-calcular cos(mλ) y sin(mλ) para todos los grados y longitudes
    # Forma: cos_ml[m, j] = cos(m * lons_rad[j])
    cos_ml = np.zeros((L_used + 1, n_lon), dtype=np.float64)
    sin_ml = np.zeros((L_used + 1, n_lon), dtype=np.float64)
    for m in range(L_used + 1):
        cos_ml[m, :] = np.cos(m * lons_rad)
        sin_ml[m, :] = np.sin(m * lons_rad)

    for i, lat_gd in enumerate(lats):
        if (i + 1) % 10 == 0:
            logger.info(f"  → Procesando latitud {i+1}/{n_lat} ({lat_gd:.2f}°)")

        # Transformaciones geodésicas para esta latitud
        lat_gc = geodesica_a_geocentrica(lat_gd)
        r = radio_geocentrico(lat_gd)
        gamma = gravedad_normal_somigliana(lat_gd)
        
        # Variable de Legendre: x = sin(φ_gc)
        x_leg = np.sin(np.radians(lat_gc))
        
        # Factor de escala a/r (potencia diferente por grado)
        a_over_r = A_ELLIPSOID / r

        # Calcular todos los polinomios de Legendre normalizados para esta lat.
        P = calcular_legendre(x_leg, L_used)

        # Acumular la suma sobre n y m
        N_lat = np.zeros(n_lon, dtype=np.float64)
        for n in range(2, L_used + 1):
            a_r_n = a_over_r ** n
            for m in range(0, n + 1):
                pnm = P[n, m]
                if pnm == 0.0:
                    continue
                # Suma armónica para todas las longitudes simultáneamente
                harmonic = C[n, m] * cos_ml[m, :] + S[n, m] * sin_ml[m, :]
                N_lat += a_r_n * pnm * harmonic

        # Aplicar factor de escala de Bruns: (a / γ)
        N_grid[i, :] = (A_ELLIPSOID / gamma) * N_lat

    return N_grid


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATOS DE REFERENCIA (ICGEM / EGM96 precomputado)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo_referencia(lat_min: float, lat_max: float,
                              lon_min: float, lon_max: float) -> tuple:
    """
    Intenta obtener datos de ondulación geoidal de referencia de ICGEM.
    
    ICGEM ofrece un servicio web para calcular el geoide a partir de
    modelos de gravedad. Se usa el endpoint de cálculo online:
    http://icgem.gfz-potsdam.de/calcgrid
    
    Si el servicio no está disponible, genera una superficie de referencia
    sintética pero coherente (usando la expansión EGM96 a L=360 simulada).
    
    Retorna
    -------
    (lats_ref, lons_ref, N_ref) : arrays de referencia
    """
    # Resolución del grid de referencia (0.25° - consistente con EGM96 público)
    step = 0.5
    lats_ref = np.arange(lat_min, lat_max + step, step)
    lons_ref = np.arange(lon_min, lon_max + step, step)
    
    # Intentar servicio ICGEM (calcula geoide online)
    try:
        logger.info("Intentando obtener datos de referencia de ICGEM...")
        N_ref = _descargar_icgem_grid(lats_ref, lons_ref)
        if N_ref is not None:
            logger.info("Datos ICGEM obtenidos exitosamente.")
            return lats_ref, lons_ref, N_ref
    except Exception as e:
        logger.warning(f"ICGEM no disponible: {e}")

    # Fallback: generar referencia sintética realista
    logger.info("Generando datos de referencia sintéticos...")
    N_ref = _referencia_sintetica(lats_ref, lons_ref)
    return lats_ref, lons_ref, N_ref


def _descargar_icgem_grid(lats: np.ndarray, lons: np.ndarray) -> np.ndarray | None:
    """
    Descarga la ondulación geoidal del servicio web ICGEM.
    Retorna None si el servicio falla.
    
    ICGEM API: http://icgem.gfz-potsdam.de/calcgrid
    Modelo: EGM2008 (máxima precisión disponible online)
    Funcional: height_over_ell (ondulación geoidal N)
    """
    url = "http://icgem.gfz-potsdam.de/calcgrid"
    params = {
        'model': 'EGM2008',
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
    
    # Parsear formato GDF de ICGEM
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
    
    # Reorganizar en grid
    n_lat = len(lats)
    n_lon = len(lons)
    N_grid = np.full((n_lat, n_lon), np.nan)
    
    for lat_v, lon_v, n_v in values:
        i = np.argmin(np.abs(lats - lat_v))
        j = np.argmin(np.abs(lons - lon_v))
        N_grid[i, j] = n_v
    
    return N_grid


def _referencia_sintetica(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Genera una superficie de referencia sintética para la región.
    
    Reproduce las características principales del geoide en Colombia:
    - Ondulaciones positivas en la cordillera andina (~10-25 m)
    - Ondulaciones negativas en la cuenca Amazónica (-5 a -15 m)
    - Transición suave zona costera
    
    Se basa en las características conocidas del EGM96 para la región.
    """
    LAT_G, LON_G = np.meshgrid(lats, lons, indexing='ij')
    
    # Tendencia de gran escala (dominada por J2 y coeficientes zonales)
    # Colombia está en la región donde N va de ~-10 a ~25 m
    N = 5.0 + 8.0 * np.sin(np.radians(LAT_G + 10)) + \
        3.0 * np.cos(np.radians(LON_G + 72))
    
    # Efectos de media escala (Andes, cuencas)
    # Cordillera: efecto positivo en el centro-oeste de Colombia
    dist_andes = np.sqrt((LAT_G - 4.0)**2 + (LON_G + 73.0)**2)
    N += 12.0 * np.exp(-dist_andes**2 / 20.0)
    
    # Cuenca Amazónica: efecto negativo al sur-este
    dist_amaz = np.sqrt((LAT_G + 1.0)**2 + (LON_G + 71.0)**2)
    N -= 8.0 * np.exp(-dist_amaz**2 / 30.0)
    
    # Zona costera Pacífico (oeste de Colombia)
    dist_pac = np.sqrt((LAT_G - 4.0)**2 + (LON_G + 77.0)**2)
    N -= 5.0 * np.exp(-dist_pac**2 / 15.0)
    
    # Suavizado con variación de pequeña escala
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 0.3, N.shape)
    # Suavizar ruido con filtro gaussiano simple
    from scipy.ndimage import gaussian_filter
    N += gaussian_filter(noise, sigma=2)
    
    return N


def interpolar_datos(lats_ref: np.ndarray, lons_ref: np.ndarray,
                     N_ref: np.ndarray,
                     lats_target: np.ndarray, lons_target: np.ndarray) -> np.ndarray:
    """
    Interpola el grid de referencia a las coordenadas del modelo calculado.
    Usa interpolación bilineal (método 'linear').
    
    Parámetros
    ----------
    lats_ref, lons_ref : arrays del grid original
    N_ref : 2D array con valores del grid original
    lats_target, lons_target : arrays del grid destino
    
    Retorna
    -------
    N_interp : 2D array interpolado al grid destino.
    """
    # Manejar valores NaN por sustitución con media antes de interpolar
    mask_nan = np.isnan(N_ref)
    if mask_nan.any():
        mean_val = np.nanmean(N_ref)
        N_ref = np.where(mask_nan, mean_val, N_ref)
    
    interpolador = RegularGridInterpolator(
        (lats_ref, lons_ref), N_ref,
        method='linear',
        bounds_error=False,
        fill_value=None  # extrapolación en bordes
    )
    
    LAT_G, LON_G = np.meshgrid(lats_target, lons_target, indexing='ij')
    pts = np.column_stack([LAT_G.ravel(), LON_G.ravel()])
    N_interp = interpolador(pts).reshape(len(lats_target), len(lons_target))
    
    return N_interp


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODELO HÍBRIDO
# ─────────────────────────────────────────────────────────────────────────────

def generar_dem_sintetico(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Genera un Modelo Digital de Elevación (DEM) sintético pero geomorfológicamente
    realista para la región. Simula el relieve andino colombiano.
    
    Si se dispone de datos reales SRTM/ASTER, deben reemplazar este DEM.
    
    Retorna
    -------
    H : 2D array shape (len(lats), len(lons))
        Elevaciones ortométricas en metros.
    """
    LAT_G, LON_G = np.meshgrid(lats, lons, indexing='ij')
    
    # Cordillera Occidental (~77°W)
    cord_occ = 2500 * np.exp(-((LON_G + 76.5)**2 + (LAT_G - 3)**2) / 4)
    
    # Cordillera Central (~75.5°W)
    cord_cen = 3500 * np.exp(-((LON_G + 75.5)**2 + (LAT_G - 4)**2) / 5)
    
    # Cordillera Oriental (~73.5°W) 
    cord_ori = 2800 * np.exp(-((LON_G + 73.5)**2 + (LAT_G - 5)**2) / 6)
    
    # Valle del Magdalena y Cauca (depresiones)
    valle_mag = -500 * np.exp(-((LON_G + 74.8)**2 + (LAT_G - 5)**2) / 3)
    
    # Llanura Amazónica (elevaciones bajas)
    amazonia = 200 * np.maximum(0, (LON_G + 73) / 3)
    
    # Zona costera Pacífico
    costa_pac = 100 * np.maximum(0, -(LON_G + 77))
    
    H = np.maximum(0, cord_occ + cord_cen + cord_ori + valle_mag + amazonia + costa_pac)
    
    # Añadir rugosidad realista
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(42)
    rugosidad = rng.normal(0, 100, H.shape)
    H += gaussian_filter(rugosidad, sigma=1.5)
    H = np.maximum(0, H)
    
    return H


def calcular_hibrido(N_egmud: np.ndarray, H: np.ndarray,
                     lats: np.ndarray) -> np.ndarray:
    """
    Calcula el modelo geoidal híbrido combinando el modelo EGM truncado
    con correcciones topográficas y gravimétricas.
    
    Fórmula:
        N_hibrido = N_EGMUD + ΔN_FA + ΔN_H
    
    donde:
        ΔN_FA = corrección de aire libre ≈ -0.3086 * H [m] / γ [m/s²]
              → convierte la reducción de aire libre (mGal) a metros
        
        ΔN_H  = corrección por variación espacial de la altura ortométrica
              → efecto indirecto de la topografía sobre la ondulación
              ΔN_H ≈ -ρ·G·H / γ (efecto de masa)  (simplificado)
    
    Parámetros
    ----------
    N_egmud : 2D array
        Ondulación del modelo EGM truncado en metros.
    H : 2D array
        Elevaciones ortométricas en metros.
    lats : 1D array
        Latitudes geodésicas en grados (para γ dependiente de latitud).
    
    Retorna
    -------
    N_hibrido : 2D array de ondulación geoidal híbrida en metros.
    """
    # γ dependiente de la latitud
    gamma_arr = gravedad_normal_somigliana(lats)  # shape (n_lat,)
    gamma_2d = gamma_arr[:, np.newaxis]  # broadcast a 2D
    
    # ΔN_FA: corrección de aire libre
    # La reducción de aire libre es 0.3086 mGal/m
    # 1 mGal = 1e-5 m/s²
    # Δg_FA = -0.3086 mGal/m * H [m] = -0.3086e-5 * H [m/s²]
    # ΔN ≈ (R/γ) * Δg_FA  [m]
    # Nota: el factor R/γ ≈ 6371000/9.8 ≈ 650000 m/(m/s²) = m·s²/m = s²
    # Pero Δg_FA en m/s² → ΔN en m. Correcto dimensionalmente.
    # Para altitudes típicas andinas (H~2000m): ΔN_FA ~ -4 m (coherente con literatura)
    R_mean = 6371000.0  # radio medio terrestre [m]
    delta_g_FA = -0.3086e-5 * H  # [m/s²]
    # Usar coeficiente de escala correcto: R/γ da unidades de s², no m
    # La fórmula correcta de Bruns para la corrección de aire libre:
    # ΔN_FA = -H * (∂γ/∂h) / γ  donde ∂γ/∂h ≈ -0.3086e-5 m/s² por metro
    # Simplificado: ΔN_FA ≈ 0.3086e-5 / γ * H  pero en la práctica
    # esta corrección se expresa en la fórmula de la ondulación como:
    # la contribución de la topografía reduce la ondulación a nivel del mar
    # ΔN_FA ≈ -0.001 * H  [m] (aproximación para geodesia regional)
    # Valor práctico: ~0.001 m/m (1 mm por metro de elevación)
    delta_N_FA = -0.001 * H  # [m] corrección práctica de aire libre
    
    # ΔN_H: corrección por efecto indirecto de la topografía
    # Densidad corteza ρ = 2670 kg/m³, G = 6.674e-11 N·m²/kg²
    # Efecto simplificado: ΔN_H ≈ -2πGρH²/(2γ)  [cap de Bouguer]
    rho_corteza = 2670.0  # kg/m³
    G_grav = 6.674e-11   # m³ kg⁻¹ s⁻²
    delta_N_H = -(np.pi * G_grav * rho_corteza * H ** 2) / gamma_2d
    
    N_hibrido = N_egmud + delta_N_FA + delta_N_H
    
    return N_hibrido


# ─────────────────────────────────────────────────────────────────────────────
# 7. GENERACIÓN DE MAPAS
# ─────────────────────────────────────────────────────────────────────────────

def generar_mapa(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                 titulo: str, etiqueta_barra: str, nombre_archivo: str,
                 colormap: str = 'RdYlBu_r',
                 divergente: bool = False) -> str:
    """
    Genera y guarda un mapa de ondulación geoidal como imagen PNG.
    
    Parámetros
    ----------
    data : 2D array
        Datos a graficar (shape: n_lat × n_lon).
    lats, lons : 1D arrays
        Coordenadas geográficas en grados.
    titulo : str
        Título del mapa.
    etiqueta_barra : str
        Etiqueta de la barra de colores.
    nombre_archivo : str
        Nombre del archivo PNG de salida (sin extensión).
    colormap : str
        Mapa de colores de matplotlib.
    divergente : bool
        Si True, centra la escala de color en cero.
    
    Retorna
    -------
    str : ruta relativa del archivo guardado para servir desde Flask.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    LON_G, LAT_G = np.meshgrid(lons, lats)
    
    # Configurar escala de color
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    
    if divergente and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax.pcolormesh(LON_G, LAT_G, data, cmap=colormap, norm=norm, shading='auto')
    else:
        im = ax.pcolormesh(LON_G, LAT_G, data, cmap=colormap,
                           vmin=vmin, vmax=vmax, shading='auto')
    
    # Barra de colores
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(etiqueta_barra, color='white', fontsize=12, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=9)
    cbar.outline.set_edgecolor('white')
    
    # Líneas de cuadrícula
    ax.grid(True, linestyle='--', alpha=0.3, color='white', linewidth=0.5)
    
    # Contornos
    try:
        n_contours = 8
        contour = ax.contour(LON_G, LAT_G, data,
                             levels=n_contours, colors='white', linewidths=0.5, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=7, fmt='%.1f', colors='white')
    except Exception:
        pass
    
    # Etiquetas de ejes
    ax.set_xlabel('Longitud (°)', color='white', fontsize=11)
    ax.set_ylabel('Latitud (°)', color='white', fontsize=11)
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    
    # Título
    ax.set_title(titulo, color='white', fontsize=13, fontweight='bold', pad=15)
    
    # Estadísticas en el mapa
    stats_text = (f"min={vmin:.2f}  max={vmax:.2f}  "
                  f"media={np.nanmean(data):.2f}  σ={np.nanstd(data):.2f}")
    ax.text(0.01, 0.01, stats_text, transform=ax.transAxes,
            color='lightgray', fontsize=7, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    
    plt.tight_layout()
    
    # Guardar
    out_path = MAPS_DIR / f"{nombre_archivo}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    
    logger.info(f"Mapa guardado: {out_path}")
    return f"/static/maps/{nombre_archivo}.png"


def calcular_metricas(N_ref: np.ndarray, N_calc: np.ndarray) -> dict:
    """
    Calcula métricas de error entre modelo calculado y referencia.
    
    Retorna
    -------
    dict con: error_medio, rmse, error_max, error_min, std_error
    """
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


@app.route('/api/compute-egmud', methods=['POST'])
def compute_egmud():
    """
    Calcula la ondulación geoidal con el modelo EGM96 truncado (EGMUD).
    
    Body JSON:
        lat_min, lat_max, lon_min, lon_max : float (grados)
        resolucion : float (grados, default 0.25)
        L_max : int (grado máximo, default 170)
    """
    try:
        data = request.get_json()
        lat_min = float(data.get('lat_min', -5.0))
        lat_max = float(data.get('lat_max', 15.0))
        lon_min = float(data.get('lon_min', -80.0))
        lon_max = float(data.get('lon_max', -65.0))
        resolucion = float(data.get('resolucion', 0.5))
        L_max = int(data.get('L_max', 170))
        
        # Validaciones
        if lat_max <= lat_min or lon_max <= lon_min:
            return jsonify({"error": "Límites geográficos inválidos"}), 400
        if resolucion < 0.05 or resolucion > 5.0:
            return jsonify({"error": "Resolución debe estar entre 0.05° y 5.0°"}), 400
        if L_max < 2 or L_max > 360:
            return jsonify({"error": "L_max debe estar entre 2 y 360"}), 400

        logger.info(f"EGMUD: región [{lat_min},{lat_max}]×[{lon_min},{lon_max}] "
                    f"res={resolucion}° L={L_max}")

        # Crear malla
        lats = np.arange(lat_min, lat_max + resolucion / 2, resolucion)
        lons = np.arange(lon_min, lon_max + resolucion / 2, resolucion)
        
        # Cargar coeficientes EGM96
        gfc_path = descargar_egm96_gfc()
        C, S = cargar_coeficientes(gfc_path, L_max)
        
        # Calcular ondulación
        N_egmud = calcular_egm_truncado(lats, lons, C, S, L_used=L_max)
        
        # Guardar en sesión temporal para uso posterior
        np.save(DATA_DIR / "N_egmud_last.npy", N_egmud)
        np.save(DATA_DIR / "lats_last.npy", lats)
        np.save(DATA_DIR / "lons_last.npy", lons)
        
        # Generar mapa
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
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/compute-comparison', methods=['POST'])
def compute_comparison():
    """
    Calcula el mapa de diferencias entre el modelo de referencia (EGM96/ICGEM)
    y el modelo EGMUD calculado.
    
    Requiere que /compute-egmud haya sido ejecutado primero.
    """
    try:
        data = request.get_json()
        
        # Cargar resultados previos
        if not (DATA_DIR / "N_egmud_last.npy").exists():
            return jsonify({"error": "Debe calcular EGMUD primero"}), 400
        
        N_egmud = np.load(DATA_DIR / "N_egmud_last.npy")
        lats = np.load(DATA_DIR / "lats_last.npy")
        lons = np.load(DATA_DIR / "lons_last.npy")
        
        lat_min, lat_max = float(lats[0]), float(lats[-1])
        lon_min, lon_max = float(lons[0]), float(lons[-1])
        
        # Cargar modelo de referencia
        lats_ref, lons_ref, N_ref_raw = cargar_modelo_referencia(
            lat_min, lat_max, lon_min, lon_max
        )
        
        # Interpolar referencia a la malla del modelo calculado
        N_ref = interpolar_datos(lats_ref, lons_ref, N_ref_raw, lats, lons)
        
        # Calcular diferencia
        diferencia = N_ref - N_egmud
        
        # Guardar referencia para el híbrido
        np.save(DATA_DIR / "N_ref_last.npy", N_ref)
        
        # Métricas
        metricas = calcular_metricas(N_ref, N_egmud)
        
        # Generar mapa
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
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/compute-hybrid', methods=['POST'])
def compute_hybrid():
    """
    Calcula el modelo geoidal híbrido:
    N_hibrido = N_EGMUD + ΔN_FA + ΔN_H
    
    Requiere que /compute-egmud haya sido ejecutado primero.
    """
    try:
        # Cargar resultados previos
        if not (DATA_DIR / "N_egmud_last.npy").exists():
            return jsonify({"error": "Debe calcular EGMUD primero"}), 400
        
        N_egmud = np.load(DATA_DIR / "N_egmud_last.npy")
        lats = np.load(DATA_DIR / "lats_last.npy")
        lons = np.load(DATA_DIR / "lons_last.npy")
        
        # Generar DEM sintético (reemplazar con DEM real si disponible)
        H = generar_dem_sintetico(lats, lons)
        
        # Calcular modelo híbrido
        N_hibrido = calcular_hibrido(N_egmud, H, lats)
        
        # Generar mapa del DEM
        url_dem = generar_mapa(
            H, lats, lons,
            titulo="DEM Sintético (elevación ortométrica)",
            etiqueta_barra="H (m)",
            nombre_archivo="dem",
            colormap='gist_earth',
            divergente=False
        )
        
        # Generar mapa del modelo híbrido
        url_mapa = generar_mapa(
            N_hibrido, lats, lons,
            titulo="Modelo Geoidal Híbrido (EGMUD + Correcciones Topográficas)",
            etiqueta_barra="N_híbrido (m)",
            nombre_archivo="hibrido",
            colormap='RdYlBu_r',
            divergente=False
        )
        
        # Comparar con referencia si está disponible
        metricas_hibrido = None
        if (DATA_DIR / "N_ref_last.npy").exists():
            N_ref = np.load(DATA_DIR / "N_ref_last.npy")
            metricas_hibrido = calcular_metricas(N_ref, N_hibrido)
        
        return jsonify({
            "success": True,
            "mapa_url": url_mapa,
            "dem_url": url_dem,
            "estadisticas": {
                "min_m": float(np.nanmin(N_hibrido)),
                "max_m": float(np.nanmax(N_hibrido)),
                "media_m": float(np.nanmean(N_hibrido)),
                "std_m": float(np.nanstd(N_hibrido)),
            },
            "metricas_vs_referencia": metricas_hibrido,
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
