# ================= BACKEND: app.py (VERSIÓN MEJORADA NIVEL FINAL) =================
from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv
import requests
import os

app = Flask(__name__)

# ---------------- LOAD COEFFICIENTS ----------------
def load_egm96(file_path, Lmax):
    C, S = {}, {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                n, m = int(parts[0]), int(parts[1])
            except:
                continue
            if n > Lmax:
                continue
            C[(n, m)] = float(parts[2])
            S[(n, m)] = float(parts[3])
    return C, S

# ---------------- EGM TRUNCADO ----------------
def compute_egm(lat, lon, C, S, Lmax):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = np.zeros_like(lat_rad, dtype=float)

    for n in range(2, Lmax + 1):
        for m in range(0, n + 1):
            Pnm = lpmv(m, n, np.sin(lat_rad))
            Cnm = C.get((n, m), 0)
            Snm = S.get((n, m), 0)
            N += (Cnm * np.cos(m * lon_rad) + Snm * np.sin(m * lon_rad)) * Pnm
    return N

# ---------------- DESCARGA DATOS REALES (SIMPLIFICADO) ----------------
def get_reference_model(lat, lon):
    # Simulación de geoide real (puedes reemplazar con ICGEM real)
    return 5 * np.sin(np.radians(lat)) + 3 * np.cos(np.radians(lon))

# ---------------- CARGAR DATOS DE CORRECCIÓN (Lon, Lat, H, AAL) ----------------
def load_corrections(file_path):
    data = np.loadtxt(file_path)
    lon = data[:,0]
    lat = data[:,1]
    H = data[:,2]
    AAL = data[:,3]
    return lon, lat, H, AAL

# ---------------- INTERPOLAR CORRECCIONES A LA MALLA ----------------
def interpolate_to_grid(lon_pts, lat_pts, values, lon_grid, lat_grid):
    from scipy.interpolate import griddata
    points = np.column_stack((lon_pts, lat_pts))
    grid = griddata(points, values, (lon_grid, lat_grid), method='linear')
    return np.nan_to_num(grid)

# ---------------- MODELO HÍBRIDO ----------------
def compute_hybrid(N, H):
    dN_FA = -0.3086 * H
    dN_H = 0.05 * H
    return N + dN_FA + dN_H

# ---------------- MÉTRICAS ----------------
def compute_metrics(real, model):
    diff = real - model
    rmse = np.sqrt(np.mean(diff**2))
    mean_error = np.mean(diff)
    return rmse, mean_error, diff

# ---------------- MAPA ----------------
def plot_map(data, title, filename, extent):
    plt.figure(figsize=(6,5))
    plt.imshow(data, extent=extent, origin='lower')
    plt.colorbar(label='Ondulación (m)')
    plt.title(title)
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.savefig(filename)
    plt.close()

# ---------------- API ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    params = request.json

    lat = np.linspace(params['lat_min'], params['lat_max'], params['res'])
    lon = np.linspace(params['lon_min'], params['lon_max'], params['res'])
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Cargar coeficientes
    C, S = load_egm96('egm96.gfc', 170)

    # Modelo EGMUD
    N_model = compute_egm(lat_grid, lon_grid, C, S, 170)

    # Modelo real (referencia)
    N_real = get_reference_model(lat_grid, lon_grid)

    # Error real
    rmse, mean_error, error = compute_metrics(N_real, N_model)

    # Cargar datos reales de corrección
    lon_pts, lat_pts, H_pts, AAL_pts = load_corrections('correcciones.txt')

    # Interpolar a la malla
    H_grid = interpolate_to_grid(lon_pts, lat_pts, H_pts, lon_grid, lat_grid)
    AAL_grid = interpolate_to_grid(lon_pts, lat_pts, AAL_pts, lon_grid, lat_grid)

    # Modelo híbrido
    N_hybrid = compute_hybrid(N_model, H_grid) + AAL_grid
    H = np.random.uniform(0, 2000, size=N_model.shape)
    N_hybrid = compute_hybrid(N_model, H)

    extent = [params['lon_min'], params['lon_max'], params['lat_min'], params['lat_max']]

    plot_map(N_model, 'EGMUD L=170', 'static/egm.png', extent)
    plot_map(error, 'Error (Real - EGMUD)', 'static/error.png', extent)
    plot_map(N_hybrid, 'Modelo Híbrido', 'static/hybrid.png', extent)

    return jsonify({
        "rmse": float(rmse),
        "mean_error": float(mean_error)
    })
