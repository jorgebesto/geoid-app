/**
 * GeoModel EGM96 — Frontend Controller
 * Maneja toda la interacción con la API REST y la visualización de resultados.
 */

'use strict';

// ─────────────────────────────────────────────────────────────────
// Estado de la aplicación
// ─────────────────────────────────────────────────────────────────
const State = {
  egmudCalculado: false,
  comparacionCalculada: false,
  hibridoCalculado: false,
};

// ─────────────────────────────────────────────────────────────────
// Presets geográficos
// ─────────────────────────────────────────────────────────────────
const PRESETS = {
  colombia:      { lat_min: -5,   lat_max: 15,  lon_min: -80, lon_max: -65 },
  cundinamarca:  { lat_min:  3.5, lat_max:  5.5, lon_min: -74.5, lon_max: -73.0 },
  antioquia:     { lat_min:  5.5, lat_max:  8.5, lon_min: -77.0, lon_max: -74.0 },
};

function setPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  document.getElementById('lat-min').value = p.lat_min;
  document.getElementById('lat-max').value = p.lat_max;
  document.getElementById('lon-min').value = p.lon_min;
  document.getElementById('lon-max').value = p.lon_max;
  updateGridPreview();
  addLog(`Preset seleccionado: ${name}`, 'info');
}

// ─────────────────────────────────────────────────────────────────
// Actualizar vista previa de la malla
// ─────────────────────────────────────────────────────────────────
function updateGridPreview() {
  const latMin = parseFloat(document.getElementById('lat-min').value);
  const latMax = parseFloat(document.getElementById('lat-max').value);
  const lonMin = parseFloat(document.getElementById('lon-min').value);
  const lonMax = parseFloat(document.getElementById('lon-max').value);
  const res    = parseFloat(document.getElementById('resolucion').value);

  if (isNaN(latMin) || isNaN(latMax) || isNaN(lonMin) || isNaN(lonMax) || isNaN(res) || res <= 0) {
    document.getElementById('preview-text').textContent = 'Malla: — × — puntos';
    return;
  }

  const nLat = Math.round((latMax - latMin) / res) + 1;
  const nLon = Math.round((lonMax - lonMin) / res) + 1;
  const total = nLat * nLon;
  let warning = '';
  if (total > 50000) warning = '  ⚠ Puede ser lento';
  if (total > 200000) warning = '  ⛔ Muy grande, aumente resolución';

  document.getElementById('preview-text').textContent =
    `Malla: ${nLat} × ${nLon} = ${total.toLocaleString()} puntos${warning}`;
  document.getElementById('badge-L').textContent = document.getElementById('L-max').value;
}

// Escuchar cambios en los inputs
['lat-min','lat-max','lon-min','lon-max','resolucion'].forEach(id => {
  document.getElementById(id).addEventListener('input', updateGridPreview);
});

// ─────────────────────────────────────────────────────────────────
// Utilidades de log
// ─────────────────────────────────────────────────────────────────
function addLog(msg, type = 'info') {
  const body = document.getElementById('log-body');
  const entry = document.createElement('p');
  entry.className = `log-entry log-${type}`;
  const ts = new Date().toLocaleTimeString('es-CO', { hour12: false });
  entry.textContent = `[${ts}] ${msg}`;
  body.appendChild(entry);
  body.scrollTop = body.scrollHeight;
}

function clearLog() {
  document.getElementById('log-body').innerHTML = '';
}

// ─────────────────────────────────────────────────────────────────
// Progress bar
// ─────────────────────────────────────────────────────────────────
let _progressInterval = null;

function showProgress(label) {
  const bar = document.getElementById('progress-bar');
  const fill = document.getElementById('progress-fill');
  const lbl  = document.getElementById('progress-label');
  bar.classList.remove('hidden');
  lbl.textContent = label;
  fill.style.width = '5%';

  let progress = 5;
  _progressInterval = setInterval(() => {
    // Incremento logarítmico: avanza rápido al inicio, lento al final
    progress = Math.min(90, progress + (90 - progress) * 0.03);
    fill.style.width = progress + '%';
  }, 300);
}

function hideProgress() {
  clearInterval(_progressInterval);
  const fill = document.getElementById('progress-fill');
  fill.style.width = '100%';
  setTimeout(() => {
    document.getElementById('progress-bar').classList.add('hidden');
    fill.style.width = '0%';
  }, 500);
}

// ─────────────────────────────────────────────────────────────────
// Helper: obtener parámetros del formulario
// ─────────────────────────────────────────────────────────────────
function getParams() {
  return {
    lat_min:    parseFloat(document.getElementById('lat-min').value),
    lat_max:    parseFloat(document.getElementById('lat-max').value),
    lon_min:    parseFloat(document.getElementById('lon-min').value),
    lon_max:    parseFloat(document.getElementById('lon-max').value),
    resolucion: parseFloat(document.getElementById('resolucion').value),
    L_max:      parseInt(document.getElementById('L-max').value, 10),
  };
}

function validarParams(params) {
  if (params.lat_max <= params.lat_min) return 'Lat máx debe ser mayor que lat mín';
  if (params.lon_max <= params.lon_min) return 'Lon máx debe ser mayor que lon mín';
  if (params.resolucion < 0.05)         return 'Resolución mínima: 0.05°';
  if (params.L_max < 2 || params.L_max > 360) return 'L_max debe estar entre 2 y 360';
  return null;
}

// ─────────────────────────────────────────────────────────────────
// Mostrar imagen en una tarjeta
// ─────────────────────────────────────────────────────────────────
function mostrarMapa(containerId, imageUrl) {
  const container = document.getElementById(containerId);
  const ts = Date.now(); // cache-buster
  container.innerHTML = `<img src="${imageUrl}?t=${ts}" alt="Mapa geoidal" loading="lazy" />`;
}

// ─────────────────────────────────────────────────────────────────
// Formatear número con unidades
// ─────────────────────────────────────────────────────────────────
function fmt(val, decimals = 2, unit = 'm') {
  if (val === null || val === undefined) return '—';
  return `${parseFloat(val).toFixed(decimals)} ${unit}`;
}

// ─────────────────────────────────────────────────────────────────
// ACCIÓN 1: Calcular EGMUD
// ─────────────────────────────────────────────────────────────────
async function calcularEGMUD() {
  const params = getParams();
  const err = validarParams(params);
  if (err) { addLog(`Error: ${err}`, 'error'); return; }

  const btn = document.getElementById('btn-egmud');
  btn.disabled = true;
  btn.classList.add('loading');
  State.egmudCalculado = false;
  State.comparacionCalculada = false;
  State.hibridoCalculado = false;

  // Resetear tarjetas siguientes
  ['btn-compare','btn-hybrid'].forEach(id => {
    document.getElementById(id).disabled = true;
  });

  showProgress('Calculando ondulación geoidal EGM96…');
  addLog(`Iniciando EGMUD: L=${params.L_max}, resolución=${params.resolucion}°`, 'info');
  addLog(`Región: lat[${params.lat_min}, ${params.lat_max}] lon[${params.lon_min}, ${params.lon_max}]`, 'info');

  try {
    const resp = await fetch('/api/compute-egmud', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = await resp.json();
    hideProgress();

    if (!resp.ok || data.error) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }

    // Mostrar mapa
    mostrarMapa('container-egmud', data.mapa_url);

    // Estadísticas
    const st = data.estadisticas;
    document.getElementById('egmud-min').textContent  = fmt(st.min_m);
    document.getElementById('egmud-max').textContent  = fmt(st.max_m);
    document.getElementById('egmud-mean').textContent = fmt(st.media_m);
    document.getElementById('egmud-std').textContent  = fmt(st.std_m);
    document.getElementById('stats-egmud').style.display = 'flex';
    document.getElementById('badge-egmud-pts').textContent =
      `${data.n_lat}×${data.n_lon} pts`;

    State.egmudCalculado = true;
    document.getElementById('btn-compare').disabled = false;
    document.getElementById('btn-hybrid').disabled  = false;

    addLog(`✓ EGMUD completado. Malla: ${data.n_lat}×${data.n_lon}. Media N = ${fmt(st.media_m)}`, 'ok');

  } catch (e) {
    hideProgress();
    addLog(`✗ Error en EGMUD: ${e.message}`, 'error');
    console.error(e);
  }

  btn.disabled = false;
  btn.classList.remove('loading');
}

// ─────────────────────────────────────────────────────────────────
// ACCIÓN 2: Calcular Comparación
// ─────────────────────────────────────────────────────────────────
async function calcularComparacion() {
  if (!State.egmudCalculado) {
    addLog('Calcule primero el modelo EGMUD.', 'warn'); return;
  }

  const btn = document.getElementById('btn-compare');
  btn.disabled = true;
  btn.classList.add('loading');

  showProgress('Obteniendo datos de referencia y calculando error…');
  addLog('Iniciando comparación con referencia EGM96/ICGEM…', 'info');

  try {
    const resp = await fetch('/api/compute-comparison', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const data = await resp.json();
    hideProgress();

    if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);

    mostrarMapa('container-compare', data.mapa_url);

    const m = data.metricas;
    document.getElementById('cmp-mean').textContent = fmt(m.error_medio);
    document.getElementById('cmp-rmse').textContent = fmt(m.rmse);
    document.getElementById('cmp-max').textContent  = fmt(m.error_max);
    document.getElementById('cmp-min').textContent  = fmt(m.error_min);
    document.getElementById('cmp-n').textContent    = m.n_puntos.toLocaleString();
    document.getElementById('stats-compare').style.display = 'flex';

    State.comparacionCalculada = true;
    addLog(`✓ Comparación completada. RMSE = ${fmt(m.rmse)}, Error medio = ${fmt(m.error_medio)}`, 'ok');

  } catch (e) {
    hideProgress();
    addLog(`✗ Error en comparación: ${e.message}`, 'error');
    console.error(e);
  }

  btn.disabled = false;
  btn.classList.remove('loading');
}

// ─────────────────────────────────────────────────────────────────
// ACCIÓN 3: Calcular Modelo Híbrido
// ─────────────────────────────────────────────────────────────────
async function calcularHibrido() {
  if (!State.egmudCalculado) {
    addLog('Calcule primero el modelo EGMUD.', 'warn'); return;
  }

  const btn = document.getElementById('btn-hybrid');
  btn.disabled = true;
  btn.classList.add('loading');

  showProgress('Calculando correcciones topográficas y modelo híbrido…');
  addLog('Iniciando modelo híbrido: EGMUD + ΔN_FA + ΔN_H…', 'info');

  try {
    const resp = await fetch('/api/compute-hybrid', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const data = await resp.json();
    hideProgress();

    if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);

    mostrarMapa('container-hybrid', data.mapa_url);

    const st = data.estadisticas;
    document.getElementById('hyb-min').textContent  = fmt(st.min_m);
    document.getElementById('hyb-max').textContent  = fmt(st.max_m);
    document.getElementById('hyb-mean').textContent = fmt(st.media_m);

    if (data.metricas_vs_referencia) {
      document.getElementById('hyb-rmse').textContent = fmt(data.metricas_vs_referencia.rmse);
    }
    document.getElementById('stats-hybrid').style.display = 'flex';

    // Mostrar DEM
    if (data.dem_url) {
      mostrarMapa('container-dem', data.dem_url);
      document.getElementById('dem-panel').style.display = 'block';
    }

    State.hibridoCalculado = true;
    addLog(`✓ Modelo híbrido completado. Media N = ${fmt(st.media_m)}`, 'ok');

  } catch (e) {
    hideProgress();
    addLog(`✗ Error en modelo híbrido: ${e.message}`, 'error');
    console.error(e);
  }

  btn.disabled = false;
  btn.classList.remove('loading');
}

// ─────────────────────────────────────────────────────────────────
// Verificar/Descargar coeficientes EGM96
// ─────────────────────────────────────────────────────────────────
async function descargarCoeficientes() {
  const btn = document.getElementById('btn-download');
  btn.disabled = true;
  btn.classList.add('loading');
  showProgress('Verificando archivo EGM96.gfc…');
  addLog('Verificando coeficientes EGM96…', 'info');

  try {
    const resp = await fetch('/api/descargar-coeficientes', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}),
    });
    const data = await resp.json();
    hideProgress();

    if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);

    addLog(`✓ EGM96.gfc disponible (${data.size_mb} MB)`, 'ok');

  } catch (e) {
    hideProgress();
    addLog(`✗ Error: ${e.message}`, 'error');
  }

  btn.disabled = false;
  btn.classList.remove('loading');
}

// ─────────────────────────────────────────────────────────────────
// Verificar estado del servidor al cargar
// ─────────────────────────────────────────────────────────────────
async function checkServerStatus() {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();
    if (data.status === 'ok') {
      dot.className = 'status-dot ok';
      if (data.egm96_disponible) {
        text.textContent = `Servidor activo · EGM96.gfc (${data.egm96_size_mb} MB)`;
        addLog(`Servidor activo. EGM96.gfc disponible (${data.egm96_size_mb} MB).`, 'ok');
      } else {
        text.textContent = 'Servidor activo · EGM96.gfc no encontrado';
        addLog('Servidor activo. EGM96.gfc no encontrado — use "Verificar EGM96.gfc".', 'warn');
      }
    }
  } catch (e) {
    dot.className = 'status-dot error';
    text.textContent = 'Servidor no responde';
    addLog('No se pudo conectar al servidor backend.', 'error');
  }
}

// ─────────────────────────────────────────────────────────────────
// Inicialización
// ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  updateGridPreview();
  checkServerStatus();
  addLog('Interfaz lista. Configure la región de estudio y calcule.', 'info');
});
