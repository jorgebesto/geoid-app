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
  // Store grid data per map container for pixel value lookup
  gridData: {},  // { containerId: { bounds, data, n_lat, n_lon } }
};

// ─────────────────────────────────────────────────────────────────
// Extraer bounds automáticamente del archivo de correcciones
// ─────────────────────────────────────────────────────────────────
async function extractBoundsFromCorrections() {
  return new Promise((resolve, reject) => {
    const fileCorr = document.getElementById('file-corr').files[0];
    if (!fileCorr) {
      reject('Se requiere el archivo de correcciones (.csv/.txt) para auto-ajustar la región.');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.trim().split('\n');
      if (lines.length < 2) {
        reject('El archivo de correcciones está vacío o no tiene datos.');
        return;
      }
      let lat_min = 90, lat_max = -90, lon_min = 180, lon_max = -180;
      let validCount = 0;
      
      for (let i = 1; i < lines.length; i++) {
        let line = lines[i].trim();
        if (!line) continue;
        
        let parts;
        if (line.includes(';')) parts = line.split(';');
        else if (line.includes('\t')) parts = line.split('\t');
        else if ((line.match(/,/g) || []).length >= 3) parts = line.split(',');
        else parts = line.split(/\s+/);
        
        if (parts.length >= 2) {
          let lon = parseFloat(parts[0].trim().replace(',', '.'));
          let lat = parseFloat(parts[1].trim().replace(',', '.'));
          
          if (!isNaN(lon) && !isNaN(lat)) {
            if (lon < lon_min) lon_min = lon;
            if (lon > lon_max) lon_max = lon;
            if (lat < lat_min) lat_min = lat;
            if (lat > lat_max) lat_max = lat;
            validCount++;
          }
        }
      }
      if (validCount === 0) {
        reject('No se encontraron coordenadas válidas en el archivo de correcciones.');
      } else {
        // Expandir un margen de 0.05 grados para asegurar que los bordes cubran bien los puntos
        resolve({
          lat_min: Math.floor((lat_min - 0.05) * 100) / 100,
          lat_max: Math.ceil((lat_max + 0.05) * 100) / 100,
          lon_min: Math.floor((lon_min - 0.05) * 100) / 100,
          lon_max: Math.ceil((lon_max + 0.05) * 100) / 100
        });
      }
    };
    reader.onerror = () => reject('Error al leer el archivo de correcciones.');
    reader.readAsText(fileCorr);
  });
}

function updateGridPreview() {
  document.getElementById('badge-L').textContent = document.getElementById('L-max').value;
}

document.getElementById('resolucion').addEventListener('input', updateGridPreview);

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
// Helper: validar resolución y L_max
// ─────────────────────────────────────────────────────────────────
function validarParams(res, L_max) {
  if (res < 0.05) return 'Resolución mínima: 0.05°';
  if (L_max < 2 || L_max > 360) return 'L_max debe estar entre 2 y 360';
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
  const resolucion = parseFloat(document.getElementById('resolucion').value);
  const L_max = parseInt(document.getElementById('L-max').value, 10);
  
  const err = validarParams(resolucion, L_max);
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

  const fileGfc = document.getElementById('file-gfc').files[0];
  if (!fileGfc) {
    addLog('Por favor, seleccione su archivo .gfc', 'error');
    btn.disabled = false;
    btn.classList.remove('loading');
    return;
  }

  showProgress('Leyendo archivo de correcciones para auto-ajustar región…');
  
  let bounds;
  try {
    bounds = await extractBoundsFromCorrections();
  } catch (e) {
    addLog(e, 'error');
    btn.disabled = false;
    btn.classList.remove('loading');
    hideProgress();
    return;
  }
  
  const params = {
    lat_min: bounds.lat_min,
    lat_max: bounds.lat_max,
    lon_min: bounds.lon_min,
    lon_max: bounds.lon_max,
    resolucion: resolucion,
    L_max: L_max
  };

  showProgress('Calculando ondulación geoidal EGMUD…');
  addLog(`Región auto-ajustada: lat[${params.lat_min}, ${params.lat_max}] lon[${params.lon_min}, ${params.lon_max}]`, 'info');
  addLog(`Iniciando EGMUD: L=${params.L_max}, resolución=${params.resolucion}°`, 'info');

  const formData = new FormData();
  Object.keys(params).forEach(key => formData.append(key, params[key]));
  formData.append('file_gfc', fileGfc);

  try {
    const resp = await fetch('/api/compute-egmud', {
      method: 'POST',
      body: formData,
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

    // Store grid data for pixel tooltip
    if (data.grid_data && data.grid_bounds) {
      State.gridData['container-egmud'] = { bounds: data.grid_bounds, data: data.grid_data };
    }

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

    // Store grid data for pixel tooltip
    if (data.grid_data && data.grid_bounds) {
      State.gridData['container-compare'] = { bounds: data.grid_bounds, data: data.grid_data };
    }

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

  const fileCorr = document.getElementById('file-corr').files[0];
  if (!fileCorr) {
    addLog('Por favor, seleccione el archivo de correcciones', 'error');
    btn.disabled = false;
    btn.classList.remove('loading');
    hideProgress();
    return;
  }

  const formData = new FormData();
  formData.append('file_corr', fileCorr);

  try {
    const resp = await fetch('/api/compute-hybrid', {
      method: 'POST',
      body: formData,
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

    // Store grid data for pixel tooltip
    if (data.grid_data && data.grid_bounds) {
      State.gridData['container-hybrid'] = { bounds: data.grid_bounds, data: data.grid_data };
    }

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
// Modal: expandir mapa al hacer clic en botón ⛶
// ─────────────────────────────────────────────────────────────────
function fullscreenMap(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const img = container.querySelector('img');
  if (!img) return;
  openModal(img.src.split('?')[0]);
}
function openModal(imgSrc) {
  const modal = document.getElementById('image-modal');
  document.getElementById('modal-img').src = imgSrc;
  modal.classList.add('active');
}
function closeModal() {
  document.getElementById('image-modal').classList.remove('active');
}
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

// ─────────────────────────────────────────────────────────────────
// Tooltip: mostrar valor en metros al hacer clic sobre un mapa
// ─────────────────────────────────────────────────────────────────
function setupMapPixelTooltip() {
  const tooltip = document.getElementById('pixel-tooltip');

  // Use event delegation on document for dynamically added images
  document.addEventListener('click', (e) => {
    const img = e.target;
    if (!img || img.tagName !== 'IMG') return;
    const container = img.closest('.map-container');
    if (!container) return;

    // Find which map this container belongs to
    const containerId = container.id;
    const gridInfo = State.gridData[containerId];

    if (!gridInfo || !gridInfo.data) {
      tooltip.innerHTML = 'Sin datos de grilla';
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top = (e.clientY - 30) + 'px';
      setTimeout(() => { tooltip.style.display = 'none'; }, 2000);
      return;
    }

    // Map pixel position to grid indices
    // matplotlib images have the plot area within a border (axes area)
    // We approximate: the axes area is roughly 10-88% horizontally, 8-85% vertically
    const rect = img.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const relY = (e.clientY - rect.top) / rect.height;

    // Approximate matplotlib axes bounds within image
    const axLeft = 0.12, axRight = 0.82, axTop = 0.08, axBottom = 0.88;
    
    if (relX < axLeft || relX > axRight || relY < axTop || relY > axBottom) {
      tooltip.style.display = 'none';
      return; // Click outside plot area
    }

    // Normalize within axes area
    const normX = (relX - axLeft) / (axRight - axLeft);
    const normY = (relY - axTop) / (axBottom - axTop);

    const b = gridInfo.bounds;
    const n_lat = gridInfo.data.length;
    const n_lon = gridInfo.data[0].length;

    // Map to grid indices (Y is inverted in image: top=lat_max, bottom=lat_min)
    const iRow = Math.round(normY * (n_lat - 1));  // top=0=lat_max
    const iCol = Math.round(normX * (n_lon - 1));

    if (iRow < 0 || iRow >= n_lat || iCol < 0 || iCol >= n_lon) {
      tooltip.style.display = 'none';
      return;
    }

    // In matplotlib with origin='lower', row 0 = lat_min (bottom)
    // But in screen, top = row 0. So we need to flip.
    const rowFlipped = n_lat - 1 - iRow;
    const value = gridInfo.data[rowFlipped] ? gridInfo.data[rowFlipped][iCol] : null;

    // Calculate lat/lon
    const lat = b.lat_min + (rowFlipped / (n_lat - 1)) * (b.lat_max - b.lat_min);
    const lon = b.lon_min + (iCol / (n_lon - 1)) * (b.lon_max - b.lon_min);

    if (value !== null && !isNaN(value)) {
      tooltip.innerHTML = `<b>${value.toFixed(3)} m</b><br>Lat: ${lat.toFixed(3)}°  Lon: ${lon.toFixed(3)}°`;
    } else {
      tooltip.innerHTML = 'Sin dato';
    }

    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY - 30) + 'px';
    setTimeout(() => { tooltip.style.display = 'none'; }, 4000);
  });
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
  setupMapPixelTooltip();
  addLog('Interfaz lista. Cargue sus archivos y calcule.', 'info');
});
