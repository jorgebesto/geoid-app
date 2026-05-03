# Modelo Geoidal EGM96 — Geodesia Física
## Universidad Distrital Francisco José de Caldas

Aplicación web para el cálculo y visualización de la ondulación geoidal
usando el modelo EGM96 truncado y un modelo híbrido con correcciones topográficas.

---

## Estructura del Proyecto

```
geoid_app/
├── backend.py              ← API REST principal (Flask)
├── requirements.txt        ← Dependencias Python
├── data/                   ← Coeficientes EGM96.gfc (se crea automáticamente)
├── templates/
│   └── index.html          ← Interfaz web principal
└── static/
    ├── css/styles.css      ← Estilos
    ├── js/script.js        ← Lógica frontend
    └── maps/               ← Mapas PNG generados
```

---

## Instalación y Ejecución

### 1. Crear entorno virtual (recomendado)
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# o bien:
.venv\Scripts\activate           # Windows
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Iniciar el servidor
```bash
cd geoid_app/
python backend.py
```

### 4. Abrir en el navegador
```
http://localhost:5000
```

---

## Uso de la Aplicación

### Paso 1 — Región de Estudio
Defina los límites geográficos (lat/lon mín-máx) o use un preset
(Colombia, Cundinamarca, Antioquia). Configure la resolución de la malla.

> ⚠ Resolución < 0.1° con L=170 puede tardar varios minutos.

### Paso 2 — Calcular EGMUD (①)
Calcula la ondulación geoidal N(φ,λ) usando la expansión en armónicos
esféricos hasta grado L. Al primer uso, descargará o generará los
coeficientes EGM96 automáticamente.

### Paso 3 — Comparación (②)
Calcula la diferencia entre la referencia (ICGEM/sintética) y el
modelo calculado. Muestra error medio y RMSE.

### Paso 4 — Modelo Híbrido (③)
Agrega correcciones topográficas:
- ΔN_FA: corrección de aire libre
- ΔN_H:  efecto indirecto de la topografía (cap de Bouguer)

---

## Coeficientes EGM96

El archivo `EGM96.gfc` se descarga automáticamente de ICGEM
(http://icgem.gfz-potsdam.de/) la primera vez.

Si no hay conexión a internet, se genera un archivo sintético
de demostración con las mismas propiedades estadísticas.

Para usar el archivo oficial:
1. Descargue `EGM96.gfc` manualmente de:
   https://icgem.gfz-potsdam.de/tom_longtime
2. Colóquelo en la carpeta `geoid_app/data/`

---

## API REST

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/status` | GET | Estado del servidor |
| `/api/compute-egmud` | POST | Calcular modelo EGM truncado |
| `/api/compute-comparison` | POST | Calcular mapa de diferencias |
| `/api/compute-hybrid` | POST | Calcular modelo híbrido |
| `/api/descargar-coeficientes` | POST | Verificar/descargar EGM96.gfc |

### Ejemplo de request para EGMUD:
```json
POST /api/compute-egmud
{
  "lat_min": -5,
  "lat_max": 15,
  "lon_min": -80,
  "lon_max": -65,
  "resolucion": 0.5,
  "L_max": 170
}
```

---

## Modelo Matemático

### Síntesis Geoidal (Fórmula de Bruns)
```
N(φ,λ) = (a/γ(φ)) · Σ_{n=2}^{L} Σ_{m=0}^{n} (a/r)^n · P̄_nm(sin φ_gc) · (C_nm·cos(mλ) + S_nm·sin(mλ))
```

### Modelo Híbrido
```
N_híbrido = N_EGMUD + ΔN_FA + ΔN_H
ΔN_FA ≈ −(R/γ) · 0.3086×10⁻⁵ · H
ΔN_H  ≈ −πGρH² / γ
```

---

## Notas

- Los resultados son de carácter **académico**.
- Los coeficientes EGM96 son propiedad de NGA/NASA.
- Para escalar a L=360 cambie `L_max` en el formulario (mayor tiempo de cómputo).
- El DEM sintético puede reemplazarse con datos SRTM reales modificando
  la función `generar_dem_sintetico()` en `backend.py`.

---

*Geodesia Física · Prof. Andrés Cárdenas Contreras · Nov 2025*
