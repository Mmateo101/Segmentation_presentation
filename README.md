# Redes convolucionales de segmentación

Presentación y notebook que explican, paso a paso, cómo funcionan las redes de segmentación:
la **U-Net** por dentro, operación por operación, con una U-Net real entrenada en **CamVid**, y sus parientes (FCN, SegNet, DeepLab, Mask R-CNN, SAM).

| Archivo | Qué es |
|---|---|
| `index.html` | La presentación (ábrela en el navegador; clic o → para avanzar, `f` pantalla completa) |
| `Segmentation_UNet.ipynb` | Carga CamVid, entrena la U-Net y una versión sin skip connections, y genera todas las imágenes y números de las diapositivas |
| `assets/generated/` | Salida del notebook que usa la presentación (fotos, máscaras, mapas de características, sprites, `stats.js`) |
| `tools/` | `nb_source.py` es la fuente editable del notebook; `build_notebook.py` reconstruye el `.ipynb` |

## Presentación

Sirve la carpeta (`python -m http.server`) y abre `http://localhost:8000`.
Navegación: clic o →/Espacio para avanzar, ← para regresar, `f` pantalla completa, `#N` en la URL salta a la escena N.
Algunas son interactivas: las probabilidades por píxel al pasar el cursor (18), la curva de entrenamiento (21)
y las pestañas de arquitecturas (23). Cada diapositiva de la red muestra un mapa de la U-Net con el paso actual.

| # | Diapositiva |
|---|---|
| 1 | Portada: la foto se vuelve píxeles y cada píxel toma su clase |
| 2 | Qué es segmentar: clasificación, detección, segmentación semántica y de instancias |
| 3 | Solo reducir (detección) vs. reducir y ampliar (U-Net) |
| 4 | La arquitectura U-Net y las definiciones necesarias |
| 5 | Paso 1 · la imagen |
| 6 | Paso 2 · un filtro: la ventanita recorre la foto y pinta un mapa |
| 7 | Paso 3 · 16 ventanitas → 16 mapas |
| 8 | Paso 4 · BatchNorm + ReLU |
| 9 | Paso 5 · la segunda convolución (3 × 3 × 16) |
| 10 | Paso 6 · max pooling sobre cada mapa |
| 11 | Paso 7 · bajar 4 veces |
| 12 | Paso 8 · el cuello de botella |
| 13 | Paso 9 · convolución transpuesta |
| 14 | Paso 10 · concatenar (skip) |
| 15 | Paso 11 · convoluciones del decoder |
| 16 | Paso 12 · subir hasta 192 × 256 |
| 17 | Paso 13 · conv 1 × 1 |
| 18 | Paso 14 · softmax por píxel (interactivo) |
| 19 | Paso 15 · argmax |
| 20 | Paso 16 · qué se entrena y qué no |
| 21 | Cómo aprende: la predicción después de cada época, junto a la máscara real |
| 22 | IoU por clase en test y mapa de errores |
| 23 | Otras arquitecturas: FCN, SegNet, DeepLab v3+, Mask R-CNN y SAM |
| 24 | Resumen y preguntas |

### Publicar en Vercel

El sitio es estático. `vercel.json` lo configura y `.vercelignore` deja fuera el notebook y las herramientas.
En Vercel: **Add New → Project → Import** este repositorio, preset **Other**, y *Deploy*.

## Notebook

Por defecto **no entrena**: carga la U-Net ya entrenada de `models/` y genera todas las imágenes en ~1 min
(`ENTRENAR = False`, sección 1). Para entrenar las dos redes desde cero, pon `ENTRENAR = True`
(~5 min con GPU en Colab, ~80 min en CPU); los pesos nuevos se guardan en `models/`.

**Google Colab:** abre `Segmentation_UNet.ipynb` y *Ejecutar todo*. Si no encuentra `models/`, descarga los pesos de
este repositorio. Al final se descarga `presentation_assets.zip`; descomprímelo dentro de `assets/generated/`.

**Local:** desde la raíz del repositorio (escribe directo en `assets/generated/`):

```bash
pip install tensorflow scipy matplotlib pillow nbconvert
python -m jupyter nbconvert --to notebook --execute Segmentation_UNet.ipynb --inplace
```

| `models/` | Qué es |
|---|---|
| `unet.keras`, `encoder_decoder.keras` | Pesos de la U-Net y de la versión sin skips (40 épocas, mejores pesos en validación) |
| `*.json` | Pérdida y mIoU de validación por época |
| `*_frames.npz` | La predicción sobre la foto principal después de cada época |

Resultado en test (233 fotos): **mIoU 0.43** con U-Net contra **0.39** sin skips.

**Datos:** CamVid (Brostow, Fauqueur y Cipolla, 2009), en la versión del repositorio
[SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial): 367 fotos de entrenamiento, 101 de validación y
233 de prueba, de 360 × 480, con 11 clases.

**Referencias:** Long, Shelhamer y Darrell (2015), *Fully Convolutional Networks for Semantic Segmentation* ·
Ronneberger, Fischer y Brox (2015), *U-Net* · Badrinarayanan, Kendall y Cipolla (2015), *SegNet* ·
Chen et al. (2017), *DeepLab* · He et al. (2017), *Mask R-CNN* · Kirillov et al. (2023), *Segment Anything*.
