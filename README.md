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
Algunas escenas son interactivas: las probabilidades por píxel al pasar el cursor (6, paso 12),
el deslizador de *sin skips vs U-Net* (7), la curva de entrenamiento (9) y las pestañas de arquitecturas (11).

| # | Escena |
|---|---|
| 1 | Portada: la foto se vuelve píxeles y cada píxel toma su clase |
| 2 | Qué es segmentar: clasificación, detección, segmentación semántica y de instancias |
| 3 | La arquitectura U-Net (con los tamaños reales de la red) y las definiciones necesarias |
| 4 | Paso a paso · encoder: entrada, convolución 3 × 3 con números reales, BatchNorm + ReLU, max pooling, repetir |
| 5 | Paso a paso · decoder: convolución transpuesta, concatenar (skip), convoluciones, repetir |
| 6 | Paso a paso · salida: conv 1 × 1, softmax por píxel (interactivo) y argmax |
| 7 | Para qué sirven las skips (comparación y mIoU en test) |
| 8 | Cómo se entrena: cross-entropy por píxel e IoU |
| 9 | Cómo aprende: la predicción después de cada época |
| 10 | IoU por clase en test y mapa de errores |
| 11 | Otras arquitecturas: FCN, SegNet, DeepLab v3+, Mask R-CNN y SAM (diagrama, pasos, diferencias y tabla comparativa) |
| 12 | Resumen y preguntas |

### Publicar en Vercel

El sitio es estático. `vercel.json` lo configura y `.vercelignore` deja fuera el notebook y las herramientas.
En Vercel: **Add New → Project → Import** este repositorio, preset **Other**, y *Deploy*.

## Notebook

**Google Colab (recomendado):** abre `Segmentation_UNet.ipynb`, elige *GPU* y *Ejecutar todo* (~5 min).
Al final se descarga `presentation_assets.zip`; descomprímelo dentro de `assets/generated/`.

**Local:** córrelo desde la raíz del repositorio (en CPU tarda ~1.5 h por las 40 épocas de cada red):

```bash
pip install tensorflow scipy matplotlib pillow nbconvert
python -m jupyter nbconvert --to notebook --execute Segmentation_UNet.ipynb --inplace
```

**Datos:** CamVid (Brostow, Fauqueur y Cipolla, 2009), en la versión del repositorio
[SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial): 367 fotos de entrenamiento, 101 de validación y
233 de prueba, de 360 × 480, con 11 clases.

**Referencias:** Long, Shelhamer y Darrell (2015), *Fully Convolutional Networks for Semantic Segmentation* ·
Ronneberger, Fischer y Brox (2015), *U-Net* · Badrinarayanan, Kendall y Cipolla (2015), *SegNet* ·
Chen et al. (2017), *DeepLab* · He et al. (2017), *Mask R-CNN* · Kirillov et al. (2023), *Segment Anything*.
