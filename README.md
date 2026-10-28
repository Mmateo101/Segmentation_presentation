# Redes convolucionales de segmentación

Presentación (~10 min) y notebook que explican, paso a paso, cómo funcionan las redes de segmentación:
de la CNN de clasificación a la FCN, el encoder–decoder y la **U-Net**, con una U-Net real entrenada en **CamVid**.

| Archivo | Qué es |
|---|---|
| `index.html` | La presentación (ábrela en el navegador; clic o → para avanzar, `f` pantalla completa) |
| `Segmentation_UNet.ipynb` | Carga CamVid, entrena la U-Net y una versión sin skip connections, y genera todas las imágenes y números de las diapositivas |
| `assets/generated/` | Salida del notebook que usa la presentación (fotos, máscaras, mapas de características, sprites, `stats.js`) |
| `tools/` | `nb_source.py` es la fuente editable del notebook; `build_notebook.py` reconstruye el `.ipynb` |

## Presentación

Sirve la carpeta (`python -m http.server`) y abre `http://localhost:8000`.
Navegación: clic o →/Espacio para avanzar, ← para regresar, `f` pantalla completa, `#N` en la URL salta a la escena N.
Algunas escenas son interactivas: el deslizador de *sin skips vs U-Net* (8), las probabilidades por píxel al pasar
el cursor (9) y la curva de entrenamiento (10).

| # | Escena |
|---|---|
| 1 | Portada: la foto se vuelve píxeles y cada píxel toma su clase |
| 2 | Clasificación, detección, segmentación semántica y de instancias sobre la misma foto |
| 3 | El problema: una CNN de clasificación reduce 192 × 256 a 12 × 16 y olvida el *dónde* |
| 4 | Idea 1 · FCN: sin capas densas, convolución 1 × 1 y estirar × 16 |
| 5 | Cómo se agranda un mapa: vecino más cercano, bilineal y convolución transpuesta |
| 6 | Idea 2 · encoder–decoder (predicción real sin skips) |
| 7 | Idea 3 · la arquitectura **U-Net**, dibujada con los tamaños reales de la red entrenada |
| 8 | Lo que aportan las skip connections (comparación y mIoU en test) |
| 9 | Softmax por píxel, cross-entropy y IoU |
| 10 | Cómo aprende: la predicción después de cada época |
| 11 | IoU por clase en test y mapa de errores |
| 12 | Más allá: DeepLab, Mask R-CNN y SAM |
| 13 | Resumen y preguntas |

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
