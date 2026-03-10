import cv2
import numpy as np
import matplotlib.pyplot as plt


# Fonction histogramme manuel
def calcul_histogramme(image_gray):
    hist = np.zeros(256, dtype=int)
    height, width = image_gray.shape

    for i in range(height):
        for j in range(width):
            intensite = image_gray[i, j]
            hist[intensite] += 1

    return hist


# Charger images
lena = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
lung = cv2.imread("lung.jpg", cv2.IMREAD_GRAYSCALE)

if lena is None or lung is None:
    print("Erreur : image non trouvée")
    exit()

# Redimensionner lung
scale_lung = cv2.resize(lung, (lung.shape[1] // 2, lung.shape[0] // 2))
cv2.imwrite("scale_lung.jpg", scale_lung)

# Calcul histogrammes
hist_lena = calcul_histogramme(lena)
hist_lung = calcul_histogramme(lung)
hist_scale_lung = calcul_histogramme(scale_lung)

# === Affichage 3 subplots ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Histogramme Lena")
plt.bar(range(256), hist_lena)
plt.xlabel("Niveau de gris")
plt.ylabel("Pixels")

plt.subplot(1, 3, 2)
plt.title("Histogramme Lung")
plt.bar(range(256), hist_lung)
plt.xlabel("Niveau de gris")

plt.subplot(1, 3, 3)
plt.title("Histogramme Scale Lung")
plt.bar(range(256), hist_scale_lung)
plt.xlabel("Niveau de gris")

plt.tight_layout()
plt.show()
