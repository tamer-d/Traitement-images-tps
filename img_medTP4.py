import cv2
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# Chargement image bruitée
# ===============================

img = cv2.imread("noisy_lena.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Erreur image introuvable")
    exit()


# ===============================
# Fonction convolution manuelle
# ===============================


def convolution(image, kernel):

    h, w = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    result = np.zeros_like(image)

    for i in range(h):
        for j in range(w):

            region = padded[i : i + kh, j : j + kw]

            result[i, j] = np.sum(region * kernel)

    return result


# ===============================
# Filtre moyenneur
# ===============================


def filtre_moyenneur(size):

    return np.ones((size, size)) / (size * size)


# Application filtres moyenneurs

kernel3 = filtre_moyenneur(3)
kernel5 = filtre_moyenneur(5)
kernel7 = filtre_moyenneur(7)

img_mean3 = convolution(img, kernel3)
img_mean5 = convolution(img, kernel5)
img_mean7 = convolution(img, kernel7)


# ===============================
# Filtre gaussien
# ===============================


def filtre_gaussien(size, sigma):

    k = size // 2

    x = np.arange(-k, k + 1)
    y = np.arange(-k, k + 1)

    X, Y = np.meshgrid(x, y)

    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    kernel = kernel / np.sum(kernel)

    return kernel


# Application gaussien

gauss3 = filtre_gaussien(3, 1)
gauss5 = filtre_gaussien(5, 1)
gauss7 = filtre_gaussien(7, 2)

img_gauss3 = convolution(img, gauss3)
img_gauss5 = convolution(img, gauss5)
img_gauss7 = convolution(img, gauss7)


# ===============================
# Filtre médian
# ===============================


def filtre_median(image, size):

    h, w = image.shape

    pad = size // 2

    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="constant")

    result = np.zeros_like(image)

    for i in range(h):
        for j in range(w):

            region = padded[i : i + size, j : j + size]

            result[i, j] = np.median(region)

    return result


# Application filtre médian

img_med3 = filtre_median(img, 3)
img_med5 = filtre_median(img, 5)
img_med7 = filtre_median(img, 7)


import cv2
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# Chargement image bruitée
# ===============================

img = cv2.imread("noisy_lena.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Erreur image introuvable")
    exit()


# ===============================
# Fonction convolution manuelle
# ===============================


def convolution(image, kernel):

    h, w = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    result = np.zeros_like(image)

    for i in range(h):
        for j in range(w):

            region = padded[i : i + kh, j : j + kw]

            result[i, j] = np.sum(region * kernel)

    return result


# ===============================
# Filtre moyenneur
# ===============================


def filtre_moyenneur(size):

    return np.ones((size, size)) / (size * size)


# Application filtres moyenneurs

kernel3 = filtre_moyenneur(3)
kernel5 = filtre_moyenneur(5)
kernel7 = filtre_moyenneur(7)

img_mean3 = convolution(img, kernel3)
img_mean5 = convolution(img, kernel5)
img_mean7 = convolution(img, kernel7)


# ===============================
# Filtre gaussien
# ===============================


def filtre_gaussien(size, sigma):

    k = size // 2

    x = np.arange(-k, k + 1)
    y = np.arange(-k, k + 1)

    X, Y = np.meshgrid(x, y)

    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    kernel = kernel / np.sum(kernel)

    return kernel


# Application gaussien

gauss3 = filtre_gaussien(3, 1)
gauss5 = filtre_gaussien(5, 1)
gauss7 = filtre_gaussien(7, 2)

img_gauss3 = convolution(img, gauss3)
img_gauss5 = convolution(img, gauss5)
img_gauss7 = convolution(img, gauss7)


# ===============================
# Filtre médian
# ===============================


def filtre_median(image, size):

    h, w = image.shape

    pad = size // 2

    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="constant")

    result = np.zeros_like(image)

    for i in range(h):
        for j in range(w):

            region = padded[i : i + size, j : j + size]

            result[i, j] = np.median(region)

    return result


# Application filtre médian

img_med3 = filtre_median(img, 3)
img_med5 = filtre_median(img, 5)
img_med7 = filtre_median(img, 7)


# ===============================
# AFFICHAGE GRID 6×3 PROPRE
# ===============================

plt.figure(figsize=(120, 90))

plt.subplot(6, 3, 2)
plt.axis("off")

plt.subplot(6, 3, 3)
plt.axis("off")


# ===== Ligne 2 : Moyenneur =====

plt.subplot(6, 3, 4)
plt.imshow(img_mean3, cmap="gray")
plt.title("Moyenneur 3x3")
plt.axis("off")


plt.subplot(6, 3, 5)
plt.imshow(img_mean5, cmap="gray")
plt.title("Moyenneur 5x5")
plt.axis("off")


plt.subplot(6, 3, 6)
plt.imshow(img_mean7, cmap="gray")
plt.title("Moyenneur 7x7")
plt.axis("off")


# ===== Ligne 3 : Gaussien =====

plt.subplot(6, 3, 7)
plt.imshow(img_gauss3, cmap="gray")
plt.title("Gaussien 3x3 σ=1")
plt.axis("off")


plt.subplot(6, 3, 8)
plt.imshow(img_gauss5, cmap="gray")
plt.title("Gaussien 5x5 σ=1")
plt.axis("off")


plt.subplot(6, 3, 9)
plt.imshow(img_gauss7, cmap="gray")
plt.title("Gaussien 7x7 σ=2")
plt.axis("off")


# ===== Ligne 4 : Médian =====

plt.subplot(6, 3, 10)
plt.imshow(img_med3, cmap="gray")
plt.title("Median 3x3")
plt.axis("off")


plt.subplot(6, 3, 11)
plt.imshow(img_med5, cmap="gray")
plt.title("Median 5x5")
plt.axis("off")


plt.subplot(6, 3, 12)
plt.imshow(img_med7, cmap="gray")
plt.title("Median 7x7")
plt.axis("off")


# lignes restantes vides pour lisibilité

for i in range(13, 19):

    plt.subplot(6, 3, i)

    plt.axis("off")


plt.tight_layout()

plt.show()
