import cv2
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# Chargement image
# ===============================

img = cv2.imread("Flower.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Erreur image introuvable")
    exit()


# ===============================
# Sobel (gradients)
# ===============================

Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)


# ===============================
# Magnitude du gradient
# ===============================

magnitude = np.sqrt(Gx**2 + Gy**2)

magnitude = (magnitude / magnitude.max()) * 255
magnitude = magnitude.astype(np.uint8)


# ===============================
# Direction du gradient
# ===============================

direction = np.arctan2(Gy, Gx)


# ===============================
# Seuillage simple
# ===============================


def seuillage_simple(image, seuil):

    result = np.zeros_like(image)

    result[image >= seuil] = 255

    return result


edges_simple = seuillage_simple(magnitude, 80)


# ===============================
# Seuillage par hystérésis
# ===============================


def hysteresis(image, low, high):

    strong = 255
    weak = 75

    result = np.zeros_like(image)

    # pixels forts
    result[image >= high] = strong

    # pixels faibles
    mask = (image >= low) & (image < high)
    result[mask] = weak

    h, w = image.shape

    # propagation
    for i in range(1, h - 1):
        for j in range(1, w - 1):

            if result[i, j] == weak:

                if np.any(result[i - 1 : i + 2, j - 1 : j + 2] == strong):

                    result[i, j] = strong
                else:
                    result[i, j] = 0

    return result


edges_hyst = hysteresis(magnitude, 50, 100)


# ===============================
# AFFICHAGE GRID PROPRE
# ===============================

plt.figure(figsize=(16, 10))


# Ligne 1
plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Image originale")
plt.axis("off")


plt.subplot(2, 3, 2)
plt.imshow(Gx, cmap="gray")
plt.title("Gradient Gx")
plt.axis("off")


plt.subplot(2, 3, 3)
plt.imshow(Gy, cmap="gray")
plt.title("Gradient Gy")
plt.axis("off")


# Ligne 2
plt.subplot(2, 3, 4)
plt.imshow(magnitude, cmap="gray")
plt.title("Magnitude")
plt.axis("off")


plt.subplot(2, 3, 5)
plt.imshow(edges_simple, cmap="gray")
plt.title("Seuillage simple")
plt.axis("off")


plt.subplot(2, 3, 6)
plt.imshow(edges_hyst, cmap="gray")
plt.title("Hystérésis")
plt.axis("off")


plt.tight_layout()
plt.show()
