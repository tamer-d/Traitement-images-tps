import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Chargement image
# ===============================

img = cv2.imread("Objects.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Erreur image introuvable")
    exit()


# ===============================
# PARTIE 1 : Gradient + Orientation
# ===============================

Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

magnitude = np.sqrt(Gx**2 + Gy**2)

# normalisation pour affichage
magnitude = (magnitude / magnitude.max()) * 255

direction = np.arctan2(Gy, Gx) * 180 / np.pi
direction[direction < 0] += 180


# ===============================
# Quantification direction
# ===============================


def quantize_direction(angle):

    q = np.zeros_like(angle)

    q[(angle >= 0) & (angle < 22.5)] = 0
    q[(angle >= 22.5) & (angle < 67.5)] = 45
    q[(angle >= 67.5) & (angle < 112.5)] = 90
    q[(angle >= 112.5) & (angle < 157.5)] = 135
    q[(angle >= 157.5)] = 0

    return q


dir_q = quantize_direction(direction)


# ===============================
# Non-Maximum Suppression
# ===============================


def non_maximum_suppression(mag, direction):

    h, w = mag.shape
    result = np.zeros((h, w))

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            angle = direction[i, j]

            if angle == 0:
                q = mag[i, j + 1]
                r = mag[i, j - 1]

            elif angle == 90:
                q = mag[i + 1, j]
                r = mag[i - 1, j]

            elif angle == 45:
                q = mag[i - 1, j + 1]
                r = mag[i + 1, j - 1]

            elif angle == 135:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            if mag[i, j] >= q and mag[i, j] >= r:
                result[i, j] = mag[i, j]
            else:
                result[i, j] = 0

    return result


nms = non_maximum_suppression(magnitude, dir_q)


# ===============================
# Hystérésis
# ===============================


def hysteresis(img, low, high):

    strong = 255
    weak = 75

    res = np.zeros_like(img)

    res[img >= high] = strong
    res[(img >= low) & (img < high)] = weak

    h, w = img.shape

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            if res[i, j] == weak:

                if np.any(res[i - 1 : i + 2, j - 1 : j + 2] == strong):
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    return res


edges = hysteresis(nms, 50, 100)


# ===============================
# PARTIE 2 : Seuillage
# ===============================

# Seuillage simple


def seuil_simple(image, T):

    result = np.zeros_like(image)
    result[image > T] = 255

    return result


binary_simple = seuil_simple(img, 120)


# ===============================
# OTSU (manuel)
# ===============================


def otsu(image):

    hist, _ = np.histogram(image.flatten(), 256, [0, 256])

    total = image.size
    sum_total = np.sum(np.arange(256) * hist)

    sumB = 0
    wB = 0
    var_max = 0
    threshold = 0

    for t in range(256):

        wB += hist[t]
        if wB == 0:
            continue

        wF = total - wB
        if wF == 0:
            break

        sumB += t * hist[t]

        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        var_between = wB * wF * (mB - mF) ** 2

        if var_between > var_max:
            var_max = var_between
            threshold = t

    result = np.zeros_like(image)
    result[image > threshold] = 255

    return result, threshold


binary_otsu, T_otsu = otsu(img)


# ===============================
# PARTIE 3 : Composantes connexes
# ===============================


def connected_components(binary):

    h, w = binary.shape
    labels = np.zeros((h, w), dtype=int)

    label = 1

    for i in range(h):
        for j in range(w):

            if binary[i, j] == 255 and labels[i, j] == 0:

                stack = [(i, j)]

                while stack:

                    x, y = stack.pop()

                    if labels[x, y] == 0:

                        labels[x, y] = label

                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:

                                nx = x + dx
                                ny = y + dy

                                if 0 <= nx < h and 0 <= ny < w:

                                    if binary[nx, ny] == 255 and labels[nx, ny] == 0:
                                        stack.append((nx, ny))

                label += 1

    return labels


labels = connected_components(binary_otsu)


# ===============================
# AFFICHAGE FINAL
# ===============================

plt.figure(figsize=(15, 10))


plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Originale")
plt.axis("off")


plt.subplot(2, 3, 2)
plt.imshow(edges, cmap="gray")
plt.title("Contours (orientation + hyst)")
plt.axis("off")


plt.subplot(2, 3, 3)
plt.imshow(binary_simple, cmap="gray")
plt.title("Seuillage simple")
plt.axis("off")


plt.subplot(2, 3, 4)
plt.imshow(binary_otsu, cmap="gray")
plt.title(f"Otsu (T={T_otsu})")
plt.axis("off")


plt.subplot(2, 3, 5)
plt.imshow(labels, cmap="nipy_spectral")
plt.title("Composantes connexes")
plt.axis("off")


plt.tight_layout()
plt.show()
