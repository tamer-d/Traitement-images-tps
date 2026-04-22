import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# Chargement image
# ==============================

img = cv2.imread("lena.jpg", cv2.IMREAD_COLOR)

if img is None:
    print("Erreur image introuvable")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ==============================
# K-means
# ==============================


def init_centroids(pixels, K):

    indices = np.random.choice(len(pixels), K, replace=False)

    return pixels[indices]


def assign_clusters(pixels, centroids):

    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)

    return np.argmin(distances, axis=1)


def update_centroids(pixels, labels, K):

    new_centroids = []

    for k in range(K):

        cluster = pixels[labels == k]

        if len(cluster) == 0:

            new_centroids.append(pixels[np.random.randint(len(pixels))])

        else:

            new_centroids.append(cluster.mean(axis=0))

    return np.array(new_centroids)


def kmeans(pixels, K, max_iter=20):

    centroids = init_centroids(pixels, K)

    for i in range(max_iter):

        labels = assign_clusters(pixels, centroids)

        new_centroids = update_centroids(pixels, labels, K)

        if np.allclose(centroids, new_centroids):

            break

        centroids = new_centroids

    return centroids, labels


# ==============================
# Quantification K-means
# ==============================


def quantize_kmeans(image, K):

    pixels = image.reshape((-1, 3)).astype(np.float32)

    centroids, labels = kmeans(pixels, K)

    quantized_pixels = centroids[labels]

    quantized_image = quantized_pixels.reshape(image.shape)

    return quantized_image.astype(np.uint8), centroids, labels


# ==============================
# Median-cut
# ==============================


def median_cut(pixels, K):

    boxes = [pixels]

    while len(boxes) < K:

        boxes.sort(key=lambda x: np.ptp(x, axis=0).max(), reverse=True)

        box = boxes.pop(0)

        channel = np.argmax(np.ptp(box, axis=0))

        box = box[box[:, channel].argsort()]

        median = len(box) // 2

        boxes.append(box[:median])
        boxes.append(box[median:])

    centroids = [box.mean(axis=0) for box in boxes]

    return np.array(centroids)


def quantize_mediancut(image, K):

    pixels = image.reshape((-1, 3)).astype(np.float32)

    centroids = median_cut(pixels, K)

    labels = assign_clusters(pixels, centroids)

    quantized_pixels = centroids[labels]

    quantized_image = quantized_pixels.reshape(image.shape)

    return quantized_image.astype(np.uint8), centroids, labels


# ==============================
# Histogramme palette (correct)
# ==============================


def histogramme_clusters(labels, K):

    hist = np.zeros(K)

    for k in range(K):

        hist[k] = np.sum(labels == k)

    hist = hist / hist.sum()

    return hist


# ==============================
# Paramètre
# ==============================

K = 8


# ==============================
# Quantification
# ==============================

img_kmeans, palette_kmeans, labels_kmeans = quantize_kmeans(img_rgb, K)

img_median, palette_median, labels_median = quantize_mediancut(img_rgb, K)


hist_kmeans = histogramme_clusters(labels_kmeans, K)

hist_median = histogramme_clusters(labels_median, K)


# Histogramme original (RGB séparé)

hist_r = np.histogram(img_rgb[:, :, 0], bins=256)[0]
hist_g = np.histogram(img_rgb[:, :, 1], bins=256)[0]
hist_b = np.histogram(img_rgb[:, :, 2], bins=256)[0]


# ==============================
# AFFICHAGE FINAL GRID
# ==============================

plt.figure(figsize=(15, 8))


# ===== Histogramme original =====

plt.subplot(2, 3, 1)

plt.plot(hist_r, color="red")
plt.plot(hist_g, color="green")
plt.plot(hist_b, color="blue")

plt.title("Histogramme image originale RGB")


# ===== Histogramme K-means =====

plt.subplot(2, 3, 2)

plt.bar(range(K), hist_kmeans, color=palette_kmeans / 255)

plt.title("Histogramme K-means")


# ===== Histogramme Median-cut =====

plt.subplot(2, 3, 3)

plt.bar(range(K), hist_median, color=palette_median / 255)

plt.title("Histogramme Median-cut")


# ===== Images =====

plt.subplot(2, 3, 4)

plt.imshow(img_rgb)

plt.title("Image originale")

plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(img_kmeans)

plt.title("Image K-means")

plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(img_median)

plt.title("Image Median-cut")

plt.axis("off")


plt.tight_layout()

plt.show()
