import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("low_cont_xray.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Erreur : image non trouvée")
    exit()


def translation(image, k):
    img_trans = image.astype(np.int16) + k
    img_trans = np.clip(img_trans, 0, 255)
    return img_trans.astype(np.uint8)


def inversion(image):
    return 255 - image


def expansion_dynamique(image):
    Imin = np.min(image)
    Imax = np.max(image)

    image = image.astype(np.float32)
    img_exp = 255 * (image - Imin) / (Imax - Imin)

    return img_exp.astype(np.uint8)


def egalisation_histogramme(image):
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]

    M, N = image.shape
    L = 256

    cdf_norm = np.round((cdf - cdf_min) / (M * N - cdf_min) * (L - 1))
    cdf_norm = cdf_norm.astype(np.uint8)

    img_eq = cdf_norm[image]
    return img_eq


# ===== Traitements =====
img_plus = translation(img, 70)
img_moins = translation(img, -70)
img_inv = inversion(img)
img_exp = expansion_dynamique(img)
img_eq = egalisation_histogramme(img)


# ===== Liste des images à afficher =====
images = [
    (img, "Originale"),
    (img_eq, " egalisation"),
    (img_exp, "Expansion dynamique"),
    (img_plus, "Translation +70"),
    (img_moins, "Translation -70"),
    (img_inv, "Inversion"),
]


# ===== Création d'une seule fenêtre =====
plt.figure(figsize=(15, 10))

for i, (image, titre) in enumerate(images):

    # Image
    plt.subplot(len(images), 2, 2 * i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(titre)
    plt.axis("off")

    # Histogramme
    plt.subplot(len(images), 2, 2 * i + 2)
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.title("Histogramme " + titre)


plt.tight_layout()
plt.show()
