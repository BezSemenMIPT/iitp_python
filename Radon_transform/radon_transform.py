import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image


# Using linear interpolation method


def discrete_radon(path):
    # Relative Path
    input_img = Image.open(path)
    img = input_img.convert("L")
    # img = img.reduce(4)
    img = img.resize((100, 100))
    image_matrix = np.matrix(img)
    img_upd = np.ones(image_matrix.shape) * 255 - image_matrix
    (M, N) = img_upd.shape
    H, K = 500, 500
    d_x, d_y = 2 / (M - 1), 2 / (N - 1)
    x_min, y_min = -1, -1
    p_min, t_min = np.pi, 10
    p = np.linspace(-p_min, p_min, K)
    t = np.linspace(-t_min, t_min, H)
    g_radon = np.zeros((K, H))

    for k in range(K):
        alpha = p[k] * d_x / d_y
        for h in range(H):
            beta = (p[k] * x_min + t[h] - y_min) / d_y
            if alpha > 0:
                m_min = max(0, math.ceil(-beta / alpha))
                m_max = min(M - 1, math.floor((N - 1 - beta) / alpha))
            if alpha < 0:
                m_min = max(0, math.ceil((N - 1 - beta) / alpha))
                m_max = min(M - 1, math.floor(-beta / alpha))
            sum = 0
            for i in range(m_min, m_max + 1):
                nfloat = alpha * i + beta
                j = math.floor(nfloat)
                w = nfloat - j
                sum += img_upd[i, j] * (1 - w) + img_upd[i, j + 1] * w
            g_radon[k, h] = sum * d_x
    return g_radon


name = "img.png"
sinogram = discrete_radon(name)


fig = plt.figure()
plt.title("Radon transform\n(Sinogram)")
plt.imshow(sinogram)
plt.savefig("transformed picture " + name)
plt.show()

# Detecting lines -> in progress
"""
potential_lines = np.where(sinogram > 0.9 * np.max(sinogram))

# Display the original image and the sinogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_title("Original Image")
input_img = Image.open("img1.png")
image = input_img.convert("L")
img = image.resize((100, 100))
image_matrix = np.matrix(img)
ax1.imshow(image, cmap="gray")

ax2.set_title("Radon Transform (Sinogram)")
ax2.set_xlabel("Projection angle (degrees)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap="gray", extent=(0, 180, 0, sinogram.shape[0]), aspect="auto")

# Plot the detected lines on the original image
for i in range(len(potential_lines[0])):
    angle_index = potential_lines[1][i]
    angle = np.deg2rad(angle_index)
    d = potential_lines[0][i]
    x_vals = np.arange(0, image_matrix.shape[1])
    y_vals = (d - x_vals * np.cos(angle)) / np.sin(angle)
    ax1.plot(x_vals, y_vals, color="red", linewidth=2)

plt.show()
"""
