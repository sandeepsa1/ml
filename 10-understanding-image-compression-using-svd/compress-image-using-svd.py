import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = "car.jpg"
image = Image.open(image_path)
image_gray = image.convert('L')
image_array = np.array(image_gray)

plt.figure(figsize=(8, 6))

plt.subplot(2, 2, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

U, s, Vt = np.linalg.svd(image_array)

pos = 2
for r in (5, 10, 100):  # Number of singular values to keep
    compressed_image = np.dot(U[:, :r], np.dot(np.diag(s[:r]), Vt[:r, :]))
    plt.subplot(2, 2, pos)
    plt.imshow(compressed_image, cmap='gray')
    plt.title('Compressed Image (k = {})'.format(r))
    plt.axis('off')
    pos += 1

plt.tight_layout()
plt.show()

#Analyze performance with various amount of singular values
plt.figure(1)
plt.semilogy(np.diag(s))
plt.title('Singular Values')
plt.show()

plt.figure(2)
plt.semilogy(np.cumsum(np.diag(s)) / np.sum(np.diag(s)))
plt.title('Singular Values')
plt.show()