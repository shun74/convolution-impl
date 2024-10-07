import numpy as np
import matplotlib.pyplot as plt

def compare_images(img1 : np.ndarray, img2 : np.ndarray, title1 : str = "img1", title2 : str = "img2", cmap : str = "gray"):
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap=cmap)
    plt.title(title1)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap=cmap)
    plt.title(title2)
    plt.axis("off")
    plt.show()
