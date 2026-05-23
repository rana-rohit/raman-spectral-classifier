import os
import matplotlib.pyplot as plt

def plot_saliency(signal, saliency, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(signal, label="Signal")
    plt.plot(saliency, label="Saliency", alpha=0.7)

    plt.xlabel("Wavelength Index")
    plt.ylabel("Intensity / Importance")
    plt.title("Saliency Map")

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()