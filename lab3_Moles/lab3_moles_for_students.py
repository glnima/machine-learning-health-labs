# -*- coding: utf-8 -*-
"""Laboratory 3 - mole segmentation with K-Means, DBSCAN, smoothing, and Sobel."""
from pathlib import Path
from zipfile import ZipFile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import DBSCAN, KMeans


# Change PROCESS_ALL_IMAGES to False and set IMAGE_NAME for an oral-exam run.
PROCESS_ALL_IMAGES = True
IMAGE_NAME = "low_risk_4.jpg"
N_COLORS = 3
FALLBACK_COLORS = 6  # used only if three colours merge the mole with a large shadow
DBSCAN_EPS = 1.5       # joins horizontally, vertically, and diagonally adjacent dark pixels
DBSCAN_MIN_SAMPLES = 5
MIN_MOLE_PIXELS = 1_000
MEDIAN_DELTA = 2       # a 5 x 5 median filter window

LAB_DIR = Path(__file__).parent
IMAGE_DIR = LAB_DIR / "moles"
ZIP_FILE = LAB_DIR / "moles.zip"
OUTPUT_DIR = LAB_DIR / "lab3_results"


def ensure_images_available():
    """Extract the supplied archive on first run, if needed."""
    if not IMAGE_DIR.exists():
        with ZipFile(ZIP_FILE) as archive:
            archive.extractall(LAB_DIR)


def get_image_files():
    files = sorted(IMAGE_DIR.glob("*.jpg"))
    # The archive also contains hue/saturation helper files, not mole photographs.
    return [file for file in files if not file.stem.endswith(("_h", "_s"))]


def choose_mole_cluster(pixel_positions, labels, image_shape):
    """Choose a large compact DBSCAN component, preferring a central component on ties."""
    image_center = np.asarray(image_shape) / 2
    image_diagonal = np.linalg.norm(image_center)
    candidates = []
    for label in np.unique(labels):
        if label == -1:
            continue  # DBSCAN noise
        points = pixel_positions[labels == label]
        count = len(points)
        if count < MIN_MOLE_PIXELS:
            continue
        center = points.mean(axis=0)
        inertia = np.mean(np.sum((points - center) ** 2, axis=1))
        central_distance = np.linalg.norm(center - image_center) / image_diagonal
        # Large, compact components have high density; centrality resolves shadow ties.
        score = (count / inertia) / (1 + central_distance)
        candidates.append((score, label, count, inertia))
    if not candidates:
        raise RuntimeError("No DBSCAN component has at least 1000 pixels.")
    _, label, count, inertia = max(candidates, key=lambda item: item[0])
    return label, count, inertia


def median_smooth(mask):
    """Apply the requested median low-pass filter to a binary mole mask."""
    padded = np.pad(mask.astype(np.uint8), MEDIAN_DELTA, constant_values=0)
    windows = sliding_window_view(padded, (2 * MEDIAN_DELTA + 1,) * 2)
    return np.median(windows, axis=(-2, -1)) >= 0.5


def sobel_border(mask):
    """Apply both Sobel filters manually and combine their magnitudes into a border."""
    image = (mask.astype(np.int16) * 255)
    padded = np.pad(image, 1, constant_values=0)
    windows = sliding_window_view(padded, (3, 3))
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = kernel_x.T
    filtered_x = np.sum(windows * kernel_x, axis=(-2, -1))
    filtered_y = np.sum(windows * kernel_y, axis=(-2, -1))
    magnitude = np.hypot(filtered_x, filtered_y)
    return magnitude > 0


def segment_mole(file_path):
    """Segment one mole and return diagnostics needed for plots and the summary table."""
    original = mpimg.imread(file_path)
    grayscale = np.mean(original, axis=2).astype(np.uint8)
    n_rows, n_cols = grayscale.shape
    def cluster_darkest_pixels(n_colors):
        kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=10).fit(
            grayscale.reshape(-1, 1)
        )
        labels_image = kmeans.labels_.reshape(n_rows, n_cols)
        darkest_label = int(np.argmin(kmeans.cluster_centers_.ravel()))
        dark_positions = np.argwhere(labels_image == darkest_label)
        dbscan_labels = DBSCAN(
            eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean"
        ).fit_predict(dark_positions)
        mole_label, mole_pixels, inertia = choose_mole_cluster(
            dark_positions, dbscan_labels, grayscale.shape
        )
        return kmeans, labels_image, dark_positions, dbscan_labels, mole_label, mole_pixels, inertia

    kmeans, labels_image, dark_positions, dbscan_labels, mole_label, mole_pixels, inertia = (
        cluster_darkest_pixels(N_COLORS)
    )
    # In melanoma_27-like low-contrast images, three colour levels can merge the
    # mole with background shadow. A finer fallback restores a compact component.
    if mole_pixels > 75_000:
        kmeans, labels_image, dark_positions, dbscan_labels, mole_label, mole_pixels, inertia = (
            cluster_darkest_pixels(FALLBACK_COLORS)
        )
    positions = dark_positions[dbscan_labels == mole_label]
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[positions[:, 0], positions[:, 1]] = True

    margin = 5
    min_row = max(0, positions[:, 0].min() - margin)
    max_row = min(n_rows, positions[:, 0].max() + margin + 1)
    min_col = max(0, positions[:, 1].min() - margin)
    max_col = min(n_cols, positions[:, 1].max() + margin + 1)
    cropped_color = original[min_row:max_row, min_col:max_col]
    cropped_mask = mask[min_row:max_row, min_col:max_col]
    smoothed_mask = median_smooth(cropped_mask)
    border = sobel_border(smoothed_mask)
    return {
        "original": original,
        "grayscale": grayscale,
        "quantized": kmeans.cluster_centers_[labels_image].squeeze(),
        "cropped_color": cropped_color,
        "cropped_mask": cropped_mask,
        "smoothed_mask": smoothed_mask,
        "border": border,
        "mole_pixels": mole_pixels,
        "inertia": inertia,
        "dbscan_clusters": len(set(dbscan_labels)) - int(-1 in dbscan_labels),
    }


def save_example_plot(name, result):
    """Save all intermediate stages and the final colour-plus-border overlay."""
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(result["original"])
    axes[0, 0].set_title("Original image")
    axes[0, 1].imshow(result["grayscale"], cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Grayscale image")
    axes[0, 2].imshow(result["quantized"], cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Three K-Means colours")
    axes[1, 0].imshow(result["cropped_mask"], cmap="gray")
    axes[1, 0].set_title("DBSCAN mole mask")
    axes[1, 1].imshow(result["smoothed_mask"], cmap="gray")
    axes[1, 1].set_title("Median-smoothed mask")
    axes[1, 2].imshow(result["cropped_color"])
    axes[1, 2].contour(result["border"], levels=[0.5], colors="red", linewidths=1)
    axes[1, 2].set_title("Final Sobel border")
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(name)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{Path(name).stem}_pipeline.png", dpi=150)
    plt.close(figure)


def save_overview(results):
    """Create a compact visual check that the same settings work on all 54 images."""
    figure, axes = plt.subplots(6, 9, figsize=(15, 10))
    for axis, (name, result) in zip(axes.flat, results.items()):
        axis.imshow(result["cropped_color"])
        axis.contour(result["border"], levels=[0.5], colors="red", linewidths=0.7)
        axis.set_title(name, fontsize=7)
        axis.axis("off")
    figure.suptitle("Lab 3: Sobel borders for all 54 mole images")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "all_moles_border_overview.png", dpi=180)
    plt.close(figure)


def main():
    ensure_images_available()
    OUTPUT_DIR.mkdir(exist_ok=True)
    image_files = get_image_files()
    if len(image_files) != 54:
        raise RuntimeError(f"Expected 54 mole photographs, found {len(image_files)}.")
    selected = image_files if PROCESS_ALL_IMAGES else [IMAGE_DIR / IMAGE_NAME]
    results = {}
    summary = []
    for file_path in selected:
        result = segment_mole(file_path)
        results[file_path.name] = result
        summary.append({
            "image": file_path.name,
            "mole_pixels": result["mole_pixels"],
            "cluster_inertia": result["inertia"],
            "dbscan_clusters": result["dbscan_clusters"],
        })
        save_example_plot(file_path.name, result)
        print(f"{file_path.name}: {result['mole_pixels']} mole pixels")
    if PROCESS_ALL_IMAGES:
        save_overview(results)
    import pandas as pd
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "segmentation_summary.csv", index=False)
    print(f"Processed {len(results)} image(s); outputs saved in {OUTPUT_DIR}")
    if not PROCESS_ALL_IMAGES:
        plt.show()


if __name__ == "__main__":
    main()
