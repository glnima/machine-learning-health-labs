# Lab 3 roadmap: mole segmentation

## What the professor asks for

1. Segment the mole in each of the 54 photographs; do not use the snake/active-contour algorithm.
2. Quantize the grayscale image into three colours using K-Means.
3. Select the darkest colour, then use DBSCAN on its pixel coordinates to separate the mole from shadows and isolated pixels.
4. Keep a sufficiently large (at least 1,000 pixels), compact DBSCAN component as the mole.
5. Crop around the mole, smooth its binary mask with a median low-pass filter, apply both Sobel filters, and combine them to form the border.
6. Confirm that the same hyperparameters work for every image, because the professor may choose one at random during the exam.

## Run it

```zsh
source .venv/bin/activate
python lab3_Moles/lab3_moles_for_students.py
```

The first run extracts `moles.zip` automatically if the `moles/` folder is absent. The default processes all 54 photos and writes the results to `lab3_Moles/lab3_results/`.

## How to test one image at the exam

At the top of the script, set:

```python
PROCESS_ALL_IMAGES = False
IMAGE_NAME = "melanoma_12.jpg"  # replace with the randomly selected filename
```

Run the script and inspect the six-stage pipeline figure. The final panel shows the original cropped colour image with the red Sobel border overlaid.

## What to inspect

- `all_moles_border_overview.png`: visual confirmation for all 54 images.
- `segmentation_summary.csv`: selected-component sizes and DBSCAN diagnostics.
- `<image>_pipeline.png`: K-Means, DBSCAN mask, smoothing, and final contour for an individual image.
