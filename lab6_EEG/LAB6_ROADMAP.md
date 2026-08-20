# Lab 6 roadmap: Gaussianity, ICA, and EEG

## Executable requirements completed

1. Run Box-Muller samples for `Ns = 100, 500, 1000` and test whether Gaussianity is rejected at `alpha = 0.05`.
2. Run the central-limit construction with `Ns = 1000`, `N = 1` through `20`, then repeat with a different NumPy seed.
3. Apply the PDF's t-score, excess-kurtosis, and Anderson-Darling Gaussianity checks. Kurtosis and A-D p-values are estimated by simulation, as requested.
4. Run the supplied artificial four-signal ICA example and compare FastICA with PCA using matched source/component correlations.

## Run

```zsh
source .venv/bin/activate
python lab6_EEG/lab6_analysis.py
```

Results are saved in `lab6_EEG/lab6_results/`.

## How to interpret

- In `gaussianity_results.csv`, `do_not_reject=True` means all three p-values are at least 0.05. The minimum central-limit `N` can differ by seed because the experiment is random.
- `ica_pca_recovery.csv` compares absolute correlations after matching components to sources. FastICA should recover the non-Gaussian source signals more faithfully than PCA.

## EEGLAB part (manual)

The PDF also requires MATLAB + Signal Processing Toolbox + EEGLAB and EEGLAB's `eeglab_data_epochs_ica.set` sample dataset. These are not in this repository, so run this part manually: load the sample data, high-pass at 1 Hz, low-pass at 50 Hz (or use a 60 Hz notch), inspect spectra/ERPs/time-frequency plots, then run **Tools > Decompose Data by ICA** and identify artifact components such as eye blinks.
