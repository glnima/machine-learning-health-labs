"""Lab 6: Gaussianity experiments and artificial ICA/PCA comparison."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA, PCA


OUTPUT = Path(__file__).with_name("lab6_results")
MU, SIGMA, ALPHA, NEXP = 5.0, 2.0, 0.05, 2_000


def box_muller(rng, size):
    u = rng.random((size, 2))
    return np.sqrt(-2 * SIGMA**2 * np.log1p(-u[:, 0])) * np.cos(2 * np.pi * u[:, 1]) + MU


def central_limit(rng, size, n):
    k = np.sqrt(3) * SIGMA
    return rng.uniform(-k, k, size=(size, n)).sum(axis=1) / np.sqrt(n) + MU


def ad_statistic(samples):
    x = np.sort(samples)
    n = len(x)
    f = stats.norm.cdf(x, MU, SIGMA)
    f = np.clip(f, 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return -n - np.sum((2 * i - 1) / n * (np.log(f) + np.log(1 - f[::-1])))


def tests(samples, null_kurtosis, null_ad):
    t_stat = stats.ttest_1samp(samples, MU).statistic
    excess_kurtosis = stats.kurtosis(samples, fisher=True, bias=False)
    ad = ad_statistic(samples)
    p_t = 2 * stats.t.sf(abs(t_stat), len(samples) - 1)
    p_kurt = np.mean(np.abs(null_kurtosis) >= abs(excess_kurtosis))
    p_ad = np.mean(null_ad >= ad)
    return p_t, p_kurt, p_ad


def run_gaussianity():
    records = []
    # Null distributions are generated once per sample size for empirical kurtosis/A-D p-values.
    null_cache = {}
    for ns in (100, 500, 1000):
        rng = np.random.default_rng(9000 + ns)
        null = rng.normal(MU, SIGMA, size=(NEXP, ns))
        null_cache[ns] = (stats.kurtosis(null, axis=1, fisher=True, bias=False),
                          np.array([ad_statistic(row) for row in null]))
        p = tests(box_muller(np.random.default_rng(30), ns), *null_cache[ns])
        records.append({"method": "Box-Muller", "seed": 30, "Ns": ns, "N": np.nan,
                        "p_t": p[0], "p_kurtosis": p[1], "p_AD": p[2],
                        "do_not_reject": all(value >= ALPHA for value in p)})
    # The requested CLT sweep and a second seed demonstrate the experiment's randomness.
    for seed in (30, 71):
        for n in range(1, 21):
            p = tests(central_limit(np.random.default_rng(seed), 1000, n), *null_cache[1000])
            records.append({"method": "Central limit", "seed": seed, "Ns": 1000, "N": n,
                            "p_t": p[0], "p_kurtosis": p[1], "p_AD": p[2],
                            "do_not_reject": all(value >= ALPHA for value in p)})
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT / "gaussianity_results.csv", index=False)
    plt.figure(figsize=(8, 4))
    for seed, group in table[table.method.eq("Central limit")].groupby("seed"):
        plt.semilogy(group.N, group[["p_t", "p_kurtosis", "p_AD"]].min(axis=1), "-o", label=f"seed {seed}")
    plt.axhline(ALPHA, color="red", linestyle="--", label="alpha = 0.05")
    plt.xlabel("N uniforms summed")
    plt.ylabel("minimum p-value across tests")
    plt.title("Central-limit Gaussianity experiment")
    plt.grid(); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT / "central_limit_pvalues.png", dpi=150); plt.close()
    return table


def run_ica():
    rng = np.random.default_rng(50)
    n = 10_000; t = np.linspace(0, 10, n)
    x = np.vstack((np.sin(2*np.pi*.5*t-np.pi/4), np.sign(np.sin(2*np.pi*.5*np.sqrt(2)*t-np.pi/5)),
                   signal.sawtooth(2*np.pi*.5*np.sqrt(5)*t), np.cumsum(np.sign(np.sin(2*np.pi*.5*np.sqrt(2)*t-np.pi/5)))))
    x[3] = (x[3] - x[3].mean()) / x[3].std()
    observed = (rng.normal(size=(4, 4)) @ x).T
    ica = FastICA(n_components=4, algorithm="deflation", whiten="unit-variance", random_state=50)
    ica_components = ica.fit_transform(observed).T
    pca_components = PCA(n_components=4).fit_transform(observed).T
    def matched_correlation(components):
        corr = np.abs(np.corrcoef(x, components)[:4, 4:])
        rows, cols = linear_sum_assignment(-corr)
        return corr[rows, cols]
    summary = pd.DataFrame({"source": np.arange(1, 5), "FastICA_abs_correlation": matched_correlation(ica_components),
                            "PCA_abs_correlation": matched_correlation(pca_components)})
    summary.to_csv(OUTPUT / "ica_pca_recovery.csv", index=False)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for k, axis in enumerate(axes):
        axis.plot(t, x[k] / np.max(np.abs(x[k])), "--", label=f"source {k+1}")
        axis.plot(t, ica_components[k] / np.max(np.abs(ica_components[k])), label=f"ICA component {k+1}")
        axis.grid(); axis.legend(loc="upper right")
    axes[-1].set_xlabel("time (s)"); fig.suptitle("FastICA recovered components")
    fig.tight_layout(); fig.savefig(OUTPUT / "fastica_recovery.png", dpi=150); plt.close(fig)
    return summary


def main():
    OUTPUT.mkdir(exist_ok=True)
    gaussianity = run_gaussianity()
    recovery = run_ica()
    print("Box-Muller results:\n", gaussianity[gaussianity.method.eq("Box-Muller")].to_string(index=False))
    print("\nFastICA/PCA recovery:\n", recovery.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
