import os
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
# — user settings ——
DATA_DIR = "E:/code/deeptalk"
TIMEPOINTS = [0, 12, 24]
TARGET_SUM = 1e4  # same as normalize_total(target_sum=1e4)

def normalize_log1p(mat, target_sum=1e4):
    """per-spot normalize_total + log1p"""
    sums = mat.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    normed = mat / sums * target_sum
    return np.log1p(normed)


def load_data(t):
    # Load raw counts, true normalized-log data, and mapping matrix
    A_raw = np.load(os.path.join(DATA_DIR, f"A_raw_{t}h.npy"))
    B_log = np.load(os.path.join(DATA_DIR, f"B_{t}h.npy"))
    M = torch.load(os.path.join(DATA_DIR, f"mapping_matrix_{t}h.pt"), map_location="cpu")
    return A_raw, B_log, M

def evaluate_timepoint(t):
    A_raw, B_log, M = load_data(t)

    # 1) Linear reconstruction to raw count space
    A_t = torch.from_numpy(A_raw.astype(np.float32))
    pred_raw = (M.T @ A_t).detach().numpy()

    # Clip negatives to zero
    pred_raw = np.clip(pred_raw, a_min=0, a_max=None)

    # 2) Normalize + log1p
    pred = normalize_log1p(pred_raw, TARGET_SUM)
    true = B_log  # Already normalized + log1p

    print(f"\n=== Time {t}h ===")
    print(" pred shape:", pred.shape, " true shape:", true.shape)

    # Global MSE & R²
    mse_global = mean_squared_error(true.flatten(), pred.flatten())
    r2_global = r2_score(true.flatten(), pred.flatten())
    print(f"\n  Global MSE = {mse_global:.4f}")
    print(f"  Global R²  = {r2_global:.4f}")

    # Gene-wise Pearson R
    gene_rs = []
    for g in range(true.shape[1]):
        col_t, col_p = true[:, g], pred[:, g]
        if np.nanstd(col_t) > 0:
            r, _ = pearsonr(col_t, col_p)
            gene_rs.append(r)
    print(f"\n  Mean gene-wise Pearson R = {np.nanmean(gene_rs):.4f}")

    # Spot-wise Pearson R
    spot_rs = []
    for s in range(true.shape[0]):
        row_t, row_p = true[s, :], pred[s, :]
        if np.nanstd(row_t) > 0:
            r, _ = pearsonr(row_t, row_p)
            spot_rs.append(r)
    print(f"  Mean spot-wise Pearson R = {np.nanmean(spot_rs):.4f}")

    # Gene-wise R²
    gene_r2s = []
    for g in range(true.shape[1]):
        col_t, col_p = true[:, g], pred[:, g]
        if np.nanstd(col_t) > 0:
            ss_res = np.sum((col_t - col_p) ** 2)
            ss_tot = np.sum((col_t - col_t.mean()) ** 2)
            gene_r2s.append(1 - ss_res / ss_tot)
    print(f"\n  Mean gene-wise R² = {np.nanmean(gene_r2s):.4f}")
    P_row = F.normalize(torch.from_numpy(pred), dim=1).numpy()  # [n_spots, G]
    B_row = F.normalize(torch.from_numpy(true), dim=1).numpy()
    P_col = F.normalize(torch.from_numpy(pred).T, dim=1).numpy()  # [G, n_spots]
    B_col = F.normalize(torch.from_numpy(true).T, dim=1).numpy()

    # 计算余弦相似度
    row_sims = (P_row * B_row).sum(axis=1)  # 每个 spot 的相似度
    col_sims = (P_col * B_col).sum(axis=1)  # 每个 gene 的相似度

    # 画箱型图
    plt.figure(figsize=(5, 4))
    plt.boxplot([row_sims, col_sims], labels=['spot→cell', 'cell→spot'])
    plt.ylabel('Cosine similarity')
    plt.title(f'Static similarity distribution at {t}h')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Original evaluation prints
    for t in TIMEPOINTS:
        evaluate_timepoint(t)

    # Collect metrics without printing
    metrics_list = []
    for t in TIMEPOINTS:
        A_raw, B_log, M = load_data(t)
        A_t = torch.from_numpy(A_raw.astype(np.float32))
        pred_raw = (M.T @ A_t).detach().numpy()
        pred_raw = np.clip(pred_raw, a_min=0, a_max=None)
        pred = normalize_log1p(pred_raw, TARGET_SUM)
        true = B_log
        mse_global = mean_squared_error(true.flatten(), pred.flatten())
        r2_global  = r2_score(true.flatten(), pred.flatten())

        gene_rs = [
            pearsonr(true[:, g], pred[:, g])[0]
            for g in range(true.shape[1]) if np.nanstd(true[:, g]) > 0
        ]
        spot_rs = [
            pearsonr(true[s, :], pred[s, :])[0]
            for s in range(true.shape[0]) if np.nanstd(true[s, :]) > 0
        ]

        metrics_list.append({
            'time': t,
            'mse': mse_global,
            'r2':  r2_global,
            'gene_rs': gene_rs,
            'spot_rs': spot_rs
        })

    # Plot Pred vs True scatter & residuals
    for m in metrics_list:
        t = m['time']
        A_raw, B_log, M = load_data(t)
        A_t = torch.from_numpy(A_raw.astype(np.float32))
        pred_raw = (M.T @ A_t).detach().numpy()
        pred_raw = np.clip(pred_raw, 0, None)
        pred = normalize_log1p(pred_raw, TARGET_SUM)
        true = B_log

        # Scatter of predicted vs. true values
        flat_true = true.flatten()
        flat_pred = pred.flatten()
        idx = np.random.choice(len(flat_true), size=min(10000, len(flat_true)), replace=False)

        plt.figure(figsize=(6,6))
        plt.scatter(flat_true[idx], flat_pred[idx], s=5, alpha=0.3)
        mx = max(flat_true[idx].max(), flat_pred[idx].max())
        plt.plot([0, mx], [0, mx], 'r--', label='y=x')
        plt.xlabel('True expression (log1p)')
        plt.ylabel('Predicted expression (log1p)')
        plt.title(f'Pred vs True at {t}h')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Residual histogram
        residuals = flat_pred - flat_true
        plt.figure()
        plt.hist(residuals, bins=100)
        plt.xlabel('Predicted - True')
        plt.ylabel('Frequency')
        plt.title(f'Residual Distribution at {t}h')
        plt.tight_layout()
        plt.show()

        # Gene-level mean scatter
        gene_true_mean = true.mean(axis=0)
        gene_pred_mean = pred.mean(axis=0)
        plt.figure(figsize=(6,6))
        plt.scatter(gene_true_mean, gene_pred_mean, s=10, alpha=0.6)
        mn, mx = min(gene_true_mean.min(), gene_pred_mean.min()), max(gene_true_mean.max(), gene_pred_mean.max())
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel('Mean True per Gene')
        plt.ylabel('Mean Pred per Gene')
        plt.title(f'Gene-Level Mean at {t}h')
        plt.tight_layout()
        plt.show()

        # Spot-level mean scatter
        spot_true_mean = true.mean(axis=1)
        spot_pred_mean = pred.mean(axis=1)
        plt.figure(figsize=(6,6))
        plt.scatter(spot_true_mean, spot_pred_mean, s=10, alpha=0.6)
        mn, mx = min(spot_true_mean.min(), spot_pred_mean.min()), max(spot_true_mean.max(), spot_pred_mean.max())
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel('Mean True per Spot')
        plt.ylabel('Mean Pred per Spot')
        plt.title(f'Spot-Level Mean at {t}h')
        plt.tight_layout()
        plt.show()
