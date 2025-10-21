# ==============================================
# Rice Blast Temporal Network (RBTN-v2)
# Version: BIBM 2025 Accepted Short Paper (Enhanced Distinction)
# Author: [Your Name]
# ----------------------------------------------
# Description:
#   RBTN-v2 models each infection stage independently and integrates
#   their spatial mappings via a post-hoc temporal embedding fusion.
#   It explores the feasibility of temporal signal integration
#   without joint optimization across time points.
# ==============================================

import scanpy as sc
import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# 1. Data loading
# ------------------------------
sc_times = {t: sc.read_h5ad(f"E:/code/deeptalk/selected_HVG4000/sc_{t}h_HVG4000.h5ad") for t in [0, 12, 24]}
st_times = {t: sc.read_h5ad(f"E:/code/deeptalk/selected_HVG4000/st_{t}h_HVG4000.h5ad") for t in [0, 12, 24]}

for ad in list(sc_times.values()) + list(st_times.values()):
    ad.raw = ad.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.pca(ad, n_comps=50)

# ------------------------------
# 2. Model setup
# ------------------------------
A, B, M = {}, {}, {}
for t in [0, 12, 24]:
    sc_adata, st_adata = sc_times[t], st_times[t]
    raw_sc = sc_adata.X.toarray() if sp.issparse(sc_adata.X) else sc_adata.X
    raw_st = st_adata.X.toarray() if sp.issparse(st_adata.X) else st_adata.X

    sums = raw_st.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    st_norm = raw_st / sums * 1e4
    X_st = np.log1p(st_norm)

    A[t] = torch.tensor(raw_sc, dtype=torch.float32, device=device)
    B[t] = torch.tensor(X_st, dtype=torch.float32, device=device)

# Static cosine reconstruction loss
def static_loss(pred_expr, true_expr):
    P_row = F.normalize(pred_expr, dim=1)
    B_row = F.normalize(true_expr, dim=1)
    return - (P_row * B_row).sum(dim=1).mean()

# ------------------------------
# 3. Stage-wise independent training
# ------------------------------
epochs = 800
lr = 1e-3
loss_history = {}

for t in [0, 12, 24]:
    n_cells, n_genes = A[t].shape
    n_spots, _ = B[t].shape
    M[t] = nn.Parameter(torch.randn(n_cells, n_spots, device=device))
    optimizer = optim.Adam([M[t]], lr=lr)

    loss_history[t] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = M[t].T @ A[t]  # [n_spots, n_genes]
        loss = static_loss(pred, B[t])
        loss.backward()
        optimizer.step()
        loss_history[t].append(loss.item())
        if epoch % 100 == 0:
            print(f"[Stage {t}h] Epoch {epoch}/{epochs} — loss_static: {loss.item():.4f}")

    torch.save(M[t].detach().cpu(), f"E:/code/deeptalk/RBTNv2_mapping_matrix_{t}h.pt")

# ------------------------------
# 4. Post-hoc temporal embedding fusion
# ------------------------------
# Instead of temporal consistency loss, fuse embeddings linearly
# Example: combine 0h & 12h → intermediate 6h representation
with torch.no_grad():
    E0 = F.normalize((M[0].T @ A[0]), dim=1)
    E12 = F.normalize((M[12].T @ A[12]), dim=1)
    E24 = F.normalize((M[24].T @ A[24]), dim=1)

    # Temporal interpolation (soft fusion)
    E_6h = 0.4 * E0 + 0.6 * E12
    E_18h = 0.5 * E12 + 0.5 * E24

# Save fused embeddings for later visualization
np.save("E:/code/deeptalk/RBTNv2_embedding_6h.npy", E_6h.cpu().numpy())
np.save("E:/code/deeptalk/RBTNv2_embedding_18h.npy", E_18h.cpu().numpy())

# ------------------------------
# 5. Visualization
# ------------------------------
plt.figure(figsize=(7, 5))
for t, losses in loss_history.items():
    plt.plot(losses, label=f'{t}h static loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Stage-wise Static Loss (RBTN-v2)")
plt.legend()
plt.tight_layout()
plt.show()

print("RBTN-v2 training completed. Stage-wise mapping + temporal fusion embeddings saved.")
