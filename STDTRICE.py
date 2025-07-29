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
# 1. 数据加载与预处理
# ------------------------------
# 你需要根据实际路径修改下面的文件名
sc_times = {
    t: sc.read_h5ad(f"E:/code/deeptalk/selected_HVG4000/sc_{t}h_HVG4000.h5ad")
    for t in [0,12,24]
}
st_times = {
    t: sc.read_h5ad(f"E:/code/deeptalk/selected_HVG4000/st_{t}h_HVG4000.h5ad")
    for t in [0,12,24]
}

for ad in list(sc_times.values()) + list(st_times.values()):
    # 1) 把原始 counts 存到 .raw
    ad.raw = ad.copy()

    # 2) 在 .X 上做 normalize + log1p
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    # 3) （可选）下面这些操作只为了可视化，不影响训练
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.pca(ad, n_comps=50)
    sc.pp.neighbors(ad, n_pcs=50)
    sc.tl.umap(ad, min_dist=0.5)

# ------------------------------
# 2. 构建表达矩阵与映射矩阵参数
# ------------------------------
A = {}  # 单细胞表达：tensor [n_cells, n_genes]
B = {}  # 空间表达：tensor [n_spots, n_genes]
M = {}  # 存放每个时点的映射矩阵参数

for t in [0, 12, 24]:
    # 1) 读取原始 counts
    sc_adata = sc_times[t]   # 假设 sc_times[t].X 还是原始 counts
    st_adata = st_times[t]   # 同理，st_times[t].X 保留了原始 counts

    raw_sc = sc_adata.X
    raw_st = st_adata.X

    # 如果是 sparse，就转成 dense
    if sp.issparse(raw_sc):
        raw_sc = raw_sc.toarray()
    if sp.issparse(raw_st):
        raw_st = raw_st.toarray()

    # 2) SC 端我们直接用原始 counts
    X_sc = raw_sc

    # 3) ST 端要做 normalize_total + log1p
    #    这里直接用 NumPy 重写 normalize+log1p
    sums = raw_st.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    st_norm = raw_st / sums * 1e4
    X_st = np.log1p(st_norm)

    # 4) 转成 torch Tensor
    A[t] = torch.tensor(X_sc, dtype=torch.float32, device=device)  # [n_cells, n_genes]
    B[t] = torch.tensor(X_st, dtype=torch.float32, device=device)  # [n_spots, n_genes]

    # 5) 构建映射矩阵参数
    n_cells, _ = A[t].shape
    n_spots, _ = B[t].shape
    M[t] = nn.Parameter(torch.randn(n_cells, n_spots, device=device))

# ------------------------------
# 3. 损失函数与优化器
# ------------------------------
loss_history = []
lambda_temp = 0.1  # 时序正则系数，可调
optimizer = optim.Adam(list(M.values()), lr=1e-3)
epochs = 2000


# 余弦重构损失：对基因和对 spot 两个方向都要计算
def static_loss(pred_expr, true_expr):
    # pred_expr, true_expr: [n_spots, n_genes]
    # 列方向（基因）余弦：先转置
    P_col = F.normalize(pred_expr.T, dim=1)
    B_col = F.normalize(true_expr.T, dim=1)
    loss_col = - (P_col * B_col).sum(dim=1).mean()
    # 行方向（spots）余弦
    P_row = F.normalize(pred_expr, dim=1)
    B_row = F.normalize(true_expr, dim=1)
    loss_row = - (P_row * B_row).sum(dim=1).mean()
    # print("pred_expr    shape:", pred_expr.shape)
    # print("pred_expr.T  shape:", pred_expr.T.shape)
    # print("P_col        shape:", P_col.shape)
    # print("B_col        shape:", B_col.shape)
    # print("P_row        shape:", P_row.shape)
    # print("B_row        shape:", B_row.shape)
    return loss_col + loss_row


# ------------------------------
# 4. 训练循环
# ------------------------------
loss_static_history = []
loss_temp_history = []
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    # 4.1 静态映射损失
    loss_static = torch.tensor(0.0, device=device)
    pred = {}
    for t in [0, 12, 24]:
        # 预测表达: (M^t)^T @ A^t  -> [n_spots, n_genes]
        pred[t] = M[t].T @ A[t]
        loss_static += static_loss(pred[t], B[t])

    # 4.2 时序一致性损失（以 spot embedding 为例）
    loss_temp = torch.tensor(0.0, device=device)

    # 0h <-> 12h
    E0 = F.normalize(pred[0], dim=1)  # [n0, D]
    E12 = F.normalize(pred[12], dim=1)  # [n12, D]
    # 相似度矩阵：S[i,j] = cos(E0[i], E12[j])
    S0_12 = torch.mm(E0, E12.T)  # [n0, n12]

    # 对于每个 0h spot，找在 12h 上最相似的那个
    max0_12, _ = S0_12.max(dim=1)  # [n0]
    # 对于每个 12h spot，找在 0h 上最相似的那个
    max12_0, _ = S0_12.max(dim=0)  # [n12]

    # 双向平均
    loss_temp += - (max0_12.mean() + max12_0.mean()) * 0.5

    # 12h <-> 24h，同理：
    E24 = F.normalize(pred[24], dim=1)  # [n24, D]
    S12_24 = torch.mm(E12, E24.T)  # [n12, n24]
    max12_24, _ = S12_24.max(dim=1)  # [n12]
    max24_12, _ = S12_24.max(dim=0)  # [n24]
    loss_temp += - (max12_24.mean() + max24_12.mean()) * 0.5

    # 4.3 总损失
    loss = loss_static + lambda_temp * loss_temp
    loss.backward()
    optimizer.step()
    loss_static_history.append(loss_static.item())
    loss_temp_history.append(loss_temp.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} — "
              f"loss_static: {loss_static.item():.4f}, "
              f"loss_temp: {loss_temp.item():.4f}, "
              f"total_loss: {loss.item():.4f}")
# ------------------------------
# 5. 结果输出
# ------------------------------
# 训练结束后，M[0], M[12], M[24] 即为映射矩阵，可用于后续分析或可视化。
for t in [0, 12, 24]:
    # 1) 先把 A[t], B[t] 都转到 CPU 上，再变成 NumPy
    if torch.is_tensor(A[t]):
        A_np = A[t].detach().cpu().numpy()
    else:
        A_np = A[t]

    if torch.is_tensor(B[t]):
        B_np = B[t].detach().cpu().numpy()
    else:
        B_np = B[t]

    # 2) 用 A_np, B_np 保存
    np.save(f"E:/code/deeptalk/A_{t}h.npy", A_np)
    np.save(f"E:/code/deeptalk/B_{t}h.npy", B_np)
    torch.save(M[t].detach().cpu(), f"mapping_matrix_{t}h.pt")
    A_np_raw = A[t].detach().cpu().numpy() if torch.is_tensor(A[t]) else A[t]
    np.save(f"E:/code/deeptalk/A_raw_{t}h.npy", A_np_raw)

    # —— 新增：保存 normalize+log1p 后的 ST 表达 —— #
    B_np_norm = B[t].detach().cpu().numpy() if torch.is_tensor(B[t]) else B[t]
    np.save(f"E:/code/deeptalk/B_norm_{t}h.npy", B_np_norm)
print("Static losses:", loss_static_history)
print("Temporal losses:", loss_temp_history)
print("训练完成，映射矩阵已保存。")
x = range(1, epochs + 1)
plt.figure()
plt.plot(x, loss_static_history, label='loss_static')
plt.plot(x, loss_temp_history, label='loss_temp')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Static vs. Temporal Loss")
plt.legend()
plt.tight_layout()
plt.show()
