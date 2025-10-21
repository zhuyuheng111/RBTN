# 🌾 Rice Blast Temporal Network (RBTN)
### BIBM 2025 Accepted Short Paper — Prototype for Spatio-Temporal Transcriptomic Integration  
**Author:** [Yuheng Zhu]  
**Affiliation:** Jilin University  
**Version:** RBTN
**Date:** 2025  

---

## 🧠 English Summary

**Rice Blast Temporal Network (RBTN)** is a prototype model developed to explore the feasibility of integrating temporal information into spatial transcriptomics during *Magnaporthe oryzae* infection in rice.  
It independently trains mapping matrices for each infection stage (0h, 12h, 24h) and fuses their embeddings through **post-hoc temporal interpolation**, without joint optimization across time points.  

RBTN serves as the **conceptual foundation of the Plant Spatio-Temporal Integration Network (PSTN)**, which generalizes this idea into a unified, jointly optimized spatio-temporal framework.

---

### 🚀 Key Features
- 🧬 **Stage-wise Independent Mapping:** Each infection stage is modeled separately.  
- ⏱️ **Post-hoc Temporal Fusion:** Temporal embeddings are linearly interpolated to generate intermediate-stage representations (e.g., 6h, 18h).  
- 💡 **Lightweight Prototype:** Fast training, low computational cost, and easy extension to other time-resolved datasets.  
- 🌿 **Biological Context:** Designed for rice–blast fungus interaction analysis using scRNA-seq and spatial transcriptomics data.

---
