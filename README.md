# LDA-GCTO: An end-to-end joint optimization model for lncRNA-disease association prediction based on multi-layer graph convolution with skip connections
Deciphering lncRNA-disease associations (LDAs) is crucial for understanding disease mechanisms and advancing therapeutic strategies. However, existing computational methods often struggle with high-dimensional feature embedding and class imbalance in biological datasets. To mitigate these issues, we propose an end-to-end joint optimization model named LDA-GCTO, to robustly infer new LDAs. The LDA-GCTO framework begins by constructing similarity matrices for lncRNAs and diseases using their biological features and association network. A multi-layer graph convolutional network enhanced with skip connections is then employed to capture both lncRNA/disease local and global features from their local neighborhood patterns and global structural characteristics. Subsequently, an end-to-end joint optimization strategy is designed to select the most informative lncRNA-disease (L-D) features. Finally, a tree ensemble model is utilized to reconstruct the potential association matrix. To systematically evaluate the performance of LDA-GCTO, we conducted comprehensive experiments under multiple settings, including 5-fold cross-validations on lncRNA and disease "cold-start" scenarios, L-D pairs, and independent validation. Across three independent datasets (MNDR v2.0, lncRNADisease v3.0, and Lnc2Cancer v3.0), LDA-GCTO consistently outperformed four state-of-the-art benchmark methods, demonstrating superior robustness and generalization capability. Further comparative evaluation against five leading imbalanced data processing models confirmed LDA-GCTO's effectiveness in handling data imbalance. Visualization of the L-D feature distribution, coupled with a comparative analysis against three established boosting models, further verified the critical contributions of the graph convolution architecture with skip connections and the end-to-end joint optimization strategy. Case studies on colorectal neoplasms validated the reproducibility of LDA-GCTO. Additionally, we curated a new LDA dataset to supplement existing resources. LDA-GCTO is freely available at GitHub https://github.com/lzjfox/LDA-GCTO.

#1. Flowchart

![整体流程](Fig.png)

流程说明：
1. **输入**：疾病语义相似性矩阵、lncRNA功能相似性矩阵、已知关联标签。
2. **特征提取**：通过 `Gcn_skip.py` 构建异构图并生成节点嵌入。
3. **特征选择**：对高维嵌入进行筛选或降维。
4. **分类训练**：使用 `main.py` 调用分类器（如 RF/XGBoost）进行训练。
5. **交叉验证**：通过 `CV.py` 进行 k 折交叉验证，输出 AUC/AUPR 等指标。

---

## 🚀 快速开始

### 环境依赖
```bash
# 建议 Python >= 3.8
pip install torch pandas numpy scikit-learn matplotlib
