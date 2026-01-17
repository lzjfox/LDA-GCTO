# LDA-GCTO: An end-to-end joint optimization model for lncRNA-disease association prediction based on multi-layer graph convolution with skip connections
Deciphering lncRNA-disease associations (LDAs) is crucial for understanding disease mechanisms and advancing therapeutic strategies. However, existing computational methods often struggle with high-dimensional feature embedding and class imbalance in biological datasets. To mitigate these issues, we propose an end-to-end joint optimization model named LDA-GCTO, to robustly infer new LDAs. The LDA-GCTO framework begins by constructing similarity matrices for lncRNAs and diseases using their biological features and association network. A multi-layer graph convolutional network enhanced with skip connections is then employed to capture both lncRNA/disease local and global features from their local neighborhood patterns and global structural characteristics. Subsequently, an end-to-end joint optimization strategy is designed to select the most informative lncRNA-disease (L-D) features. Finally, a tree ensemble model is utilized to reconstruct the potential association matrix. 
# 1. Flowchart
![Figure 1:The flowchart of LDA-GCTO](Fig.png)

# 2. Running environment
```bash
python version 3.9.18 
numpy==1.23.2
pandas==2.1.4
torch==1.6.1+pt20cpu
torch-geometric==2.6.1            
torch-scatter==2.1.1+pt20cpu   
torch-sparse==0.6.17+pt20cpu   
 
## 🚀 快速开始

### 环境依赖
```bash
# 建议 Python >= 3.8
pip install torch pandas numpy scikit-learn matplotlib
