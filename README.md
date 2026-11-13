# D-Fire: Real-Time Smoke & Fire Detection with RF-DETR + DoRA

Smoke and fire detection pipeline on the D-Fire dataset ([gaia-solutions-on-demand/DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset)) using **RF-DETR Nano** and **DoRA (Weight-Decomposed Low-Rank Adaptation)**.

---

## Highlights

- **Architecture:** RF-DETR Nano ($384 \times 384$ input resolution).
- **PEFT Adaptation:** DoRA adapters on ViT self-attention layers ($r=16, \alpha=32$), training 24.35% of model parameters.
- **Loss Formulation:** Categorical Focal Loss + Complete IoU (CIoU) + L1 box loss.
- **Dataset Stratification:** 80/20 stratified split preserving all 9,838 negative background images to reduce false alarms.
- **Training:** Trained on NVIDIA Tesla T4 GPU (Google Colab).

---

## Results

| Model | Resolution | Adaptation | Epochs / Setup | Val Loss | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **RF-DETR Nano** | $384 \times 384$ | **DoRA ($r=16$)** | **5 (Progressive Curriculum)** | **0.7697** | 2 ep gray, 2 ep mixed, 1 ep color. |
| **RF-DETR Nano** | $384 \times 384$ | DoRA ($r=16$) | 3 (Colab T4, Gray) | 1.6978 | Grayscale baseline. |

---

## Method Details

### 1. Model Adaptation (DoRA)
DoRA adapters ($r=16, \alpha=32$) are injected into the multi-head self-attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) of the RF-DETR backbone. This updates 24.35% of the total model parameters while keeping base weights frozen.

### 2. Progressive Color Curriculum
To prevent the model from relying solely on color shortcuts (like mistaking red objects for fire or clouds for smoke), training follows a 3-stage curriculum:

- **Stage 1 (Epochs 1-2):** 100% grayscale (`R = G = B`) forces the model to focus on plume shapes, edges, and textures rather than color.
- **Stage 2 (Epochs 3-4):** 50% random chance of grayscale or color per sample, transitioning weights toward color data.
- **Stage 3 (Epoch 5):** 100% color training to align with natural camera feeds.
- **Result:** Validation loss dropped from 1.6978 to 0.7697.

### 3. Background Negative Sampling
The D-Fire dataset contains 9,838 images with no smoke or fire. Rather than removing them, the dataset is split 80/20 across four disjoint categories (`smoke only`, `fire only`, `both`, `none`). Preserving all background samples teaches the model to predict background ($\varnothing$) on clear scenes, reducing false alarms.

### 4. Loss Function
The model trains on a weighted combination of:
- **Categorical Focal Loss:** Mitigates class imbalance across foreground objects and background queries.
- **L1 Loss:** Direct bounding box coordinate regression.
- **Complete IoU (CIoU) Loss:** Accounts for overlap area, center distance, and aspect ratio consistency to better fit expanding, irregular smoke plumes.

---

## Acknowledgements

- Dataset sourced from [gaia-solutions-on-demand/DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
