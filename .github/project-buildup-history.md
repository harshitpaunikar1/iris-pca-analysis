# Project Buildup History: Iris PCA Analysis

- Repository: `iris-pca-analysis`
- Category: `data_science`
- Subtype: `pca`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2021-09-20 - Day 3: PCA implementation

- Task summary: Worked through the PCA implementation for Iris today. Ran the decomposition on the standardized features and plotted the explained variance ratio — the first two components captured over 97 percent of the variance which is reassuringly clean for this dataset. The 2D projection separated the three species visually quite well with only the versicolor/virginica boundary being a bit blurry. Added a loadings plot to show how each original feature contributed to the principal components and wrote up a short interpretation of what each axis seemed to represent physically.
- Deliverable: PCA complete. PC1 mostly reflects overall size, PC2 petal shape ratio.
## 2021-09-27 - Day 4: Reconstruction and evaluation

- Task summary: Extended the iris PCA work to include reconstruction error analysis. Reconstructed the original features from 1, 2, and 3 components and measured the MSE at each level. Also added a biplot combining scores and loadings in one view — had to adjust the scaling factor a few times before it looked readable. The notebook was getting a bit long so spent the last hour reorganizing it into logical sections with clear markdown headers.
- Deliverable: Biplot added, reconstruction analysis done, notebook reorganized.
