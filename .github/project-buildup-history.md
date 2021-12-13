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
## 2021-09-27 - Day 4: Reconstruction and evaluation

- Task summary: Quick fix before closing: the color legend on the scatter plot was overlapping with the axis labels on smaller screen sizes. Adjusted the legend position and added a tight layout call.
- Deliverable: Plot layout fixed. Looks clean at standard notebook width now.
## 2021-11-08 - Day 5: Comparison with LDA

- Task summary: Added a Linear Discriminant Analysis comparison to the iris notebook to contrast with PCA. Since LDA uses the class labels it naturally gives a better separation in 2D — that is expected and worth explaining. The goal was to write a clear side-by-side that shows when you would prefer each. Wrote a brief explanation of supervised vs unsupervised dimensionality reduction in the markdown and added a 2x2 subplot layout for the comparison.
- Deliverable: PCA vs LDA comparison added. Clear explanation of tradeoff written up.
## 2021-11-08 - Day 5: Comparison with LDA

- Task summary: Minor fix: the LDA implementation was using all three components but the 2D plot was only using the first two. Made it explicit and added a note explaining why three components exist for a three-class problem.
- Deliverable: LDA component count documented. Plot is intentionally 2D.
## 2021-12-13 - Day 6: Write-up finalization

- Task summary: Finalized the iris PCA analysis write-up today. Added a conclusion section summarizing the key findings — how much variance is captured, which features load on which components, and how the projection relates to the species taxonomy. Also added a short section at the end on what next steps would look like if this were a real production task (dimensionality reduction before a downstream classifier, monitoring component drift). Cleaned up all the inline comments and made the notebook presentable.
- Deliverable: Write-up complete. Notebook presentable and tells a coherent story.
