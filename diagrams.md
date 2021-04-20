# PCA on Iris Dataset Diagrams

Generated on 2026-04-26T04:17:39Z from repository evidence.

## Architecture Overview

```mermaid
flowchart LR
    A[Repository Inputs] --> B[Preparation and Validation]
    B --> C[ML Case Study Core Logic]
    C --> D[Output Surface]
    D --> E[Insights or Actions]
```

## Workflow Sequence

```mermaid
flowchart TD
    S1["PCA IRIS dataset"]
    S2["Scaling the data"]
    S1 --> S2
    S3["Applying PCA on the data"]
    S2 --> S3
    S4["Making the scree plot"]
    S3 --> S4
    S5["PCA with 2 components"]
    S4 --> S5
```
