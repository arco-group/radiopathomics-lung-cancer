# Data Layout

The analysis scripts expect the following input layout:

```text
data/
  pathomics.csv
  radiomics.csv
  radiomics_handcrafted.csv
  semantic-features.csv
  deep_features/
    pathomics-resnet18.csv
    radiomics-googlenet.csv
  pathomics/
    dataset_sliding.csv
    crop_label.csv
    sliding_window/
```

Expected tabular convention:

- The first column contains the sample or subject identifier used for grouping.
- Feature columns are numeric.
- The last column contains the binary adaptive-radiotherapy label.
