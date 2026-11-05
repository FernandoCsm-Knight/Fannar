# Readings

| reading | what it measures |
|---|---|
| `knn_prediction` | leave-one-out k-NN accuracy against the model's predictions |
| `correctness_auc` | AUC of the geometric margin as a predictor of the model being right |
| `confusion_agreement` | Spearman between class proximity and the model's (soft) confusion |
| `containment` | class directions inside the retained subspace (r = K − 1) |
| `e_ratio_error_auc` | AUC of the orthogonal-energy ratio as a predictor of an error |

::: fannar.readings
    options:
      show_root_heading: false
      show_docstring_description: false
      members_order: source
