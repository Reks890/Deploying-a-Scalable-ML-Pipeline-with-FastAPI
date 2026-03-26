# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random Forest Classifier using default scikit-learn version 1.5.1, with default hyperparameters.

## Intended Use

This model is used to classify population segments based on salary. The target variable salary is predicted using either <=50K or >50K. Intended use case for this model is purely educational exploration.

## Training Data

This data was obtained from the UC Irvine Machine Learning Repository 
https://archive.ics.uci.edu/dataset/20/census+income. For training a one hot encoder was applied to the categorical features, and labels binarized using a label binarizer.

## Evaluation Data

The evaluation set uses a 20% holdout from the census data. This 80/20 split utilized stratification on the salary
variable to ensure equal distribution. Evaluation data has been properly seperated from training data to preserve model integrity.

## Metrics

This model uses the metrics precision, recall, and f1 score to evaluate performance. A random state of 42 is used
to ensure reproducibility and the results are as follows... Precision: 0.7353 | Recall: 0.6378 | F1: 0.6831.
Per data slice calculations are offered by the slice_output.txt file created upon running train_model.py.


## Ethical Considerations

The dataset includes some sensitive demographics like race, sex, or education, but does well to exclude
PII at a granular level for any unique individual. Model results should be used exclusively for educational
purposes.

## Caveats and Recommendations

While the overall model performance appears reasonable, f1 score variation across different categories like education, native country, and marital-status may indicate introduced bias. Education in particular appears to have a high variability in results, with individuals with little to no college education being strongly underrepresented. Recommendations for further study would be to gather a more varied sample and improve representation.