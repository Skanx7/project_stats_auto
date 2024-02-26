# Import necessary libraries
from pycaret.classification import *

# Load your datasets
import pandas as pd
train = pd.read_csv('data/preprocessed_train.csv')
test = pd.read_csv('data/preprocessed_test.csv')

# Setup PyCaret - updated for the latest version
clf1 = setup(data=train, target='Transported', session_id=123, use_gpu=True)

# Compare models and sort by AUC
best_model = compare_models(sort='Accuracy', n_select=1)

# Tune the best model (optional)
tuned_model = tune_model(best_model, optimize='Accuracy', n_iter=100)

# Finalize the model for predictions
final_model = finalize_model(tuned_model)

# Predict on the test set
predictions = predict_model(final_model, data=test)

# Save predictions to CSV
predictions = predictions[['PassengerId', 'prediction_label']]
predictions = predictions.rename(columns={'prediction_label': 'Transported'})
predictions['Transported'] = predictions['Transported'].astype(bool)
predictions.to_csv('data/submission.csv', index=False)

print("Predictions saved successfully.")
