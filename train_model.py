import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# TODO: load the cencus.csv data
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")

data = pd.read_csv(data_path)

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
LABEL = "salary"  # put this near the top (once), or reuse if you already have it

train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data[LABEL],  # keeps salary class balance
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",       
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",       
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model.pkl")
encoder_path = os.path.join(model_dir, "encoder.pkl")
lb_path = os.path.join(model_dir, "lb.pkl")

save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

# load the model
model = load_model(model_path)
encoder = load_model(encoder_path)
lb = load_model(lb_path)

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Write slice metrics to slice_output.txt (overwrite each run)

slice_out_path = os.path.join(project_path, "slice_output.txt")

with open(slice_out_path, "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = (test[col] == slicevalue).sum()

            p_s, r_s, fb_s = performance_on_categorical_slice(
                test,
                col,
                slicevalue,
                cat_features,
                LABEL,
                encoder,
                lb,
                model,
            )

            f.write(f"{col}: {slicevalue}, Count: {count}\n")
            f.write(f"Precision: {p_s:.4f} | Recall: {r_s:.4f} | F1: {fb_s:.4f}\n\n")