import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, TFSMLayer
from keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "multitask_loan_model_tf"
X_TEST_PATH = "X_test.csv"
Y_TEST_PATH = "y_test.csv"
PD_THRESHOLD = 0.5

# -----------------------------
# LOAD MODEL (SAME METHOD)
# -----------------------------
print("Loading model...")

tfsm_layer = TFSMLayer(
    MODEL_PATH,
    call_endpoint="serving_default"
)

inputs = Input(shape=(None,), dtype=tf.float32)
outputs = tfsm_layer(inputs)
model = Model(inputs=inputs, outputs=outputs)

print("Model loaded successfully.")
print("TensorFlow version:", tf.__version__)

# -----------------------------
# LOAD DATA
# -----------------------------
print("\nLoading test data...")

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH)

y_true = y_test["default"].values.astype(int)

print(f"Test samples: {len(X_test)}")

# -----------------------------
# RUN PREDICTIONS
# -----------------------------
print("\nRunning predictions...")

X_tensor = tf.convert_to_tensor(X_test.values.astype(np.float32))
outputs = model(X_tensor)

# PD is output_0
pd_probs = outputs["output_0"].numpy().flatten()
pd_preds = (pd_probs >= PD_THRESHOLD).astype(int)

# -----------------------------
# METRICS
# -----------------------------
tn, fp, fn, tp = confusion_matrix(y_true, pd_preds).ravel()

accuracy  = accuracy_score(y_true, pd_preds)
precision = precision_score(y_true, pd_preds, zero_division=0)
recall    = recall_score(y_true, pd_preds)
auc       = roc_auc_score(y_true, pd_probs)

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n================ PD EVALUATION RESULTS ================")
print(f"Threshold: {PD_THRESHOLD}")
print(f"AUC:        {auc:.4f}")
print(f"Accuracy:   {accuracy*100:.2f}%")
print(f"Precision:  {precision*100:.2f}%")
print(f"Recall:     {recall*100:.2f}%")

print("\nConfusion Matrix:")
print(f"  True Positives  (Correct Defaults):     {tp}")
print(f"  False Positives (False Alarms):          {fp}")
print(f"  False Negatives (Missed Defaults):       {fn}")
print(f"  True Negatives (Correct Non-Defaults):  {tn}")

print("\nCounts:")
print(f"  Actual Defaults:    {y_true.sum()}")
print(f"  Predicted Defaults:{pd_preds.sum()}")

# -----------------------------
# TRUE POSITIVE ROW INDICES
# -----------------------------
true_positive_indices = np.where((y_true == 1) & (pd_preds == 1))[0]

print("\nRow numbers of correctly flagged defaults (True Positives):")
print(true_positive_indices)

print("\nTotal True Positives found:", len(true_positive_indices))
print("=======================================================")
