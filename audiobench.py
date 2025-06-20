import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score
from sklearn.calibration import calibration_curve
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

# === Load the dataset ===
df = pd.read_csv("data/dfaudiofeatures.csv")

# === Rename emotion 'neutral' to 'unknown' ===
df["true_emotion"] = df["true_emotion"].replace("neutral", "unknown")

# Drop ID column if present
if 'id' in df.columns:
    df = df.drop(columns=["id"])

# === Encode emotion labels ===
label_encoder = LabelEncoder()
df["emotion_encoded"] = label_encoder.fit_transform(df["true_emotion"])

# === Split features and labels ===
X = df.drop(columns=["true_emotion", "emotion_encoded"])
y = df["emotion_encoded"]

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === Train Random Forest Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Predict and evaluate ===
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# === Classification Report ===
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("=== Classification Report ===\n")
print(report)

# === Macro F1 Score ===
macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"Macro F1 Score: {macro_f1:.4f}")

# === Cohen's Kappa ===
kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")

# === Chi-squared Test ===
contingency = pd.crosstab(y_test, y_pred)
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"Chi-squared Test: chi2={chi2:.2f}, p={p:.4f}, dof={dof}")

# === Expected Calibration Error (ECE) ===
def compute_ece(y_true, y_proba, n_bins=10):
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true)
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(mask):
            bin_accuracy = np.mean(accuracies[mask])
            bin_confidence = np.mean(confidences[mask])
            ece += np.abs(bin_confidence - bin_accuracy) * np.sum(mask) / len(y_true)
    return ece

ece = compute_ece(y_test.to_numpy(), y_proba)
print(f"Expected Calibration Error (ECE): {ece:.4f}")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("output/audio_confusion_matrix.png")
plt.show()

# === Save model, encoder, and scaler ===
joblib.dump(clf, "output/audio_rf_model.pkl")
joblib.dump(label_encoder, "output/audio_label_encoder.pkl")
joblib.dump(scaler, "output/audio_scaler.pkl")

# === Calibration Curves ===
plt.figure(figsize=(6, 6))
for i in range(len(label_encoder.classes_)):
    prob_true, prob_pred = calibration_curve((y_test == i).astype(int), y_proba[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, label=f"{label_encoder.classes_[i]}")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Predicted confidence")
plt.ylabel("True probability")
plt.title("Calibration Curves per Class")
plt.legend()
plt.grid(True)
plt.show()

# === Compare Audio Model vs LLM Predictions ===
print("\n=== Comparing Audio Model vs LLM (Lyrics) Predictions ===")

# === Add audio predictions only to test set ===
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test["lyrics"] = df.loc[y_test.index, "lyrics"].values  # recover lyrics using index
df_test["true_emotion"] = df.loc[y_test.index, "true_emotion"].values
df_test["audio_emotion"] = label_encoder.inverse_transform(y_pred)

# === Load LLM output (e.g., gpt_emotion) ===
llm_df = pd.read_csv("data/mergedresult3.csv")
llm_df["gpt_emotion"] = llm_df["gpt_emotion"].str.lower().replace("neutral", "unknown")
llm_df["true_emotion"] = llm_df["true_emotion"].str.lower().replace("neutral", "unknown")
df_test["true_emotion"] = df_test["true_emotion"].str.lower().replace("neutral", "unknown")

# === Merge on id ===
merged = pd.merge(llm_df, df_test, on="id", how="inner")

# === Encode emotions consistently
le = LabelEncoder()
merged["true_encoded"] = le.fit_transform(merged["true_emotion"])
merged["gpt_encoded"] = le.transform(merged["gpt_emotion"])
merged["audio_encoded"] = le.transform(merged["audio_emotion"])

# === Evaluation: LLM vs Ground Truth
f1_llm = f1_score(merged["true_encoded"], merged["gpt_encoded"], average="macro")
kappa_llm = cohen_kappa_score(merged["true_encoded"], merged["gpt_encoded"])

f1_audio = f1_score(merged["true_encoded"], merged["audio_encoded"], average="macro")
kappa_audio = cohen_kappa_score(merged["true_encoded"], merged["audio_encoded"])

print("\n=== Comparing Audio Model vs LLM (Lyrics) Predictions ===")
print(f"LLM (Lyrics) — Macro F1: {f1_llm:.4f}, Cohen’s Kappa: {kappa_llm:.4f}")
print(f"Audio Model — Macro F1: {f1_audio:.4f}, Cohen’s Kappa: {kappa_audio:.4f}")

# === Modality Disagreement Matrix
agreement_matrix = confusion_matrix(merged["gpt_encoded"], merged["audio_encoded"])

plt.figure(figsize=(10, 8))
sns.heatmap(
    agreement_matrix,
    annot=True,
    fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap="YlGnBu"
)
plt.xlabel("Audio Prediction")
plt.ylabel("LLM Prediction")
plt.title("Modality Disagreement Matrix: LLM vs Audio")
plt.tight_layout()
plt.savefig("output/modality_disagreement_matrix.png")
plt.show()
