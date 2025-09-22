# processing_fixed.py
import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

# Extra libs
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from textblob import TextBlob

# SHAP
import shap
# silence warnings for cleaner output
warnings.filterwarnings("ignore")

# -----------------------------
# Step 0: Load Data
# -----------------------------
df = pd.read_csv("data.csv")   # ensure your csv has columns used below
df['Juror_ID'] = range(1, len(df) + 1)
df_original = df.copy()        # human-readable copy

# -----------------------------
# Step 1: Encode Categorical Variables (keep original df_original)
# -----------------------------
categorical_cols = ['Gender', 'Region', 'Education', 'Profession']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# -----------------------------
# Step 2: Ensure numeric bias columns exist and clip
# -----------------------------
epsilon = 1e-6
for col in ['Civil_Bias', 'Criminal_Bias', 'Fairness_Score']:
    if col not in df_encoded.columns:
        df_encoded[col] = 0.5  # neutral default
    df_encoded[col] = df_encoded[col].astype(float).clip(epsilon, 1 - epsilon)

# -----------------------------
# Step 3: Preprocess Text Data (Questionnaire) & TF-IDF
# -----------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r'[^a-zA-Z\s]', ' ', s)
    tokens = [w for w in s.split() if w not in stop_words]
    return " ".join(tokens)

df_encoded['Questionnaire_clean'] = df_encoded.get('Questionnaire', pd.Series([""]*len(df_encoded))).apply(preprocess_text)

# Save/load TF-IDF vectorizer for consistent dims across runs
vectorizer_path = "vectorizer.pkl"
if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
    text_features = vectorizer.transform(df_encoded['Questionnaire_clean'])
else:
    vectorizer = TfidfVectorizer(max_features=500)
    text_features = vectorizer.fit_transform(df_encoded['Questionnaire_clean'])
    joblib.dump(vectorizer, vectorizer_path)

# -----------------------------
# Step 4: Combine Features & make dense for scalers/linear models
# -----------------------------
numerical_cols = [c for c in ['Age','Political_Leaning','Civil_Bias','Criminal_Bias','Fairness_Score'] if c in df_encoded.columns]
X_numerical = df_encoded[numerical_cols].astype(float).values  # (n_samples, n_num)
# text_features is sparse; convert to dense
X_text = text_features.toarray()  # (n_samples, n_text_feats)

# Full design matrix
X = np.hstack([X_numerical, X_text])  # dense numpy array
feature_names = numerical_cols + list(vectorizer.get_feature_names_out())

# -----------------------------
# Step 5: Targets (scale them to [0,1])
# -----------------------------
# Ensure targets exist; if not, create neutral defaults
y_pro = df_encoded.get('Prosecution_Bias', pd.Series(0.5, index=df_encoded.index)).astype(float).values.reshape(-1,1)
y_def = df_encoded.get('Defense_Bias', pd.Series(0.5, index=df_encoded.index)).astype(float).values.reshape(-1,1)

# Scale inputs and targets
scaler_X_path = "scaler_X.pkl"
if os.path.exists(scaler_X_path):
    scaler_X = joblib.load(scaler_X_path)
    X_scaled = scaler_X.transform(X)
else:
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    joblib.dump(scaler_X, scaler_X_path)

# targets -> MinMax in [0,1]
target_scaler_pro_path = "target_scaler_pro.pkl"
target_scaler_def_path = "target_scaler_def.pkl"
if os.path.exists(target_scaler_pro_path) and os.path.exists(target_scaler_def_path):
    target_scaler_pro = joblib.load(target_scaler_pro_path)
    target_scaler_def = joblib.load(target_scaler_def_path)
    y_pro_scaled = target_scaler_pro.transform(y_pro)
    y_def_scaled = target_scaler_def.transform(y_def)
else:
    target_scaler_pro = MinMaxScaler(feature_range=(0,1))
    target_scaler_def = MinMaxScaler(feature_range=(0,1))
    y_pro_scaled = target_scaler_pro.fit_transform(y_pro)
    y_def_scaled = target_scaler_def.fit_transform(y_def)
    joblib.dump(target_scaler_pro, target_scaler_pro_path)
    joblib.dump(target_scaler_def, target_scaler_def_path)

# -----------------------------
# Step 6: Train models (SGDRegressor) on scaled data
# -----------------------------
pro_model_path = "pro_model.pkl"
def_model_path = "def_model.pkl"

pro_model = SGDRegressor(max_iter=2000, tol=1e-4, random_state=42)
def_model = SGDRegressor(max_iter=2000, tol=1e-4, random_state=42)

# Fit (fresh training)
pro_model.fit(X_scaled, y_pro_scaled.ravel())
def_model.fit(X_scaled, y_def_scaled.ravel())

# Save models
joblib.dump(pro_model, pro_model_path)
joblib.dump(def_model, def_model_path)

# Quick train eval (MSE on scaled targets)
print("Prosecution MSE (scaled target):", mean_squared_error(y_pro_scaled, pro_model.predict(X_scaled)))
print("Defense MSE (scaled target):", mean_squared_error(y_def_scaled, def_model.predict(X_scaled)))

# -----------------------------
# Step 7: Predict (and inverse-transform back to [0,1])
# -----------------------------
pro_pred_scaled = pro_model.predict(X_scaled).reshape(-1,1)
def_pred_scaled = def_model.predict(X_scaled).reshape(-1,1)

# Inverse transform to original [0,1] range (target scaler was fitted to original y ranges)
pro_scores = np.clip(target_scaler_pro.inverse_transform(pro_pred_scaled).ravel(), 0.0, 1.0)
def_scores = np.clip(target_scaler_def.inverse_transform(def_pred_scaled).ravel(), 0.0, 1.0)

# Safe neutral: ensure within [0,1]
neutral_scores = np.clip(1 - (pro_scores + def_scores) / 2.0, 0.0, 1.0)

# Attach to df_encoded & df_original (human-friendly)
df_encoded['Predicted_Prosecution_Bias'] = pro_scores
df_encoded['Predicted_Defense_Bias'] = def_scores
df_encoded['Predicted_Neutral_Bias'] = neutral_scores
df_encoded['Overall_Fairness'] = (pro_scores + def_scores + neutral_scores)/3

df_original = df_original.copy()
df_original['Predicted_Prosecution_Bias'] = df_encoded['Predicted_Prosecution_Bias'].values
df_original['Predicted_Defense_Bias'] = df_encoded['Predicted_Defense_Bias'].values
df_original['Predicted_Neutral_Bias'] = df_encoded['Predicted_Neutral_Bias'].values
df_original['Overall_Fairness'] = df_encoded['Overall_Fairness'].values

# Print small sanity sample
print("\nSample predicted biases (first 8 jurors):")
print(df_original[['Juror_ID','Name','Predicted_Prosecution_Bias','Predicted_Defense_Bias','Predicted_Neutral_Bias']].head(8))

# -----------------------------
# Step 8: Balanced Jury Selection
# -----------------------------
df_encoded['Fairness_Distance'] = abs(df_encoded['Overall_Fairness'] - 0.5)
balanced_jury = df_encoded.nsmallest(20, 'Fairness_Distance').copy()
print("\nSelected Balanced Jury (top 20):")
print(balanced_jury[['Juror_ID','Name','Predicted_Prosecution_Bias','Predicted_Defense_Bias','Overall_Fairness']])

# -----------------------------
# Step 9: Clustering for Diversity (use scaled X)
# -----------------------------
num_clusters = 3
kmeans_full = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters_full = kmeans_full.fit_predict(X_scaled)
df_original['Cluster'] = clusters_full
print("\nCluster sizes (full dataset):")
print(df_original['Cluster'].value_counts())

# -----------------------------
# Step 10: Final Jury (simulate elimination)
# -----------------------------
# Step 10: Final Jury (simulate elimination)
# Step 10a: Assign clusters to balanced_jury before elimination
X_balanced = balanced_jury[['Predicted_Prosecution_Bias','Predicted_Defense_Bias','Predicted_Neutral_Bias']].values
num_clusters = min(3, len(balanced_jury))
kmeans_balanced = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(X_balanced)
balanced_jury['Cluster'] = kmeans_balanced.labels_
# Step 10b: Simulate elimination (fixed IDs for consistency)
# In practice, these would be input by the user
advocate_eliminated_ids = [2,5,7,10,12,15,17,19]
eliminated_df = balanced_jury[balanced_jury['Juror_ID'].isin(advocate_eliminated_ids)].copy()
remaining_df = balanced_jury[~balanced_jury['Juror_ID'].isin(advocate_eliminated_ids)].copy()

# Select top 12 by fairness distance if more than 12 remain
if len(remaining_df) > 12:
    remaining_df = remaining_df.copy()
    remaining_df['Fairness_Distance'] = abs(remaining_df['Overall_Fairness'] - 0.5)
    final_jury = remaining_df.nsmallest(12, 'Fairness_Distance').copy()
else:
    final_jury = remaining_df.copy()

# Merge Cluster info from balanced_jury
final_jury = remaining_df.nsmallest(12, 'Fairness_Distance').copy()
final_jury = final_jury.merge(
    balanced_jury[['Juror_ID', 'Cluster']], 
    on='Juror_ID', 
    how='left'
)

# Plot boxplots
plt.figure(figsize=(8,4))
sns.boxplot(data=final_jury, x='Cluster', y='Predicted_Prosecution_Bias', palette='pastel')
plt.title("Prosecution Bias by Cluster (Final Jury)"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(data=final_jury, x='Cluster', y='Predicted_Defense_Bias', palette='pastel')
plt.title("Defense Bias by Cluster (Final Jury)"); plt.tight_layout(); plt.show()

print("\n✅ Final Jury (12 Members):")
print(final_jury[['Juror_ID','Name','Predicted_Prosecution_Bias','Predicted_Defense_Bias','Overall_Fairness','Cluster']])

# -----------------------------
# Step 11: Clustering the Final Jury (PCA for plotting)
# -----------------------------
# Step 11 (Layman-friendly PCA plot for 20 balanced jurors)
X_balanced = balanced_jury[['Predicted_Prosecution_Bias','Predicted_Defense_Bias','Predicted_Neutral_Bias']].values

# Optional: cluster for colors
num_clusters = min(3, len(balanced_jury))
kmeans_balanced = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(X_balanced)
balanced_jury['Cluster'] = kmeans_balanced.labels_

# PCA 2D reduction
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_balanced)
balanced_jury['PCA1'] = coords[:,0]
balanced_jury['PCA2'] = coords[:,1]

# Layman-friendly scatter plot
plt.figure(figsize=(10,7))
palette = sns.color_palette('Set1', n_colors=num_clusters)

for i, cluster in enumerate(range(num_clusters)):
    cluster_data = balanced_jury[balanced_jury['Cluster']==cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
                color=palette[i], s=150, label=f'Cluster {cluster}', edgecolor='k')

# Add juror names as labels
for _, row in balanced_jury.iterrows():
    plt.text(row['PCA1']+0.002, row['PCA2']+0.002, row['Name'], fontsize=9)

plt.title("Balanced Jury Clusters (20 Jurors) - PCA reduced", fontsize=14)
plt.xlabel("PCA 1", fontsize=12)
plt.ylabel("PCA 2", fontsize=12)
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplots for clarity
plt.figure(figsize=(8,4))
sns.boxplot(data=final_jury, x='Cluster', y='Predicted_Prosecution_Bias', palette='pastel')
plt.title("Prosecution Bias by Cluster (Final Jury)"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(data=final_jury, x='Cluster', y='Predicted_Defense_Bias', palette='pastel')
plt.title("Defense Bias by Cluster (Final Jury)"); plt.tight_layout(); plt.show()

# -----------------------------
# Step 12: Simulate Hearings (small drift) - aggregated median plot
# -----------------------------
try:
    n_hearings = int(input("\nEnter number of hearings in this trial (e.g. 3): "))
except Exception:
    n_hearings = 1

hearing_results = []
for h in range(1, max(1,n_hearings)+1):
    temp = final_jury.copy()
    temp['Predicted_Prosecution_Bias'] = np.clip(temp['Predicted_Prosecution_Bias'] + np.random.normal(0, 0.02, len(temp)), 0, 1)
    temp['Predicted_Defense_Bias'] = np.clip(temp['Predicted_Defense_Bias'] + np.random.normal(0, 0.02, len(temp)), 0, 1)
    temp['Predicted_Neutral_Bias'] = np.clip(1 - (temp['Predicted_Prosecution_Bias'] + temp['Predicted_Defense_Bias'])/2, 0, 1)
    temp['Hearing'] = h
    hearing_results.append(temp)

hearing_df = pd.concat(hearing_results)
agg = hearing_df.groupby('Hearing')[['Predicted_Prosecution_Bias','Predicted_Defense_Bias','Predicted_Neutral_Bias']].median().reset_index()
plt.figure(figsize=(7,4))
plt.plot(agg['Hearing'], agg['Predicted_Prosecution_Bias'], marker='o', label='Prosecution (median)')
plt.plot(agg['Hearing'], agg['Predicted_Defense_Bias'], marker='o', label='Defense (median)')
plt.plot(agg['Hearing'], agg['Predicted_Neutral_Bias'], marker='o', label='Neutral (median)')
plt.xlabel("Hearing"); plt.ylabel("Median Bias"); plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------
# Step 13: Case-topic adjustment (trial only)
# -----------------------------
case_stage = input("\nEnter case stage (preliminary/trial): ").strip().lower()
case_topic = input("Enter case topic (Fraud, Assault, Civil Rights, etc.): ").strip().lower()

adjusted_jury = final_jury.copy()
if case_stage == "trial":
    if "fraud" in case_topic:
        adjusted_jury['Predicted_Prosecution_Bias'] = np.clip(adjusted_jury['Predicted_Prosecution_Bias'] * 1.05, 0, 1)
    elif "assault" in case_topic or "criminal" in case_topic:
        adjusted_jury['Predicted_Prosecution_Bias'] = np.clip(adjusted_jury['Predicted_Prosecution_Bias'] * 1.08, 0, 1)
    elif "civil" in case_topic:
        adjusted_jury['Predicted_Defense_Bias'] = np.clip(adjusted_jury['Predicted_Defense_Bias'] * 1.05, 0, 1)
    adjusted_jury['Predicted_Neutral_Bias'] = np.clip(1 - (adjusted_jury['Predicted_Prosecution_Bias'] + adjusted_jury['Predicted_Defense_Bias'])/2, 0, 1)
    print("\nAdjusted jury biases (trial):")
    print(adjusted_jury[['Juror_ID','Name','Predicted_Prosecution_Bias','Predicted_Defense_Bias','Predicted_Neutral_Bias']])
else:
    print("\nNo case-topic adjustment applied (preliminary).")

# -----------------------------
# STEP X: Sentiment Analysis (TextBlob)
# -----------------------------
df_original['Sentiment'] = df_original['Questionnaire'].fillna("").apply(lambda t: TextBlob(str(t)).sentiment.polarity)
print("\nSentiment sample (first 6):")
print(df_original[['Juror_ID','Name','Sentiment']].head(6))

# -----------------------------
# STEP Y: Explainable AI (SHAP)
# -----------------------------
# SHAP LinearExplainer works well with linear models (SGDRegressor)
# Use a small subset for plotting/shap computations to keep it fast
background = X_scaled[np.random.choice(X_scaled.shape[0], min(100, X_scaled.shape[0]), replace=False)]
explainer = shap.LinearExplainer(pro_model, background, feature_names=feature_names)
# explain first N jurors to avoid heavy computation
N_explain = min(30, X_scaled.shape[0])
shap_values = explainer.shap_values(X_scaled[:N_explain])

# Summary plot (top features) - will display in plotting window
shap.summary_plot(shap_values, features=X_scaled[:N_explain], feature_names=feature_names, show=True)

# Example per-juror explanation (first juror)
print("\nExplaining first juror prediction (SHAP):", df_original.loc[0,'Name'])
shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_scaled[0], feature_names=feature_names))

# -----------------------------
# Persist final adjusted results
# -----------------------------
out_file = "final_jury_results_fixed.csv"
adjusted_jury.to_csv(out_file, index=False)
print(f"\nSaved adjusted final jury results to: {out_file}")

# -----------------------------
# Notes:
# - Models, vectorizer, scalers are saved to disk; future runs will use the same transforms.
# - To improve performance: collect true outcome labels and periodically re-fit models (or use partial_fit).
# - If you run SHAP in a non-interactive console it may error — run in Jupyter / IDE for plots.
# -----------------------------
