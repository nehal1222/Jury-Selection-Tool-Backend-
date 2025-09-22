import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data.csv")
df['Juror_ID'] = range(1, len(df) + 1)

# -----------------------------
# Step 1: Encode Categorical Variables
# -----------------------------
# Convert categorical columns into numeric format using one-hot encoding
categorical_cols = ['Gender', 'Region', 'Education', 'Profession']
df_original = df.copy()   # keep human-readable version

# Encode for modeling
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# -----------------------------
# Step 2: Clip Numerical Bias Scores
# -----------------------------
epsilon = 1e-4
for col in ['Civil_Bias', 'Criminal_Bias', 'Fairness_Score']:
    df_encoded[col] = df_encoded[col].clip(epsilon, 1-epsilon)

# -----------------------------
# Step 3: Preprocess Text Data (Questionnaire)
# -----------------------------
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

df_encoded['Questionnaire_clean'] = df_encoded['Questionnaire'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=500)
text_features = vectorizer.fit_transform(df_encoded['Questionnaire_clean'])

# -----------------------------
# Step 4: Combine Features for Modeling
# -----------------------------
from scipy.sparse import hstack
import numpy as np

numerical_cols = ['Age', 'Political_Leaning', 'Civil_Bias', 'Criminal_Bias', 'Fairness_Score']
X_numerical = df_encoded[numerical_cols].values
X = hstack([X_numerical, text_features])   # combine numeric + text

# Targets
y_prosecution = df_encoded['Prosecution_Bias'].values
y_defense = df_encoded['Defense_Bias'].values
y_neutral = df_encoded['Neutral_Bias'].values

# -----------------------------
# Step 5: Train Models
# -----------------------------
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# Paths to save models
pro_model_path = "pro_model.pkl"
def_model_path = "def_model.pkl"
vectorizer_path = "vectorizer.pkl"

# Delete existing models to retrain from scratch
if os.path.exists(pro_model_path):
    os.remove(pro_model_path)
if os.path.exists(def_model_path):
    os.remove(def_model_path)
if os.path.exists(vectorizer_path):
    os.remove(vectorizer_path)

# Create new models
pro_model = SGDRegressor(max_iter=1000, tol=1e-3)
def_model = SGDRegressor(max_iter=1000, tol=1e-3)

# Train models (use fit() for first time training)
pro_model.fit(X, y_prosecution)
def_model.fit(X, y_defense)

# Save models and vectorizer for future runs
joblib.dump(pro_model, pro_model_path)
joblib.dump(def_model, def_model_path)
joblib.dump(vectorizer, vectorizer_path)

# Evaluate
print("Prosecution MSE:", mean_squared_error(y_prosecution, pro_model.predict(X)))
print("Defense MSE:", mean_squared_error(y_defense, def_model.predict(X)))

# -----------------------------
# Step 6: Predict Bias Scores for All Jurors
# -----------------------------
pro_scores = pro_model.predict(X)
def_scores = def_model.predict(X)
neutral_scores = 1 - (pro_scores + def_scores) / 2  # rough neutral approximation

# 🔧 FIX: Convert all bias scores to positive values using MinMaxScaler
scaler_pro = MinMaxScaler(feature_range=(10, 90))  # Scale to 10-90 for better interpretation
scaler_def = MinMaxScaler(feature_range=(10, 90))
scaler_neu = MinMaxScaler(feature_range=(10, 90))

pro_scores_positive = scaler_pro.fit_transform(pro_scores.reshape(-1, 1)).flatten()
def_scores_positive = scaler_def.fit_transform(def_scores.reshape(-1, 1)).flatten()
neutral_scores_positive = scaler_neu.fit_transform(neutral_scores.reshape(-1, 1)).flatten()

df_encoded['Predicted_Prosecution_Bias'] = pro_scores_positive
df_encoded['Predicted_Defense_Bias'] = def_scores_positive
df_encoded['Predicted_Neutral_Bias'] = neutral_scores_positive

# -----------------------------
# Step 6.5: Fairness + Demographic Summaries
# -----------------------------
# 🔧 FIX: Calculate fairness using normalized values
df_encoded['Overall_Fairness'] = 100 - abs(pro_scores_positive - def_scores_positive)  # Higher = more balanced
df_original['Predicted_Prosecution_Bias'] = df_encoded['Predicted_Prosecution_Bias']
df_original['Predicted_Defense_Bias'] = df_encoded['Predicted_Defense_Bias']
df_original['Predicted_Neutral_Bias'] = df_encoded['Predicted_Neutral_Bias']
df_original['Overall_Fairness'] = df_encoded['Overall_Fairness']

# Summaries with positive values
print("\n📊 Average Bias by Gender (Scale 10-90):")
print(df_original.groupby('Gender')[['Predicted_Prosecution_Bias','Predicted_Defense_Bias']].mean())
print("\n📊 Average Bias by Region (Scale 10-90):")
print(df_original.groupby('Region')[['Predicted_Prosecution_Bias','Predicted_Defense_Bias']].mean())
print("\n📊 Average Bias by Profession (Scale 10-90):")
print(df_original.groupby('Profession')[['Predicted_Prosecution_Bias','Predicted_Defense_Bias']].mean())
print("\n📊 Average Bias by Education (Scale 10-90):")
print(df_original.groupby('Education')[['Predicted_Prosecution_Bias','Predicted_Defense_Bias']].mean())

# -----------------------------
# Step 6.6: Balanced Jury Selection
# -----------------------------
# 🔧 FIX: Use fairness score for selection (higher = more balanced)
balanced_jury = df_encoded.nlargest(20, 'Overall_Fairness')  # Select most balanced jurors
print("\n✅ Selected Balanced Jury (Top 20 Most Fair):")
print(balanced_jury[['Name','Predicted_Prosecution_Bias','Predicted_Defense_Bias','Overall_Fairness']])

# -----------------------------
# Step 6.7: Clustering for diversity
# -----------------------------
from sklearn.cluster import KMeans
num_clusters = 3
clusters = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(X)
df_original['Cluster'] = clusters
print("\n📈 Jury Pool Distribution by Cluster:")
print(df_original.groupby('Cluster').size())
for c in range(num_clusters):
    print(f"\nCluster {c} Demographics:")
    print(df_original[df_original['Cluster']==c][['Gender','Profession','Region']].mode())

# -----------------------------
# Step 6.8: Bias-prioritized juries with positive interpretation
# -----------------------------
def select_specialized_jury(df, bias_preference='balanced', size=20):
    temp_df = df.copy()
    
    if bias_preference == 'prosecution':
        # Select jurors with higher prosecution bias
        return temp_df.nlargest(size, 'Predicted_Prosecution_Bias')
    elif bias_preference == 'defense':
        # Select jurors with higher defense bias
        return temp_df.nlargest(size, 'Predicted_Defense_Bias')
    else:  # balanced/neutral
        # Select most balanced jurors
        return temp_df.nlargest(size, 'Overall_Fairness')

jury_pro = select_specialized_jury(df_encoded, 'prosecution')
jury_def = select_specialized_jury(df_encoded, 'defense')
jury_neutral = select_specialized_jury(df_encoded, 'balanced')

print("\n⚖️ Prosecution-Favoring Jury (Highest Prosecution Scores):")
display_jury = jury_pro[['Name','Predicted_Prosecution_Bias','Predicted_Defense_Bias']].copy()
print(display_jury.round(1))

print("\n🛡️ Defense-Favoring Jury (Highest Defense Scores):")
display_jury = jury_def[['Name','Predicted_Prosecution_Bias','Predicted_Defense_Bias']].copy()
print(display_jury.round(1))

print("\n🎯 Most Balanced Jury (Highest Fairness Scores):")
display_jury = jury_neutral[['Name','Predicted_Prosecution_Bias','Predicted_Defense_Bias']].copy()
print(display_jury.round(1))

# -----------------------------
# Step 7: Final 20 → 12 Jurors after Advocate Elimination
# -----------------------------
advocate_eliminated_ids = [2, 5, 7, 10, 12, 15, 17, 19]
eliminated_df = balanced_jury[balanced_jury['Juror_ID'].isin(advocate_eliminated_ids)]
remaining_df = balanced_jury[~balanced_jury['Juror_ID'].isin(advocate_eliminated_ids)]

if len(remaining_df) > 12:
    final_jury = remaining_df.nlargest(12, 'Overall_Fairness')  # Select 12 most balanced
else:
    final_jury = remaining_df

# 🎨 Enhanced Final Jury Display
print("\n" + "="*80)
print("🏛️  FINAL JURY SELECTION - 12 MEMBERS")
print("="*80)
print("📏 Bias Scale: 10 (Strongly Against) → 50 (Neutral) → 90 (Strongly For)")
print("-"*80)

for idx, row in final_jury.iterrows():
    name = row['Name']
    pros_bias = row['Predicted_Prosecution_Bias']
    def_bias = row['Predicted_Defense_Bias']
    fairness = row['Overall_Fairness']
    
    # Determine bias tendency
    if abs(pros_bias - def_bias) < 10:
        tendency = "⚖️ Highly Balanced"
        bias_icon = "🟢"
    elif pros_bias > def_bias:
        diff = pros_bias - def_bias
        if diff > 20:
            tendency = f"⚖️ Leans Prosecution (Strong +{diff:.1f})"
            bias_icon = "🔴"
        else:
            tendency = f"⚖️ Leans Prosecution (Mild +{diff:.1f})"
            bias_icon = "🟡"
    else:
        diff = def_bias - pros_bias
        if diff > 20:
            tendency = f"🛡️ Leans Defense (Strong +{diff:.1f})"
            bias_icon = "🔵"
        else:
            tendency = f"🛡️ Leans Defense (Mild +{diff:.1f})"
            bias_icon = "🟡"
    
    print(f"{bias_icon} {name}")
    print(f"   Prosecution Score: {pros_bias:.1f}/90")
    print(f"   Defense Score: {def_bias:.1f}/90")
    print(f"   Balance Rating: {fairness:.1f}/100")
    print(f"   Overall: {tendency}")
    print("-"*50)

# Summary statistics
avg_pros = final_jury['Predicted_Prosecution_Bias'].mean()
avg_def = final_jury['Predicted_Defense_Bias'].mean()
avg_fairness = final_jury['Overall_Fairness'].mean()

print(f"\n📊 JURY SUMMARY STATISTICS:")
print(f"Average Prosecution Favorability: {avg_pros:.1f}/90")
print(f"Average Defense Favorability: {avg_def:.1f}/90")
print(f"Average Balance Rating: {avg_fairness:.1f}/100")
if abs(avg_pros - avg_def) < 5:
    print("✅ JURY ASSESSMENT: Excellent Balance Achieved")
elif abs(avg_pros - avg_def) < 10:
    print("✅ JURY ASSESSMENT: Good Balance Achieved")
else:
    print("⚠️ JURY ASSESSMENT: Moderate Bias Detected")

# -----------------------------
# Step 9.1: Clustering Jurors by Bias Tendencies
# -----------------------------
from sklearn.cluster import KMeans

# Features for clustering
X_cluster = final_jury[['Predicted_Prosecution_Bias',
                       'Predicted_Defense_Bias',
                       'Predicted_Neutral_Bias']]

# Run KMeans (3 clusters: Prosecution-leaning, Defense-leaning, Neutral)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
final_jury['Cluster'] = kmeans.fit_predict(X_cluster)

# Show cluster assignments with positive interpretation
print("\n🎯 JUROR CLUSTER ANALYSIS:")
print("-"*60)
cluster_names = {0: "Balanced/Neutral", 1: "Prosecution-Leaning", 2: "Defense-Leaning"}

for cluster_id in sorted(final_jury['Cluster'].unique()):
    cluster_data = final_jury[final_jury['Cluster'] == cluster_id]
    avg_pros = cluster_data['Predicted_Prosecution_Bias'].mean()
    avg_def = cluster_data['Predicted_Defense_Bias'].mean()
    
    print(f"\n📊 Cluster {cluster_id}: {cluster_names.get(cluster_id, 'Unknown')}")
    print(f"   Members: {len(cluster_data)}")
    print(f"   Avg Prosecution Score: {avg_pros:.1f}")
    print(f"   Avg Defense Score: {avg_def:.1f}")
    print(f"   Jurors: {', '.join(cluster_data['Name'].tolist())}")

# Visualize Clusters (2D PCA)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Run PCA on features
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_cluster)

# Add PCA coordinates to a copy of final_jury
final_jury = final_jury.copy()
final_jury['PCA1'] = coords[:, 0]
final_jury['PCA2'] = coords[:, 1]

# Scatter plot
plt.figure(figsize=(10,8))
colors = ['green', 'red', 'blue']
for i, cluster_id in enumerate(sorted(final_jury['Cluster'].unique())):
    cluster_data = final_jury[final_jury['Cluster'] == cluster_id]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
               c=colors[i], label=f'Cluster {cluster_id}: {cluster_names.get(cluster_id, "Unknown")}',
               s=100, alpha=0.7, edgecolor='black')

# Add juror names
for i, row in final_jury.iterrows():
    plt.annotate(row['Name'], (row['PCA1'], row['PCA2']), 
                xytext=(5, 5), textcoords='offset points', 
                fontsize=8, ha='left')

plt.title("Final Jury - Bias Tendency Clusters", fontsize=14, fontweight='bold')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Jury Clusters", loc="best")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 9.2: Simulate Hearings (bias evolution over time)
# -----------------------------
n_hearings = int(input("\nEnter number of hearings in this trial: "))

hearing_results = []
for hearing in range(1, n_hearings+1):
    temp = final_jury.copy()
    # Small random shifts simulating experience changes (keeping in 10-90 range)
    noise_pro = np.random.normal(0, 2, len(temp))  # Smaller noise for positive scale
    noise_def = np.random.normal(0, 2, len(temp))
    
    temp['Predicted_Prosecution_Bias'] = np.clip(temp['Predicted_Prosecution_Bias'] + noise_pro, 10, 90)
    temp['Predicted_Defense_Bias'] = np.clip(temp['Predicted_Defense_Bias'] + noise_def, 10, 90)
    temp['Predicted_Neutral_Bias'] = 100 - (temp['Predicted_Prosecution_Bias'] + temp['Predicted_Defense_Bias'])/2
    temp['Hearing'] = hearing
    hearing_results.append(temp)

hearing_df = pd.concat(hearing_results)

# Plot evolution with positive scales
plt.figure(figsize=(12,7))
melted_data = hearing_df.melt(id_vars=["Name","Hearing"],
                              value_vars=['Predicted_Prosecution_Bias',
                                         'Predicted_Defense_Bias',
                                         'Predicted_Neutral_Bias'])

# Create more readable labels
melted_data['Bias_Type'] = melted_data['variable'].map({
    'Predicted_Prosecution_Bias': 'Prosecution Favorability',
    'Predicted_Defense_Bias': 'Defense Favorability', 
    'Predicted_Neutral_Bias': 'Neutral Tendency'
})

sns.lineplot(data=melted_data, x="Hearing", y="value", 
             hue="Bias_Type", style="Name", markers=True, linewidth=2)
plt.title("Juror Bias Evolution Across Hearings", fontsize=14, fontweight='bold')
plt.ylabel("Bias Score (10-90 Scale)")
plt.xlabel("Hearing Number")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 9.3: Case-topic based adjustment (only for trial stage)
# -----------------------------
case_stage = input("\nEnter case stage (preliminary/trial): ").lower()
case_topic = input("Enter case topic (Fraud, Assault, Civil Rights, etc.): ")

adjusted_jury = final_jury.copy()

if case_stage == "trial":
    print(f"\n⚖️ APPLYING CASE-SPECIFIC ADJUSTMENTS FOR '{case_topic.upper()}' CASE")
    print("-"*60)
    
    adjustment_factor = 1.0
    if "fraud" in case_topic.lower():
        adjusted_jury['Predicted_Prosecution_Bias'] *= 1.1
        adjustment_factor = 1.1
        print("📈 Prosecution bias slightly increased (Fraud cases favor prosecution)")
    elif "assault" in case_topic.lower() or "criminal" in case_topic.lower():
        adjusted_jury['Predicted_Prosecution_Bias'] *= 1.15
        adjustment_factor = 1.15
        print("📈 Prosecution bias increased (Criminal cases favor prosecution)")
    elif "civil" in case_topic.lower():
        adjusted_jury['Predicted_Defense_Bias'] *= 1.1
        adjustment_factor = 1.1
        print("📉 Defense bias slightly increased (Civil cases favor defendants)")
    
    # Keep values in valid range
    adjusted_jury['Predicted_Prosecution_Bias'] = np.clip(adjusted_jury['Predicted_Prosecution_Bias'], 10, 90)
    adjusted_jury['Predicted_Defense_Bias'] = np.clip(adjusted_jury['Predicted_Defense_Bias'], 10, 90)
    
    # Recalculate neutral and fairness
    total = (adjusted_jury['Predicted_Prosecution_Bias'] + adjusted_jury['Predicted_Defense_Bias'])
    adjusted_jury['Predicted_Neutral_Bias'] = np.clip(100 - total/2, 10, 90)
    adjusted_jury['Overall_Fairness'] = 100 - abs(adjusted_jury['Predicted_Prosecution_Bias'] - 
                                                  adjusted_jury['Predicted_Defense_Bias'])

    print(f"\n🎯 FINAL ADJUSTED JURY FOR {case_topic.upper()} CASE:")
    print("="*70)
    
    for idx, row in adjusted_jury.iterrows():
        name = row['Name']
        pros_bias = row['Predicted_Prosecution_Bias']
        def_bias = row['Predicted_Defense_Bias']
        fairness = row['Overall_Fairness']
        
        print(f"👤 {name}")
        print(f"   Prosecution: {pros_bias:.1f}/90 | Defense: {def_bias:.1f}/90 | Balance: {fairness:.1f}/100")
        
        # Show change from original
        if adjustment_factor != 1.0:
            if "prosecution" in case_topic.lower() or "fraud" in case_topic.lower() or "criminal" in case_topic.lower() or "assault" in case_topic.lower():
                change = (pros_bias / adjustment_factor) - pros_bias
                print(f"   📊 Prosecution score adjusted by: {change:+.1f}")
        print("-"*50)
        
else:
    print("\n✅ No case-topic adjustment applied (preliminary hearing).")
    print("Final jury bias scores remain unchanged.")