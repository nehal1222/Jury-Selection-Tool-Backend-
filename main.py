import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.sparse import hstack
import joblib
import os
import random

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

print("🏛️ ENHANCED JURY SELECTION SYSTEM WITH IMPROVED AI BIAS PREDICTION")
print("="*80)

# -----------------------------
# Step 1: Load and Prepare Data
# -----------------------------
try:
    df = pd.read_csv("data.csv")
    print(f"✅ Data loaded successfully: {len(df)} potential jurors")
except FileNotFoundError:
    print("⚠️ data.csv not found. Creating enhanced sample data...")
    np.random.seed(42)
    
    n_samples = 2000
    names = [f'Juror_{i:03d}' for i in range(1, n_samples + 1)]
    
    # Create more realistic correlated features
    sample_data = {
        'Name': names,
        'Age': np.random.randint(21, 70, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples),
        'Education': np.random.choice(['High School', 'Graduate', 'Postgraduate'], n_samples, p=[0.3, 0.5, 0.2]),
        'Profession': np.random.choice(['Teacher', 'Engineer', 'Doctor', 'Lawyer', 'Business', 'Retired', 'Student', 'Nurse', 'Police'], n_samples),
        'Political_Leaning': np.random.uniform(0, 1, n_samples),
        'Income_Level': np.random.choice(['Low', 'Middle', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
        'Previous_Jury_Experience': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Trust_in_Police': np.random.uniform(0, 1, n_samples),
        'Trust_in_Courts': np.random.uniform(0, 1, n_samples),
        'Questionnaire': [f'Justice questionnaire response {i} with various opinions' for i in range(n_samples)]
    }
    df = pd.DataFrame(sample_data)
    print(f"✅ Enhanced sample data created: {len(df)} potential jurors")

df['Juror_ID'] = range(1, len(df) + 1)
df_original = df.copy()

# -----------------------------
# Step 2: Enhanced Feature Engineering
# -----------------------------
print("\n📋 ENHANCED PREPROCESSING & FEATURE ENGINEERING...")
categorical_cols = ['Gender', 'Region', 'Education', 'Profession', 'Income_Level']
available_categorical = [col for col in categorical_cols if col in df.columns]
df_encoded = pd.get_dummies(df, columns=available_categorical)

# Create interaction features
if 'Age' in df_encoded.columns:
    df_encoded['Age_squared'] = df_encoded['Age'] ** 2
    df_encoded['Age_normalized'] = (df_encoded['Age'] - df_encoded['Age'].min()) / (df_encoded['Age'].max() - df_encoded['Age'].min())

if 'Political_Leaning' in df_encoded.columns:
    df_encoded['Political_Extreme'] = np.abs(df_encoded['Political_Leaning'] - 0.5)  # Distance from center

# Process questionnaire with enhanced text features
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

if 'Questionnaire' in df_encoded.columns:
    df_encoded['Questionnaire_clean'] = df_encoded['Questionnaire'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=30, min_df=2, max_df=0.8, ngram_range=(1,2))
    clean_text = df_encoded['Questionnaire_clean'].fillna('')
    text_features = vectorizer.fit_transform(clean_text)
    print(f"Text features shape: {text_features.shape}")
else:
    vectorizer = TfidfVectorizer(max_features=30)
    dummy_text = ['sample text'] * len(df_encoded)
    text_features = vectorizer.fit_transform(dummy_text)

# Prepare numerical features
base_numerical = ['Age', 'Political_Leaning', 'Trust_in_Police', 'Trust_in_Courts', 'Previous_Jury_Experience']
engineered_numerical = ['Age_squared', 'Age_normalized', 'Political_Extreme']
all_numerical = [col for col in base_numerical + engineered_numerical if col in df_encoded.columns]

if all_numerical:
    X_numerical = df_encoded[all_numerical].values
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)
    print(f"Numerical features: {len(all_numerical)}")
else:
    X_numerical_scaled = np.zeros((len(df_encoded), 1))

# Get categorical features
categorical_features = []
for cat in available_categorical:
    categorical_features.extend([col for col in df_encoded.columns if col.startswith(f'{cat}_')])

if categorical_features:
    X_categorical = df_encoded[categorical_features].values
    print(f"Categorical features: {len(categorical_features)}")
else:
    X_categorical = np.zeros((len(df_encoded), 1))

# Combine all features
X_combined = np.hstack([X_numerical_scaled, X_categorical])
X = hstack([X_combined, text_features])
print(f"Total feature matrix shape: {X.shape}")

# -----------------------------
# Step 3: Create More Realistic Target Variables
# -----------------------------
print("\n🎯 CREATING REALISTIC BIAS TARGETS...")

# Create more realistic bias scores with stronger feature relationships
def create_prosecution_bias(df_enc):
    base_score = 0.5
    
    # Political leaning effect (stronger for conservative)
    if 'Political_Leaning' in df_enc.columns:
        base_score += (df_enc['Political_Leaning'] - 0.5) * 0.4
    
    # Profession effects
    if 'Profession_Police' in df_enc.columns:
        base_score += df_enc['Profession_Police'] * 0.3
    if 'Profession_Lawyer' in df_enc.columns:
        base_score += df_enc['Profession_Lawyer'] * 0.2
    if 'Profession_Business' in df_enc.columns:
        base_score += df_enc['Profession_Business'] * 0.15
    
    # Trust in police effect
    if 'Trust_in_Police' in df_enc.columns:
        base_score += (df_enc['Trust_in_Police'] - 0.5) * 0.25
    
    # Age effect (older tend to be more prosecution-leaning)
    if 'Age' in df_enc.columns:
        age_normalized = (df_enc['Age'] - 30) / 40  # Center around 30
        base_score += age_normalized * 0.15
    
    # Add controlled noise
    base_score += np.random.normal(0, 0.1, len(df_enc))
    
    return np.clip(base_score, 0.1, 0.9)

def create_defense_bias(df_enc):
    base_score = 0.4
    
    # Political leaning effect (stronger for liberal)
    if 'Political_Leaning' in df_enc.columns:
        base_score += (0.5 - df_enc['Political_Leaning']) * 0.4
    
    # Profession effects
    if 'Profession_Teacher' in df_enc.columns:
        base_score += df_enc['Profession_Teacher'] * 0.25
    if 'Profession_Student' in df_enc.columns:
        base_score += df_enc['Profession_Student'] * 0.2
    
    # Education effect
    if 'Education_Postgraduate' in df_enc.columns:
        base_score += df_enc['Education_Postgraduate'] * 0.2
    if 'Education_Graduate' in df_enc.columns:
        base_score += df_enc['Education_Graduate'] * 0.1
    
    # Trust in courts (inverse effect)
    if 'Trust_in_Courts' in df_enc.columns:
        base_score += (0.5 - df_enc['Trust_in_Courts']) * 0.2
    
    # Age effect (younger tend to be more defense-leaning)
    if 'Age' in df_enc.columns:
        age_normalized = (45 - df_enc['Age']) / 40  # Inverse age effect
        base_score += age_normalized * 0.1
    
    # Add controlled noise
    base_score += np.random.normal(0, 0.1, len(df_enc))
    
    return np.clip(base_score, 0.1, 0.9)

# Generate target variables
np.random.seed(42)
y_prosecution = create_prosecution_bias(df_encoded)
y_defense = create_defense_bias(df_encoded)
y_neutral = 1 - 0.5 * (y_prosecution + y_defense) + np.random.normal(0, 0.05, len(df_encoded))
y_neutral = np.clip(y_neutral, 0.1, 0.9)

print(f"Target correlations:")
print(f"  Prosecution vs Defense: {np.corrcoef(y_prosecution, y_defense)[0,1]:.3f}")
print(f"  Prosecution vs Neutral: {np.corrcoef(y_prosecution, y_neutral)[0,1]:.3f}")

# -----------------------------
# Step 4: Enhanced Model Training with Multiple Algorithms
# -----------------------------
print("\n🤖 TRAINING ENHANCED AI MODELS...")
X_dense = X.toarray() if hasattr(X, 'toarray') else X

# Split data
X_train, X_test, y_pros_train, y_pros_test, y_def_train, y_def_test, y_neut_train, y_neut_test = train_test_split(
    X_dense, y_prosecution, y_defense, y_neutral, test_size=0.2, random_state=42
)

# Define multiple models to try
models = {
    'Ridge': Ridge(alpha=10.0, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42)
}

best_models = {}
results = {}

for target_name, y_train, y_test in [
    ('Prosecution', y_pros_train, y_pros_test),
    ('Defense', y_def_train, y_def_test),
    ('Neutral', y_neut_train, y_neut_test)
]:
    print(f"\n🎯 Training models for {target_name} Bias:")
    
    best_r2 = -np.inf
    best_model = None
    target_results = {}
    
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            target_results[model_name] = {'r2': r2, 'mae': mae, 'mse': mse}
            
            print(f"   {model_name:15s} | R²: {r2:6.3f} | MAE: {mae:.3f} | MSE: {mse:.3f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                
        except Exception as e:
            print(f"   {model_name:15s} | Failed: {e}")
    
    best_models[target_name] = best_model
    results[target_name] = target_results
    print(f"   ✅ Best model for {target_name}: {best_r2:.3f} R²")

# -----------------------------
# Step 5: Generate Enhanced Predictions
# -----------------------------
print("\n📊 GENERATING ENHANCED BIAS PREDICTIONS...")

predictions = {}
for target_name, model in best_models.items():
    if model is not None:
        pred = model.predict(X_dense)
        predictions[f'{target_name}_Bias'] = pred

# Scale predictions to meaningful ranges
scaler_range = (15, 85)  # More realistic range
scalers = {}

for bias_type, scores in predictions.items():
    scaler = MinMaxScaler(feature_range=scaler_range)
    scaled_scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    df_original[f'Predicted_{bias_type}'] = scaled_scores
    scalers[bias_type] = scaler

# Calculate overall fairness and confidence
if 'Predicted_Prosecution_Bias' in df_original.columns and 'Predicted_Defense_Bias' in df_original.columns:
    df_original['Bias_Difference'] = abs(df_original['Predicted_Prosecution_Bias'] - df_original['Predicted_Defense_Bias'])
    df_original['Overall_Fairness'] = 100 - df_original['Bias_Difference']
    df_original['Prediction_Confidence'] = np.random.uniform(0.6, 0.95, len(df_original))  # Simulated confidence

# -----------------------------
# Step 6: Enhanced Jury Selection with Strategic Sampling
# -----------------------------
print(f"\n🎲 STRATEGIC JURY POOL SELECTION...")
print("="*80)

# Select jury pool with better balance
def select_balanced_jury_pool(df, n_jurors=20):
    """Select a balanced jury pool considering bias diversity"""
    
    # Separate jurors by bias tendency
    if 'Predicted_Prosecution_Bias' in df.columns and 'Predicted_Defense_Bias' in df.columns:
        bias_diff = df['Predicted_Prosecution_Bias'] - df['Predicted_Defense_Bias']
        
        pro_prosecution = df[bias_diff > 8].copy()
        pro_defense = df[bias_diff < -8].copy()
        neutral = df[abs(bias_diff) <= 8].copy()
        
        # Aim for balanced representation
        n_pros = min(len(pro_prosecution), n_jurors // 3)
        n_def = min(len(pro_defense), n_jurors // 3)
        n_neut = n_jurors - n_pros - n_def
        n_neut = min(len(neutral), n_neut)
        
        selected = []
        
        if n_pros > 0 and len(pro_prosecution) > 0:
            selected.extend(pro_prosecution.sample(n_pros, random_state=42).index.tolist())
        
        if n_def > 0 and len(pro_defense) > 0:
            selected.extend(pro_defense.sample(n_def, random_state=42).index.tolist())
            
        if n_neut > 0 and len(neutral) > 0:
            selected.extend(neutral.sample(n_neut, random_state=42).index.tolist())
        
        # Fill remaining slots randomly if needed
        remaining_needed = n_jurors - len(selected)
        if remaining_needed > 0:
            available = df.drop(selected).index.tolist()
            if len(available) >= remaining_needed:
                additional = np.random.choice(available, remaining_needed, replace=False)
                selected.extend(additional)
        
        return df.loc[selected[:n_jurors]]
    
    else:
        # Fallback to random selection
        return df.sample(min(n_jurors, len(df)), random_state=42)

jury_pool = select_balanced_jury_pool(df_original, 20)
jury_pool = jury_pool.reset_index(drop=True)

print(f"✅ Selected {len(jury_pool)} jurors for strategically balanced jury pool")

# Enhanced jury analysis
print(f"\n📋 STRATEGIC JURY POOL - ENHANCED ANALYSIS:")
print("="*80)

for idx, (_, juror) in enumerate(jury_pool.iterrows(), 1):
    pros_bias = juror.get('Predicted_Prosecution_Bias', 50)
    def_bias = juror.get('Predicted_Defense_Bias', 50)
    neutral_bias = juror.get('Predicted_Neutral_Bias', 50)
    fairness = juror.get('Overall_Fairness', 50)
    confidence = juror.get('Prediction_Confidence', 0.8)
    
    # Enhanced classification
    bias_diff = pros_bias - def_bias
    
    if bias_diff > 10:
        tendency = "STRONG PRO-PROSECUTION"
        icon = "🔴"
        risk_level = "HIGH PROSECUTION RISK"
    elif bias_diff > 5:
        tendency = "MODERATE PRO-PROSECUTION"
        icon = "🟠"
        risk_level = "MODERATE PROSECUTION LEAN"
    elif bias_diff < -10:
        tendency = "STRONG PRO-DEFENSE"
        icon = "🔵"
        risk_level = "HIGH DEFENSE RISK"
    elif bias_diff < -5:
        tendency = "MODERATE PRO-DEFENSE"
        icon = "🟣"
        risk_level = "MODERATE DEFENSE LEAN"
    else:
        tendency = "BALANCED/NEUTRAL"
        icon = "🟢"
        risk_level = "LOW BIAS RISK"
    
    print(f"\n{icon} JUROR #{idx:2d}: {juror['Name']}")
    print(f"   📊 Demographics: {juror.get('Age', 'N/A')}y, {juror.get('Gender', 'N/A')}, {juror.get('Education', 'N/A')}")
    print(f"   💼 Background: {juror.get('Profession', 'N/A')} | {juror.get('Region', 'N/A')} region")
    print(f"   ⚖️  Bias Scores: Pros={pros_bias:.1f} | Def={def_bias:.1f} | Neutral={neutral_bias:.1f}")
    print(f"   📈 Fairness: {fairness:.1f}/100 | Confidence: {confidence:.1%}")
    print(f"   🎯 ASSESSMENT: {risk_level}")
    print(f"   📋 CLASSIFICATION: {tendency}")
    print("-" * 80)

# Enhanced summary with strategic insights
if 'Predicted_Prosecution_Bias' in jury_pool.columns:
    bias_diffs = jury_pool['Predicted_Prosecution_Bias'] - jury_pool['Predicted_Defense_Bias']
    
    strong_pros = len(bias_diffs[bias_diffs > 10])
    mod_pros = len(bias_diffs[(bias_diffs > 5) & (bias_diffs <= 10)])
    balanced = len(bias_diffs[abs(bias_diffs) <= 5])
    mod_def = len(bias_diffs[(bias_diffs < -5) & (bias_diffs >= -10)])
    strong_def = len(bias_diffs[bias_diffs < -10])
    
    print(f"\n📊 STRATEGIC JURY POOL ANALYSIS:")
    print(f"   🔴 Strong Pro-Prosecution: {strong_pros} ({strong_pros/len(jury_pool)*100:.1f}%)")
    print(f"   🟠 Moderate Pro-Prosecution: {mod_pros} ({mod_pros/len(jury_pool)*100:.1f}%)")
    print(f"   🟢 Balanced/Neutral: {balanced} ({balanced/len(jury_pool)*100:.1f}%)")
    print(f"   🟣 Moderate Pro-Defense: {mod_def} ({mod_def/len(jury_pool)*100:.1f}%)")
    print(f"   🔵 Strong Pro-Defense: {strong_def} ({strong_def/len(jury_pool)*100:.1f}%)")
    
    print(f"\n📈 POOL STATISTICS:")
    print(f"   📊 Average Prosecution Bias: {jury_pool['Predicted_Prosecution_Bias'].mean():.1f}")
    print(f"   📊 Average Defense Bias: {jury_pool['Predicted_Defense_Bias'].mean():.1f}")
    print(f"   ⚖️ Average Fairness Score: {jury_pool['Overall_Fairness'].mean():.1f}")
    print(f"   📊 Bias Balance (closer to 0 is better): {bias_diffs.mean():.1f}")

# -----------------------------
# Step 6: Enhanced SHAP Analysis
# -----------------------------
print("\n🔍 RUNNING ENHANCED SHAP EXPLAINABILITY ANALYSIS...")

try:
    import shap
    
    # Create comprehensive feature names
    feature_names = []
    
    # Add numerical feature names
    feature_names.extend(all_numerical)
    
    # Add categorical feature names
    feature_names.extend(categorical_features)
    
    # Add text feature names
    if hasattr(vectorizer, 'get_feature_names_out'):
        text_feature_names = [f'text_{name}' for name in vectorizer.get_feature_names_out()]
    else:
        text_feature_names = [f'text_{i}' for i in range(text_features.shape[1])]
    feature_names.extend(text_feature_names)
    
    print(f"📊 Total features for SHAP analysis: {len(feature_names)}")
    
    # Use sample for SHAP analysis (for performance)
    sample_size = min(100, len(X_dense))
    sample_indices = np.random.choice(len(X_dense), sample_size, replace=False)
    X_shap_sample = X_dense[sample_indices]
    
    # Analyze each best model
    shap_results = {}
    
    for target_name, model in best_models.items():
        if model is not None:
            print(f"\n🎯 SHAP Analysis for {target_name} Bias Model:")
            
            try:
                # Choose appropriate explainer based on model type
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_shap_sample[:20])  # Smaller sample for tree models
                else:
                    # Linear models
                    explainer = shap.LinearExplainer(model, X_shap_sample)
                    shap_values = explainer.shap_values(X_shap_sample[:20])
                
                # Calculate feature importance
                feature_importance = np.abs(shap_values).mean(0)
                
                # Get top features
                top_indices = np.argsort(feature_importance)[-15:]  # Top 15 features
                
                print(f"   ✅ Top 15 {target_name} Bias Factors:")
                for i, idx in enumerate(reversed(top_indices)):
                    if idx < len(feature_names):
                        feature_name = feature_names[idx]
                        impact = feature_importance[idx]
                        print(f"      {i+1:2d}. {feature_name:<30} | Impact: {impact:.4f}")
                
                # Store results
                shap_results[target_name] = {
                    'values': shap_values,
                    'importance': feature_importance,
                    'top_indices': top_indices,
                    'explainer': explainer
                }
                
            except Exception as e:
                print(f"   ❌ SHAP analysis failed for {target_name}: {e}")
                continue
    
    # Create SHAP visualizations if successful
    if shap_results:
        print(f"\n📊 Creating SHAP visualizations...")
        
        # Create SHAP summary plots
        n_targets = len(shap_results)
        if n_targets > 0:
            fig_shap, axes_shap = plt.subplots(1, min(n_targets, 2), figsize=(15, 8))
            if n_targets == 1:
                axes_shap = [axes_shap]
            
            for idx, (target_name, shap_data) in enumerate(list(shap_results.items())[:2]):
                ax = axes_shap[idx] if len(axes_shap) > idx else axes_shap[0]
                
                # Get top features for this target
                top_indices = shap_data['top_indices'][-10:]  # Top 10 for visualization
                top_names = [feature_names[i][:20] if i < len(feature_names) else f"f_{i}" 
                           for i in reversed(top_indices)]
                top_values = [shap_data['importance'][i] for i in reversed(top_indices)]
                
                # Create horizontal bar plot
                bars = ax.barh(range(len(top_names)), top_values, 
                              color='red' if 'Prosecution' in target_name else 'blue', 
                              alpha=0.7, edgecolor='black', linewidth=0.5)
                
                ax.set_yticks(range(len(top_names)))
                ax.set_yticklabels(top_names, fontsize=10)
                ax.set_xlabel('SHAP Feature Importance', fontsize=12)
                ax.set_title(f'Top {target_name} Bias Factors', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, top_values)):
                    ax.text(value + max(top_values)*0.01, bar.get_y() + bar.get_height()/2, 
                           f'{value:.3f}', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.suptitle('ENHANCED SHAP FEATURE IMPORTANCE ANALYSIS', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.show()
            
        # Feature interaction analysis (if multiple targets)
        if len(shap_results) >= 2:
            print(f"\n🔄 SHAP Feature Interaction Analysis:")
            
            # Compare feature importance between prosecution and defense models
            if 'Prosecution' in shap_results and 'Defense' in shap_results:
                pros_importance = shap_results['Prosecution']['importance']
                def_importance = shap_results['Defense']['importance']
                
                # Find features that have high impact on both
                combined_importance = pros_importance + def_importance
                top_combined = np.argsort(combined_importance)[-10:]
                
                print("   🎯 Features with high impact on BOTH biases:")
                for i, idx in enumerate(reversed(top_combined)):
                    if idx < len(feature_names):
                        feature_name = feature_names[idx]
                        pros_impact = pros_importance[idx]
                        def_impact = def_importance[idx]
                        combined_impact = combined_importance[idx]
                        
                        print(f"      {i+1:2d}. {feature_name:<25} | "
                              f"Pros: {pros_impact:.4f} | Def: {def_impact:.4f} | "
                              f"Combined: {combined_impact:.4f}")
                
                # Correlation between feature importances
                importance_corr = np.corrcoef(pros_importance, def_importance)[0,1]
                print(f"\n   📊 Feature importance correlation: {importance_corr:.3f}")
                if importance_corr > 0.5:
                    print("      → High correlation: Similar features drive both biases")
                elif importance_corr < -0.2:
                    print("      → Negative correlation: Different features drive opposing biases")
                else:
                    print("      → Moderate correlation: Some overlap in important features")

    print("✅ Enhanced SHAP analysis completed!")
    
except ImportError:
    print("❌ SHAP not installed. Install with: pip install shap")
    print("   Continuing without SHAP analysis...")
except Exception as e:
    print(f"⚠️ SHAP analysis encountered an error: {e}")
    print("   Continuing with remaining analysis...")

# Enhanced visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Enhanced bias comparison
if 'Predicted_Prosecution_Bias' in jury_pool.columns:
    juror_names = [f"J{i+1}" for i in range(len(jury_pool))]
    x_pos = np.arange(len(juror_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, jury_pool['Predicted_Prosecution_Bias'], width, 
                   label='Prosecution Bias', color='crimson', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, jury_pool['Predicted_Defense_Bias'], width, 
                   label='Defense Bias', color='steelblue', alpha=0.8)
    
    ax1.set_xlabel('Jurors', fontsize=12)
    ax1.set_ylabel('Bias Score', fontsize=12)
    ax1.set_title('Enhanced Juror Bias Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(juror_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

# 2. Bias distribution with better categories
if 'Predicted_Prosecution_Bias' in jury_pool.columns:
    categories = ['Strong\nPro-Prosecution', 'Moderate\nPro-Prosecution', 'Balanced', 
                 'Moderate\nPro-Defense', 'Strong\nPro-Defense']
    counts = [strong_pros, mod_pros, balanced, mod_def, strong_def]
    colors = ['darkred', 'orange', 'green', 'purple', 'darkblue']
    
    wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    ax2.set_title('Enhanced Bias Distribution', fontsize=14, fontweight='bold')

# 3. Model performance comparison
if results:
    model_names = list(models.keys())
    prosecution_r2 = [results['Prosecution'][model]['r2'] if model in results['Prosecution'] else 0 
                     for model in model_names]
    defense_r2 = [results['Defense'][model]['r2'] if model in results['Defense'] else 0 
                 for model in model_names]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    ax3.bar(x_pos - width/2, prosecution_r2, width, label='Prosecution Model', 
           color='crimson', alpha=0.7)
    ax3.bar(x_pos + width/2, defense_r2, width, label='Defense Model', 
           color='steelblue', alpha=0.7)
    
    ax3.set_xlabel('Models', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# 4. Fairness vs Confidence scatter
if 'Overall_Fairness' in jury_pool.columns and 'Prediction_Confidence' in jury_pool.columns:
    scatter = ax4.scatter(jury_pool['Overall_Fairness'], jury_pool['Prediction_Confidence']*100,
                         c=jury_pool['Predicted_Prosecution_Bias'], cmap='RdBu_r', 
                         s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    ax4.set_xlabel('Fairness Score', fontsize=12)
    ax4.set_ylabel('Prediction Confidence (%)', fontsize=12)
    ax4.set_title('Fairness vs Prediction Confidence', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Prosecution Bias →', fontsize=10)

plt.tight_layout()
plt.suptitle('ENHANCED JURY SELECTION ANALYSIS DASHBOARD', 
             fontsize=16, fontweight='bold', y=0.98)
plt.show()

print(f"\n✅ ENHANCED JURY SELECTION ANALYSIS COMPLETE!")
print(f"🎯 System analyzed {len(jury_pool)} strategically selected jurors")
print(f"🤖 Best model performance: {max([max(target_results.values(), key=lambda x: x['r2'])['r2'] for target_results in results.values()]):.3f} R²")
print(f"📋 Strategic selection ensures balanced representation across bias categories")
print("="*80)




    