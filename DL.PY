import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import random

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, GlobalMaxPooling1D, Input, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Advanced NLP imports
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")

# Alternative NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import nltk
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

warnings.filterwarnings('ignore')

print("🏛️ ADVANCED JURY SELECTION SYSTEM")
print("Deep Learning + Advanced NLP + SHAP Explainability")
print("="*70)

# -----------------------------
# PURPOSE 1: ADVANCED NLP FOR QUESTIONNAIRE ANALYSIS
# -----------------------------
class AdvancedNLPProcessor:
    """
    PURPOSE: Extract psychological insights from questionnaire responses
    - Sentiment analysis: Detect emotional tone (optimistic vs pessimistic jurors)
    - Personality traits: Identify authoritarian vs liberal language patterns
    - Linguistic complexity: Education level indicators
    - Bias detection: Implicit biases in language choices
    """
    
    def __init__(self):
        self.tokenizer = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 3))
        
        # Initialize transformer model if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
                print("✅ BERT model loaded for advanced text analysis")
            except:
                self.bert_tokenizer = None
                self.bert_model = None
                print("⚠️ BERT model failed to load, using alternatives")
        
    def extract_psychological_features(self, texts):
        """Extract psychological and linguistic features from questionnaire responses"""
        features = {}
        
        # 1. Sentiment Analysis (PURPOSE: Detect emotional stability)
        if self.sentiment_analyzer:
            sentiments = []
            for text in texts:
                if pd.isna(text) or text == '':
                    sentiments.append({'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0})
                else:
                    sentiments.append(self.sentiment_analyzer.polarity_scores(str(text)))
            
            features['sentiment_positive'] = [s['pos'] for s in sentiments]
            features['sentiment_negative'] = [s['neg'] for s in sentiments]
            features['sentiment_compound'] = [s['compound'] for s in sentiments]
        else:
            # Fallback sentiment using TextBlob
            features['sentiment_polarity'] = [
                TextBlob(str(text) if not pd.isna(text) else '').sentiment.polarity 
                for text in texts
            ]
            features['sentiment_subjectivity'] = [
                TextBlob(str(text) if not pd.isna(text) else '').sentiment.subjectivity 
                for text in texts
            ]
        
        # 2. Linguistic Complexity (PURPOSE: Education/intelligence indicators)
        features['text_length'] = [len(str(text)) if not pd.isna(text) else 0 for text in texts]
        features['word_count'] = [len(str(text).split()) if not pd.isna(text) else 0 for text in texts]
        features['avg_word_length'] = [
            np.mean([len(word) for word in str(text).split()]) if not pd.isna(text) and len(str(text).split()) > 0 else 0 
            for text in texts
        ]
        
        # 3. Authority/Control Language (PURPOSE: Detect authoritarian tendencies)
        authority_words = ['control', 'order', 'discipline', 'rules', 'law', 'punishment', 'strict', 'enforce']
        liberal_words = ['freedom', 'rights', 'fairness', 'equality', 'justice', 'compassion', 'understanding']
        
        features['authority_score'] = [
            sum(1 for word in authority_words if word in str(text).lower()) / max(1, len(str(text).split()))
            for text in texts
        ]
        features['liberal_score'] = [
            sum(1 for word in liberal_words if word in str(text).lower()) / max(1, len(str(text).split()))
            for text in texts
        ]
        
        # 4. Certainty Language (PURPOSE: Detect dogmatic vs open-minded thinking)
        certainty_words = ['always', 'never', 'definitely', 'absolutely', 'certain', 'sure', 'obvious']
        uncertainty_words = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'sometimes', 'depends']
        
        features['certainty_score'] = [
            sum(1 for word in certainty_words if word in str(text).lower()) / max(1, len(str(text).split()))
            for text in texts
        ]
        features['uncertainty_score'] = [
            sum(1 for word in uncertainty_words if word in str(text).lower()) / max(1, len(str(text).split()))
            for text in texts
        ]
        
        return pd.DataFrame(features)
    
    def get_bert_embeddings(self, texts, max_length=128):
        """PURPOSE: Get deep contextual embeddings for semantic understanding"""
        if not TRANSFORMERS_AVAILABLE or self.bert_tokenizer is None:
            return np.zeros((len(texts), 768))  # Return dummy embeddings
        
        embeddings = []
        for text in texts:
            if pd.isna(text) or text == '':
                embeddings.append(np.zeros(768))
                continue
                
            inputs = self.bert_tokenizer(str(text), return_tensors='pt', 
                                       truncation=True, padding=True, max_length=max_length)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use mean pooling of last hidden state
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)

# -----------------------------
# PURPOSE 2: DEEP LEARNING MODELS FOR COMPLEX PATTERN RECOGNITION
# -----------------------------
class DeepBiasPredictor:
    """
    PURPOSE: Capture complex non-linear relationships between features and bias
    - Neural networks can learn intricate patterns humans might miss
    - Multiple hidden layers for hierarchical feature learning
    - Dropout for regularization and better generalization
    - Multi-task learning (predict prosecution, defense, and neutral bias simultaneously)
    """
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.text_model = None
        
    def build_tabular_model(self, numerical_features, categorical_features):
        """PURPOSE: Handle structured data (age, profession, etc.) with deep networks"""
        
        # Numerical input branch
        numerical_input = Input(shape=(numerical_features,), name='numerical')
        numerical_dense = Dense(64, activation='relu')(numerical_input)
        numerical_dense = Dropout(0.3)(numerical_dense)
        numerical_dense = Dense(32, activation='relu')(numerical_dense)
        
        # Categorical input branch
        categorical_input = Input(shape=(categorical_features,), name='categorical')
        categorical_dense = Dense(32, activation='relu')(categorical_input)
        categorical_dense = Dropout(0.2)(categorical_dense)
        
        # Combine branches
        combined = concatenate([numerical_dense, categorical_dense])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(32, activation='relu')(combined)
        
        # Multi-task outputs
        prosecution_output = Dense(1, activation='sigmoid', name='prosecution')(combined)
        defense_output = Dense(1, activation='sigmoid', name='defense')(combined)
        neutral_output = Dense(1, activation='sigmoid', name='neutral')(combined)
        
        model = Model(inputs=[numerical_input, categorical_input], 
                     outputs=[prosecution_output, defense_output, neutral_output])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'prosecution': 'mse', 'defense': 'mse', 'neutral': 'mse'},
            loss_weights={'prosecution': 1.0, 'defense': 1.0, 'neutral': 0.5},
            metrics=['mae']
        )
        
        return model
    
    def build_text_model(self, vocab_size, max_length, embedding_dim=100):
        """PURPOSE: Deep learning for questionnaire text analysis"""
        
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            LSTM(64, dropout=0.3, recurrent_dropout=0.3),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='sigmoid', name='bias_outputs')  # [prosecution, defense, neutral]
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

# -----------------------------
# PURPOSE 3: ENHANCED SHAP FOR DEEP LEARNING INTERPRETABILITY
# -----------------------------
class DeepSHAPAnalyzer:
    """
    PURPOSE: Explain deep learning model predictions
    - DeepExplainer for neural networks
    - Feature interaction analysis
    - Individual prediction explanations
    - Global feature importance for model transparency
    """
    
    def __init__(self):
        try:
            import shap
            self.shap = shap
            self.explainer = None
        except ImportError:
            self.shap = None
            print("⚠️ SHAP not available for deep model explanations")
    
    def create_deep_explainer(self, model, background_data):
        """PURPOSE: Create SHAP explainer for neural network"""
        if self.shap is None:
            return None
            
        self.explainer = self.shap.DeepExplainer(model, background_data)
        return self.explainer
    
    def explain_predictions(self, model, data, feature_names, sample_size=10):
        """PURPOSE: Explain why the model made specific predictions"""
        if self.shap is None or self.explainer is None:
            return None
            
        # Get SHAP values for sample
        sample_indices = np.random.choice(len(data), min(sample_size, len(data)), replace=False)
        sample_data = data[sample_indices]
        
        shap_values = self.explainer.shap_values(sample_data)
        
        return shap_values, sample_indices

# -----------------------------
# MAIN IMPLEMENTATION
# -----------------------------

def main():
    # Load data (same as before)
    try:
        df = pd.read_csv("data.csv")
        print(f"✅ Data loaded: {len(df)} potential jurors")
    except FileNotFoundError:
        print("Creating enhanced sample data with questionnaire responses...")
        np.random.seed(42)
        
        # More realistic questionnaire responses
        questionnaire_templates = [
            "I believe in strict law enforcement and following rules exactly as written.",
            "Everyone deserves a fair trial regardless of the circumstances.",
            "I think context matters more than rigid rule-following.",
            "The system should protect society from dangerous individuals.",
            "Individual rights are paramount in any justice system.",
            "I prefer evidence-based decisions over emotional arguments.",
            "Personal responsibility is key to a functioning society.",
            "We must consider social and economic factors in justice.",
            "I trust law enforcement to do their job properly.",
            "Everyone makes mistakes and deserves second chances."
        ]
        
        sample_data = {
            'Name': [f'Juror_{i:03d}' for i in range(1, 501)],
            'Age': np.random.randint(21, 70, 500),
            'Gender': np.random.choice(['Male', 'Female'], 500),
            'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 500),
            'Education': np.random.choice(['High School', 'Graduate', 'Postgraduate'], 500, p=[0.3, 0.5, 0.2]),
            'Profession': np.random.choice(['Teacher', 'Engineer', 'Doctor', 'Lawyer', 'Business', 'Retired', 'Student', 'Nurse', 'Police'], 500),
            'Political_Leaning': np.random.uniform(0, 1, 500),
            'Questionnaire': [np.random.choice(questionnaire_templates) + f" Additional thoughts {i}." for i in range(500)]
        }
        df = pd.DataFrame(sample_data)
        print(f"✅ Enhanced sample data created: {len(df)} jurors with detailed questionnaires")
    
    df['Juror_ID'] = range(1, len(df) + 1)
    df_original = df.copy()
    
    # Initialize advanced NLP processor
    print("\n🔍 INITIALIZING ADVANCED NLP ANALYSIS...")
    nlp_processor = AdvancedNLPProcessor()
    
    # Extract psychological features from questionnaires
    if 'Questionnaire' in df.columns:
        psychological_features = nlp_processor.extract_psychological_features(df['Questionnaire'])
        print(f"✅ Extracted {len(psychological_features.columns)} psychological features")
        
        # Get BERT embeddings for deep semantic understanding
        print("🧠 Generating BERT embeddings for semantic analysis...")
        bert_embeddings = nlp_processor.get_bert_embeddings(df['Questionnaire'])
        print(f"✅ Generated embeddings shape: {bert_embeddings.shape}")
        
        # Add to dataframe
        for col in psychological_features.columns:
            df[col] = psychological_features[col]
    
    # Prepare features for deep learning
    categorical_cols = ['Gender', 'Region', 'Education', 'Profession']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Separate numerical and categorical features
    numerical_features = ['Age', 'Political_Leaning'] + list(psychological_features.columns)
    categorical_features = [col for col in df_encoded.columns if any(cat in col for cat in ['Gender_', 'Region_', 'Education_', 'Profession_'])]
    
    X_numerical = df_encoded[numerical_features].values
    X_categorical = df_encoded[categorical_features].values
    
    # Scale numerical features
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)
    
    print(f"📊 Feature dimensions:")
    print(f"   Numerical features: {X_numerical_scaled.shape}")
    print(f"   Categorical features: {X_categorical.shape}")
    print(f"   BERT embeddings: {bert_embeddings.shape}")
    
    # Create synthetic targets with realistic patterns
    np.random.seed(42)
    
    # Prosecution bias influenced by authority language, conservative traits
    prosecution_bias = (
        0.3 * df_encoded.get('authority_score', np.random.uniform(0, 0.1, len(df))) +
        0.3 * df_encoded.get('certainty_score', np.random.uniform(0, 0.1, len(df))) +
        0.2 * df_encoded.get('Political_Leaning', np.random.uniform(0.4, 0.8, len(df))) +
        0.2 * np.random.normal(0.5, 0.15, len(df))
    )
    prosecution_bias = np.clip(prosecution_bias, 0.1, 0.9)
    
    # Defense bias influenced by liberal language, uncertainty, education
    defense_bias = (
        0.3 * df_encoded.get('liberal_score', np.random.uniform(0, 0.1, len(df))) +
        0.3 * df_encoded.get('uncertainty_score', np.random.uniform(0, 0.1, len(df))) +
        0.2 * (1 - df_encoded.get('Political_Leaning', np.random.uniform(0.2, 0.6, len(df)))) +
        0.2 * np.random.normal(0.4, 0.15, len(df))
    )
    defense_bias = np.clip(defense_bias, 0.1, 0.9)
    
    # Neutral bias based on balanced language and high fairness
    neutral_bias = 1 - (prosecution_bias + defense_bias) / 2
    neutral_bias = np.clip(neutral_bias, 0.1, 0.9)
    
    # Build and train deep learning model
    print("\n🤖 BUILDING DEEP LEARNING MODELS...")
    
    bias_predictor = DeepBiasPredictor(input_dim=X_numerical_scaled.shape[1] + X_categorical.shape[1])
    
    # Split data
    indices = np.arange(len(df_encoded))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_num_train, X_num_test = X_numerical_scaled[train_idx], X_numerical_scaled[test_idx]
    X_cat_train, X_cat_test = X_categorical[train_idx], X_categorical[test_idx]
    
    y_train = {
        'prosecution': prosecution_bias[train_idx],
        'defense': defense_bias[train_idx], 
        'neutral': neutral_bias[train_idx]
    }
    y_test = {
        'prosecution': prosecution_bias[test_idx],
        'defense': defense_bias[test_idx],
        'neutral': neutral_bias[test_idx]
    }
    
    # Build model
    deep_model = bias_predictor.build_tabular_model(
        numerical_features=X_num_train.shape[1],
        categorical_features=X_cat_train.shape[1]
    )
    
    print(f"✅ Deep learning model architecture created")
    print(f"   Layers: {len(deep_model.layers)}")
    print(f"   Parameters: {deep_model.count_params():,}")
    
    # Train model
    print("\n🎯 TRAINING DEEP LEARNING MODEL...")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = deep_model.fit(
        [X_num_train, X_cat_train], 
        y_train,
        validation_data=([X_num_test, X_cat_test], y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    train_loss = deep_model.evaluate([X_num_train, X_cat_train], y_train, verbose=0)
    test_loss = deep_model.evaluate([X_num_test, X_cat_test], y_test, verbose=0)
    
    print(f"✅ Model training completed!")
    print(f"   Training loss: {train_loss[0]:.4f}")
    print(f"   Validation loss: {test_loss[0]:.4f}")
    
    # Generate predictions for all jurors
    print("\n📊 GENERATING ENHANCED PREDICTIONS...")
    
    predictions = deep_model.predict([X_numerical_scaled, X_categorical])
    
    prosecution_scores = predictions[0].flatten()
    defense_scores = predictions[1].flatten() 
    neutral_scores = predictions[2].flatten()
    
    # Scale to 10-90 range
    scaler_pro = MinMaxScaler(feature_range=(10, 90))
    scaler_def = MinMaxScaler(feature_range=(10, 90))
    scaler_neu = MinMaxScaler(feature_range=(10, 90))
    
    prosecution_scaled = scaler_pro.fit_transform(prosecution_scores.reshape(-1, 1)).flatten()
    defense_scaled = scaler_def.fit_transform(defense_scores.reshape(-1, 1)).flatten()
    neutral_scaled = scaler_neu.fit_transform(neutral_scores.reshape(-1, 1)).flatten()
    
    # Add to original dataframe
    df_original['Predicted_Prosecution_Bias'] = prosecution_scaled
    df_original['Predicted_Defense_Bias'] = defense_scaled
    df_original['Predicted_Neutral_Bias'] = neutral_scaled
    df_original['Overall_Fairness'] = 100 - abs(prosecution_scaled - defense_scaled)
    
    # Add psychological insights
    if 'sentiment_compound' in psychological_features.columns:
        df_original['Emotional_Stability'] = psychological_features['sentiment_compound']
    if 'authority_score' in psychological_features.columns:
        df_original['Authority_Orientation'] = psychological_features['authority_score']
    if 'certainty_score' in psychological_features.columns:
        df_original['Certainty_Level'] = psychological_features['certainty_score']
    
    # Deep SHAP Analysis
    print("\n🔍 DEEP SHAP EXPLAINABILITY ANALYSIS...")
    
    try:
        import shap
        
        # Create background dataset for SHAP
        background_indices = np.random.choice(train_idx, size=50, replace=False)
        background_data = [X_numerical_scaled[background_indices], X_categorical[background_indices]]
        
        # Create explainer
        explainer = shap.DeepExplainer(deep_model, background_data)
        
        # Explain predictions for sample
        sample_indices = np.random.choice(len(df_original), size=10, replace=False)
        sample_data = [X_numerical_scaled[sample_indices], X_categorical[sample_indices]]
        
        shap_values = explainer.shap_values(sample_data)
        
        print("✅ Deep SHAP analysis completed!")
        print("   Prosecution SHAP shape:", np.array(shap_values[0]).shape)
        print("   Defense SHAP shape:", np.array(shap_values[1]).shape)
        
        # Feature importance analysis
        feature_names = numerical_features + categorical_features
        
        # Average SHAP importance for prosecution model
        pros_shap = np.concatenate([shap_values[0][0], shap_values[0][1]], axis=1)
        pros_importance = np.abs(pros_shap).mean(0)
        top_pros_features = np.argsort(pros_importance)[-10:]
        
        print("\n🔍 TOP DEEP LEARNING FEATURES - PROSECUTION BIAS:")
        for i, idx in enumerate(reversed(top_pros_features)):
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"   {i+1:2d}. {feature_name:<25} | SHAP Impact: {pros_importance[idx]:.4f}")
        
        # Defense model
        def_shap = np.concatenate([shap_values[1][0], shap_values[1][1]], axis=1)
        def_importance = np.abs(def_shap).mean(0)
        top_def_features = np.argsort(def_importance)[-10:]
        
        print("\n🔍 TOP DEEP LEARNING FEATURES - DEFENSE BIAS:")
        for i, idx in enumerate(reversed(top_def_features)):
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"   {i+1:2d}. {feature_name:<25} | SHAP Impact: {def_importance[idx]:.4f}")
            
    except Exception as e:
        print(f"⚠️ Deep SHAP analysis failed: {e}")
    
    # Select random jury pool and analyze
    print(f"\n🎲 SELECTING ENHANCED JURY POOL...")
    print("="*70)
    
    random.seed(42)
    jury_pool_indices = random.sample(range(len(df_original)), min(20, len(df_original)))
    jury_pool = df_original.iloc[jury_pool_indices].copy().reset_index(drop=True)
    
    print(f"📋 ENHANCED JURY POOL ANALYSIS:")
    print("="*70)
    
    for idx, (_, juror) in enumerate(jury_pool.iterrows(), 1):
        pros_bias = juror['Predicted_Prosecution_Bias']
        def_bias = juror['Predicted_Defense_Bias']
        neutral_bias = juror['Predicted_Neutral_Bias']
        fairness = juror['Overall_Fairness']
        
        # Enhanced psychological insights
        emotional_stability = juror.get('Emotional_Stability', 0)
        authority_orientation = juror.get('Authority_Orientation', 0)
        certainty_level = juror.get('Certainty_Level', 0)
        
        # AI-powered bias prediction
        max_bias = max(pros_bias, def_bias, neutral_bias)
        
        if max_bias == pros_bias and pros_bias > def_bias + 5:
            tendency = "PRO-PROSECUTION"
            icon = "🔴"
            recommendation = "LIKELY TO FAVOR PROSECUTION"
        elif max_bias == def_bias and def_bias > pros_bias + 5:
            tendency = "PRO-DEFENSE" 
            icon = "🔵"
            recommendation = "LIKELY TO FAVOR DEFENSE"
        else:
            tendency = "BALANCED/NEUTRAL"
            icon = "🟢"
            recommendation = "LIKELY TO BE IMPARTIAL"
        
        print(f"\n{icon} JUROR #{idx:2d}: {juror['Name']}")
        print(f"   Demographics: {juror.get('Age', 'N/A')} years, {juror.get('Gender', 'N/A')}, {juror.get('Education', 'N/A')}")
        print(f"   Profession: {juror.get('Profession', 'N/A')}")
        print(f"   🧠 AI Bias Scores: Prosecution {pros_bias:.1f} | Defense {def_bias:.1f} | Neutral {neutral_bias:.1f}")
        print(f"   ⚖️ Fairness Score: {fairness:.1f}/100")
        print(f"   🎭 Psychological Profile:")
        print(f"      Emotional Stability: {emotional_stability:.3f}")
        print(f"      Authority Orientation: {authority_orientation:.3f}")  
        print(f"      Certainty Level: {certainty_level:.3f}")
        print(f"   🎯 AI PREDICTION: {recommendation}")
        print(f"   📊 CLASSIFICATION: {tendency}")
        print("-" * 70)
    
    print(f"\n✅ ADVANCED JURY SELECTION ANALYSIS COMPLETE!")
    print(f"🧠 Deep Learning + Advanced NLP + SHAP Explainability")
    print(f"📊 {len(jury_pool)} jurors analyzed with psychological profiling")
    print("="*70)

if __name__ == "__main__":
    main()