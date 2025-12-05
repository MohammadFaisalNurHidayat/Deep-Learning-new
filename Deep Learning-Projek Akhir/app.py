# app.py - Heart Disease Prediction (Enhanced for 90%+ accuracy)
# Improvements: Feature engineering, better architecture, hyperparameter tuning, ensemble diversity

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

app = Flask(__name__)
app.secret_key = 'heart-disease-prediction-2025'

# Global variables
ensemble_models = []
scaler = None
model_metrics = None
BEST_THRESHOLD = 0.5
ENSEMBLE_SIZE = 5  # Increased ensemble size

def feature_engineering(df):
    """Enhanced feature engineering for better predictions"""
    df = df.copy()
    
    # Age groups (risk increases with age)
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3])
    df['age_group'] = df['age_group'].astype(float)
    
    # Age-sex interaction (men at younger age have higher risk)
    df['age_sex'] = df['age'] * df['sex']
    
    # Cholesterol categories
    df['chol_category'] = pd.cut(df['chol'], bins=[0, 200, 240, 1000], labels=[0, 1, 2])
    df['chol_category'] = df['chol_category'].astype(float)
    
    # Blood pressure categories
    df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 1000], labels=[0, 1, 2])
    df['bp_category'] = df['bp_category'].astype(float)
    
    # Heart rate reserve (220 - age - thalach)
    df['heart_rate_reserve'] = 220 - df['age'] - df['thalach']
    
    # Normalized heart rate (percentage of max heart rate achieved)
    df['thalach_pct'] = df['thalach'] / (220 - df['age'])
    
    # Multiple risk factors
    df['risk_score'] = (
        (df['age'] > 55).astype(int) * 2 +
        (df['sex'] == 1).astype(int) +
        (df['cp'] >= 2).astype(int) * 2 +
        (df['trestbps'] > 140).astype(int) * 1.5 +
        (df['chol'] > 240).astype(int) * 1.5 +
        (df['fbs'] == 1).astype(int) +
        (df['thalach'] < 120).astype(int) * 1.5 +
        (df['exang'] == 1).astype(int) * 2 +
        (df['oldpeak'] > 2).astype(int) * 1.5 +
        (df['ca'] > 0).astype(int) * 2
    )
    
    # Interaction features
    df['cp_exang'] = df['cp'] * df['exang']
    df['oldpeak_slope'] = df['oldpeak'] * df['slope']
    df['ca_thal'] = df['ca'] * df['thal']
    
    # Polynomial features for important variables
    df['age_squared'] = df['age'] ** 2
    df['oldpeak_squared'] = df['oldpeak'] ** 2
    
    # Blood pressure to cholesterol ratio
    df['bp_chol_ratio'] = df['trestbps'] / (df['chol'] + 1)
    
    return df

def load_data():
    """Load dataset with improved preprocessing"""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(url, names=columns, na_values='?')
        print("✅ Dataset downloaded from UCI")
    except Exception as e:
        print("⚠️ Using synthetic data (for demo)", e)
        np.random.seed(42)
        n = 303
        age = np.random.randint(29, 80, n)
        sex = np.random.randint(0, 2, n)
        cp = np.random.randint(0, 4, n)
        trestbps = np.random.randint(94, 200, n)
        chol = np.random.randint(126, 564, n)
        fbs = (chol > 200).astype(int) * np.random.randint(0, 2, n)
        restecg = np.random.randint(0, 3, n)
        thalach = 220 - age + np.random.randint(-30, 30, n)
        thalach = np.clip(thalach, 71, 202)
        exang = np.random.randint(0, 2, n)
        oldpeak = np.random.uniform(0, 6.2, n)
        slope = np.random.randint(0, 3, n)
        ca = np.random.randint(0, 4, n)
        thal = np.random.randint(0, 4, n)

        risk_score = (
            (age > 55) * 2.5 +
            (sex == 1) * 1.5 +
            (cp >= 2) * 2.5 +
            (trestbps > 140) * 1.8 +
            (chol > 240) * 1.8 +
            (fbs == 1) * 1.2 +
            (thalach < 120) * 2 +
            (exang == 1) * 2.5 +
            (oldpeak > 2) * 2 +
            (ca > 0) * 2.5 +
            (thal == 3) * 1.5
        )
        risk_score += np.random.normal(0, 1.2, n)
        target = (risk_score > 7).astype(int)

        df = pd.DataFrame({
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
            'ca': ca, 'thal': thal, 'target': target
        })

    # Preprocessing
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Apply feature engineering
    df = feature_engineering(df)

    return df

def build_optimized_model(input_dim, model_idx=0):
    """Enhanced model architecture with better regularization"""
    # Vary architecture slightly per model for ensemble diversity
    if model_idx % 3 == 0:
        # Deeper network
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.35),
            
            Dense(96, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.30),
            
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(48, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.20),
            
            Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            Dropout(0.15),
            
            Dense(1, activation='sigmoid')
        ])
    elif model_idx % 3 == 1:
        # Wider network
        model = Sequential([
            Dense(192, activation='relu', input_shape=(input_dim,), kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.40),
            
            Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.35),
            
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            Dropout(0.20),
            
            Dense(1, activation='sigmoid')
        ])
    else:
        # Balanced network
        model = Sequential([
            Dense(160, activation='relu', input_shape=(input_dim,), kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.38),
            
            Dense(96, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.32),
            
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.28),
            
            Dense(48, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            BatchNormalization(),
            Dropout(0.22),
            
            Dense(24, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.001)),
            Dropout(0.15),
            
            Dense(1, activation='sigmoid')
        ])
    
    # Use slightly different learning rates for diversity
    lr = 0.0003 if model_idx % 2 == 0 else 0.00025
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def _train_single_model(X_train, y_train, X_val, y_val, seed=42, model_idx=0):
    """Train one model instance with improved callbacks"""
    tf.random.set_seed(seed + model_idx * 100)
    np.random.seed(seed + model_idx)

    model = build_optimized_model(X_train.shape[1], model_idx=model_idx)
    
    if model_idx == 0:
        print(f"\n🧩 Model Architecture (Model {model_idx+1})")
        model.summary()
    
    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=25,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )

    # Compute class weights
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    cw = dict(enumerate(cw))

    # Adjust batch size for better generalization
    batch_size = 8 if model_idx % 2 == 0 else 12

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=400,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        class_weight=cw,
        verbose=1 if model_idx == 0 else 0
    )

    model_path = f'heart_model_{model_idx}.keras'
    model.save(model_path)
    return model, history

def train_model():
    """Train ensemble model with enhanced preprocessing"""
    global ensemble_models, scaler, model_metrics, BEST_THRESHOLD

    print("\n" + "="*60)
    print("🚀 TRAINING HEART DISEASE PREDICTION MODEL (Enhanced)")
    print("="*60)

    df = load_data()
    print(f"📊 Dataset size: {len(df)} samples")
    print(f"📊 Features after engineering: {df.shape[1] - 1}")
    print(f"📊 Class distribution: {df['target'].value_counts().to_dict()}")

    X = df.drop('target', axis=1).reset_index(drop=True)
    y = df['target'].reset_index(drop=True)

    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Validation samples: {len(X_val)}")
    print(f"📊 Test samples: {len(X_test)}")

    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train ensemble
    ensemble_models = []
    histories = []
    print(f"\n🔧 Training ensemble of {ENSEMBLE_SIZE} models...")
    for i in range(ENSEMBLE_SIZE):
        print(f"\n{'='*50}")
        print(f"Training model {i+1}/{ENSEMBLE_SIZE}")
        print('='*50)
        m, h = _train_single_model(X_train_scaled, y_train.values, X_val_scaled, y_val.values, seed=42, model_idx=i)
        ensemble_models.append(m)
        histories.append(h)

    # Threshold tuning on validation set
    print("\n🔍 Tuning threshold on validation set...")
    val_probs = np.column_stack([m.predict(X_val_scaled, verbose=0).reshape(-1) for m in ensemble_models])
    val_mean_prob = val_probs.mean(axis=1)

    best_thr = 0.5
    best_f1 = 0.0
    # Optimize for F1 score instead of just accuracy
    for thr in np.linspace(0.25, 0.75, 51):
        preds = (val_mean_prob > thr).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    BEST_THRESHOLD = best_thr
    print(f"✅ Best threshold: {BEST_THRESHOLD:.3f} (val F1={best_f1:.4f})")

    # Final evaluation on test set
    test_probs = np.column_stack([m.predict(X_test_scaled, verbose=0).reshape(-1) for m in ensemble_models])
    test_mean_prob = test_probs.mean(axis=1)
    test_preds = (test_mean_prob > BEST_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_mean_prob)
    cm = confusion_matrix(y_test, test_preds)

    # Cross-validation
    cv_scores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_for_cv = np.vstack([X_train_scaled, X_val_scaled])
    y_for_cv = np.concatenate([y_train.values, y_val.values])
    
    print("\n🔄 Performing 5-Fold Cross-Validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_for_cv, y_for_cv)):
        X_fold_train = X_for_cv[train_idx]
        y_fold_train = y_for_cv[train_idx]
        X_fold_val = X_for_cv[val_idx]
        y_fold_val = y_for_cv[val_idx]

        fold_model = build_optimized_model(X_fold_train.shape[1], model_idx=fold_idx)
        fold_model.fit(X_fold_train, y_fold_train, epochs=100, batch_size=8, verbose=0)
        y_fold_pred = (fold_model.predict(X_fold_val, verbose=0) > 0.5).astype(int)
        fold_acc = accuracy_score(y_fold_val, y_fold_pred)
        cv_scores.append(fold_acc)
        print(f"  Fold {fold_idx+1}/5: {fold_acc*100:.2f}%")

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"🎯 Test Accuracy:     {accuracy*100:.2f}%")
    print(f"🎯 Test Precision:    {precision*100:.2f}%")
    print(f"🎯 Test Recall:       {recall*100:.2f}%")
    print(f"🎯 Test AUC:          {auc:.4f}")
    print(f"🎯 CV Accuracy:       {cv_mean*100:.2f}% (±{cv_std*100:.2f}%)")
    print(f"📊 Confusion Matrix:\n{cm}")
    print("="*60 + "\n")

    model_metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'auc': float(auc),
        'cv_accuracy': cv_mean,
        'cv_std': cv_std,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(df),
        'ensemble_size': ENSEMBLE_SIZE,
        'best_threshold': BEST_THRESHOLD
    }

    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(model_metrics, 'metrics.pkl')

    print("💾 Ensemble models, scaler, and metrics saved!")

    return model_metrics

def load_or_train_model():
    """Load existing ensemble models or train new ones"""
    global ensemble_models, scaler, model_metrics, BEST_THRESHOLD

    if os.path.exists('metrics.pkl') and os.path.exists('scaler.pkl'):
        try:
            model_metrics = joblib.load('metrics.pkl')
            scaler = joblib.load('scaler.pkl')
            BEST_THRESHOLD = model_metrics.get('best_threshold', 0.5)
            ensemble_models = []
            i = 0
            while True:
                path = f'heart_model_{i}.keras'
                if os.path.exists(path):
                    ensemble_models.append(load_model(path))
                    i += 1
                else:
                    break
            if len(ensemble_models) == 0:
                print("⚠️ No saved ensemble models found, training new ensemble.")
                train_model()
            else:
                print(f"✅ Loaded {len(ensemble_models)} ensemble models.")
                print(f"   Accuracy: {model_metrics.get('accuracy',0)*100:.2f}%")
                print(f"   AUC: {model_metrics.get('auc',0):.4f}")
        except Exception as e:
            print("⚠️ Error loading saved artifacts:", e)
            train_model()
    else:
        print("🏗️  No saved model/metrics found. Training new ensemble...")
        train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict using ensemble with feature engineering"""
    try:
        data = request.json
        
        # Create dataframe with input data
        input_df = pd.DataFrame([{
            'age': float(data['age']),
            'sex': float(data['sex']),
            'cp': float(data['cp']),
            'trestbps': float(data['trestbps']),
            'chol': float(data['chol']),
            'fbs': float(data['fbs']),
            'restecg': float(data['restecg']),
            'thalach': float(data['thalach']),
            'exang': float(data['exang']),
            'oldpeak': float(data['oldpeak']),
            'slope': float(data['slope']),
            'ca': float(data['ca']),
            'thal': float(data['thal'])
        }])
        
        # Apply feature engineering
        input_df = feature_engineering(input_df)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Ensemble prediction
        probs = np.column_stack([m.predict(input_scaled, verbose=0).reshape(-1) for m in ensemble_models])
        prob = float(probs.mean())
        prediction = 1 if prob > BEST_THRESHOLD else 0

        # Risk level
        if prob >= 0.75:
            risk_level = "Sangat Tinggi"
            risk_color = "red"
        elif prob >= 0.55:
            risk_level = "Tinggi"
            risk_color = "orange"
        elif prob >= 0.35:
            risk_level = "Sedang"
            risk_color = "yellow"
        else:
            risk_level = "Rendah"
            risk_color = "green"

        # Risk factors
        risk_factors = []
        if float(data['age']) > 55:
            risk_factors.append("Usia di atas 55 tahun")
        if float(data['chol']) > 240:
            risk_factors.append("Kolesterol tinggi (>240 mg/dl)")
        if float(data['trestbps']) > 140:
            risk_factors.append("Tekanan darah tinggi (>140 mm Hg)")
        if float(data['fbs']) == 1:
            risk_factors.append("Gula darah puasa tinggi (>120 mg/dl)")
        if float(data['exang']) == 1:
            risk_factors.append("Nyeri dada saat olahraga")
        if float(data['thalach']) < 100:
            risk_factors.append("Detak jantung maksimal rendah")
        if float(data['ca']) > 0:
            risk_factors.append(f"Pembuluh darah tersumbat: {data['ca']}")
        if float(data['oldpeak']) > 2:
            risk_factors.append("ST depression tinggi")

        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'probability': prob,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_factors': risk_factors
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if model_metrics:
        return jsonify({'status': 'success', 'metrics': model_metrics})
    else:
        return jsonify({'status': 'error', 'message': 'Model belum dilatih'}), 404

@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        metrics = train_model()
        return jsonify({'status': 'success', 'message': 'Model berhasil dilatih ulang', 'metrics': metrics})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 HEART DISEASE PREDICTION SYSTEM (Enhanced)")
    print("="*60 + "\n")

    load_or_train_model()

    print("\n" + "="*60)
    print("🌐 Starting Flask server...")
    print("📍 Akses aplikasi di: http://localhost:5000")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)