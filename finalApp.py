from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

app = Flask(__name__)

class SpamDetector:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.results = {}
        self.feature_importance = {}
    def load_data(self):
        try:
            df_proc = pd.read_csv(r'C:\Users\User\Downloads\processed_data.csv')
            df_enron = pd.read_csv(r'C:\Users\User\Downloads\enron_spam_data.csv')
            
            # Process processed_data.csv
            text_column_proc = None
            for col in df_proc.columns:
                if col.lower() in ['text', 'message', 'email', 'content']:
                    text_column_proc = col
                    break
            if text_column_proc is None:
                raise ValueError("No suitable text column found in processed_data.csv")
            
            # Handle labels - check if they're already numeric or strings
            if df_proc['label'].dtype == 'object':
                df_proc['label'] = df_proc['label'].map({'ham': 0, 'spam': 1})
            else:
                # Labels are already 0/1, ensure they're correct
                df_proc['label'] = df_proc['label'].astype(int)
            
            df_proc = df_proc[[text_column_proc, 'label']].rename(columns={text_column_proc: 'message'})
            
            # Process enron_data.csv
            df_enron['label'] = df_enron['Spam/Ham'].map({'ham': 0, 'spam': 1})
            df_enron['message'] = df_enron['Subject'].fillna('') + ' ' + df_enron['Message'].fillna('')
            df_enron = df_enron[['message', 'label']]
            
            # Combine and clean data
            self.df = pd.concat([df_proc, df_enron], ignore_index=True)
            self.df = self.df.dropna(subset=['label'])
            self.df['label'] = self.df['label'].astype(int)
            self.df['message'] = self.df['message'].fillna('')
            
            # Clean messages - remove very short and very long messages
            self.df = self.df[self.df['message'].str.len() > 10]  # Remove very short messages
            self.df = self.df[self.df['message'].str.len() < 5000]  # Reduce max length to save memory
            
            # Remove HTML headers and technical content
            self.df['message'] = self.df['message'].str.replace(r'Content-Type.*?\n', '', regex=True)
            self.df['message'] = self.df['message'].str.replace(r'Content-Transfer-Encoding.*?\n', '', regex=True)
            self.df['message'] = self.df['message'].str.replace(r'charset.*?\n', '', regex=True)
            
            # Reduce dataset size to prevent memory issues
            # Take a sample of 20,000 records (10,000 spam + 10,000 ham)
            spam_data = self.df[self.df['label'] == 1].sample(n=min(10000, len(self.df[self.df['label'] == 1])), random_state=42)
            ham_data = self.df[self.df['label'] == 0].sample(n=min(10000, len(self.df[self.df['label'] == 0])), random_state=42)
            self.df = pd.concat([spam_data, ham_data], ignore_index=True)
            
            print(f"Final dataset shape: {self.df.shape}")
            print(f"Spam percentage: {(self.df['label'] == 1).mean() * 100:.1f}%")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    def vectorize_text(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_df=0.95, 
            min_df=3, 
            max_features=2000,  # Reduced from 3000
            ngram_range=(1, 1),  # Reduced from (1, 2) to save memory
            dtype=np.float32  # Use float32 instead of float64 to save memory
        )
        self.X = self.vectorizer.fit_transform(self.df['message'])
        self.y = self.df['label']
    def train_models(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,  # Reduced from 200
                max_depth=10,      # Reduced from 15
                min_samples_split=5,
                class_weight='balanced', 
                random_state=42
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,  # Reduced from 200
                max_depth=4,       # Reduced from 6
                learning_rate=0.1,
                eval_metric='logloss',
                scale_pos_weight=1.0,
                random_state=42
            )
        }
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            self.results[name] = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'y_pred': y_pred,
                'y_proba': y_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
    def create_visualizations(self):
        self.plots = {}
        self._create_performance_comparison()
        self._create_confusion_matrices()
        self._create_roc_curves()
        self._create_feature_importance()
        self._create_data_distribution()
    def _create_performance_comparison(self):
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2
            values = [self.results[model][metric] for model in self.models.keys()]
            bars = ax[row, col].bar(self.models.keys(), values, 
                                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax[row, col].set_title(title, fontweight='bold')
            ax[row, col].set_ylabel(metric.title())
            ax[row, col].set_ylim(0, 1)
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        self.plots['performance_comparison'] = self._fig_to_base64(fig)
        plt.close()
    def _create_confusion_matrices(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        for i, (name, model) in enumerate(self.models.items()):
            cm = self.results[name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                       ax=axes[i])
            axes[i].set_title(f'{name}', fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        plt.tight_layout()
        self.plots['confusion_matrices'] = self._fig_to_base64(fig)
        plt.close()
    def _create_roc_curves(self):
        plt.figure(figsize=(10, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, (name, model) in enumerate(self.models.items()):
            y_proba = self.results[name]['y_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        self.plots['roc_curves'] = self._fig_to_base64(plt.gcf())
        plt.close()
    def _create_feature_importance(self):
        if not self.feature_importance:
            return
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        tree_models = {k: v for k, v in self.feature_importance.items() 
                      if k in ['Random Forest', 'XGBoost']}
        for i, (name, importance) in enumerate(tree_models.items()):
            feature_names = self.vectorizer.get_feature_names_out()
            top_indices = np.argsort(importance)[-20:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]
            axes[i].barh(range(len(top_features)), top_importance, 
                        color=['#FF6B6B', '#4ECDC4'][i])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features)
            axes[i].set_title(f'{name} - Top 20 Features', fontweight='bold')
            axes[i].set_xlabel('Importance')
        plt.tight_layout()
        self.plots['feature_importance'] = self._fig_to_base64(fig)
        plt.close()
    def _create_data_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Dataset Analysis', fontsize=16, fontweight='bold')
        label_counts = self.df['label'].value_counts()
        colors = ['#4ECDC4', '#FF6B6B']
        axes[0].pie(label_counts.values, labels=['Ham', 'Spam'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('Email Distribution', fontweight='bold')
        message_lengths = self.df['message'].str.len()
        axes[1].hist(message_lengths, bins=50, color='#45B7D1', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Message Length (characters)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Message Length Distribution', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        self.plots['data_distribution'] = self._fig_to_base64(fig)
        plt.close()
    def _fig_to_base64(self, fig):
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=300)
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()
    def predict(self, text):
        if not self.vectorizer or not self.models:
            return None
        X_input = self.vectorizer.transform([text])
        predictions = {}
        for name, model in self.models.items():
            proba = model.predict_proba(X_input)[0]
            predictions[name] = {
                'ham_probability': float(proba[0]),
                'spam_probability': float(proba[1]),
                'prediction': 'Spam' if proba[1] > 0.5 else 'Ham'
            }
        ensemble_proba = float(np.mean([predictions[name]['spam_probability'] 
                                for name in self.models.keys()]))
        predictions['Ensemble'] = {
            'ham_probability': float(1 - ensemble_proba),
            'spam_probability': ensemble_proba,
            'prediction': 'Spam' if ensemble_proba > 0.5 else 'Ham'
        }
        return predictions
spam_detector = SpamDetector()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/train')
def train_models():
    try:
        if not spam_detector.load_data():
            return jsonify({'error': 'Failed to load data'})
        spam_detector.vectorize_text()
        spam_detector.train_models()
        spam_detector.create_visualizations()
        
        # Convert NumPy types to Python native types for JSON serialization
        results_for_json = {}
        for model_name, result in spam_detector.results.items():
            results_for_json[model_name] = {
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score'])
            }
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully!',
            'results': results_for_json,
            'plots': spam_detector.plots
        })
    except Exception as e:
        return jsonify({'error': str(e)})
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'})
        predictions = spam_detector.predict(text)
        if predictions is None:
            return jsonify({'error': 'Models not trained yet. Please train models first.'})
        return jsonify({
            'success': True,
            'predictions': predictions,
            'text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)})
@app.route('/dashboard')
def dashboard():
    if not spam_detector.results:
        return render_template('dashboard.html', trained=False)
    return render_template('dashboard.html', 
                         trained=True,
                         results=spam_detector.results,
                         plots=spam_detector.plots)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 