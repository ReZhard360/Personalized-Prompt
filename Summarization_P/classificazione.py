import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score)
from imblearn.over_sampling import SMOTE

# Configura il logging su console e file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("classification_results.log", mode="w")])
logging.info("Inizio elaborazione dei dati di classificazione.")

# =======================
# 1. Caricamento del CSV
# =======================
csv_filename = "session_results.csv"
if not os.path.exists(csv_filename):
    logging.error(f"Il file {csv_filename} non è stato trovato.")
    raise FileNotFoundError(f"Il file {csv_filename} non è stato trovato.")

try:
    df = pd.read_csv(csv_filename, on_bad_lines='skip')
    logging.info("CSV caricato correttamente.")
except Exception as e:
    logging.error("Errore nel caricamento del CSV: " + str(e))
    raise e

logging.info("Colonne presenti nel CSV: " + str(df.columns.tolist()))


# =======================
# 2. Estrazione del target dalla colonna eval_metrics (BERTScore_F1)
# =======================
def extract_target(metrics_str):
    try:
        metrics = json.loads(metrics_str)
        return metrics.get("BERTScore_F1", np.nan)
    except Exception as e:
        logging.error("Errore nel parsing del JSON: " + str(e))
        return np.nan

if "eval_metrics" in df.columns:
    df["BERTScore_F1"] = df["eval_metrics"].apply(extract_target)
else:
    logging.error("La colonna 'eval_metrics' non è presente nel CSV.")
    raise KeyError("La colonna 'eval_metrics' non è presente nel CSV.")

df = df.dropna(subset=["BERTScore_F1"])
logging.info(f"Righe dopo eliminazione NaN: {len(df)}")

# =======================
# 3. Discretizzazione del target in classi
# =======================
df['Score_Class'] = pd.qcut(df["BERTScore_F1"], q=3, labels=["Basso", "Medio", "Alto"])
logging.info("Distribuzione delle classi target: " + str(df['Score_Class'].value_counts().to_dict()))

# =======================
# 4. Data Augmentation: controlla il bilanciamento
# =======================
class_counts = df['Score_Class'].value_counts(normalize=True)
use_smote = (class_counts < 0.3).any()
if use_smote:
    logging.info("Classi sbilanciate rilevate, applico SMOTE per data augmentation.")
else:
    logging.info("Bilanciamento delle classi accettabile; non applico SMOTE.")

# =======================
# 5. Selezione delle feature
# =======================
desired_numeric = ["temperature", "chunk_size", "overlap_size", "token_count"]
desired_categorical = ["model", "age_group", "interest", "narrator_type", "language", "retrieval_method"]

numeric_features = [col for col in desired_numeric if col in df.columns]
categorical_features = [col for col in desired_categorical if col in df.columns]
features = numeric_features + categorical_features
missing_features = [col for col in desired_numeric + desired_categorical if col not in df.columns]
if missing_features:
    logging.info("Le seguenti colonne desiderate non sono presenti e saranno ignorate: " + str(missing_features))

X = df[features]
y = df["Score_Class"]
logging.info("Prime 5 righe delle feature:")
logging.info(X.head().to_string())

# =======================
# 6. Pre-processamento: scaling numerico e codifica one-hot
# =======================
preprocessor = ColumnTransformer(transformers=[
    ("num", MinMaxScaler(), numeric_features),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
])
X_processed = preprocessor.fit_transform(X)
logging.info("Forma dell'array delle feature processate: " + str(X_processed.shape))

# =======================
# 7. Divisione in train/test (LOOCV se dataset piccolo)
# =======================
if len(df) < 10:
    cv_strategy = LeaveOneOut()
    X_train, X_test, y_train, y_test = X_processed, X_processed, y, y
    logging.info("Dataset piccolo: utilizzo LOOCV per la validazione.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    cv_strategy = 5
    logging.info("Train/test split effettuato con test_size=0.2.")

# =======================
# 8. Applica SMOTE se necessario
# =======================
if use_smote:
    try:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logging.info("SMOTE applicato. Nuova distribuzione delle classi nel training set:")
        logging.info(pd.Series(y_train).value_counts().to_dict())
    except Exception as e:
        logging.error("Errore nell'applicazione di SMOTE: " + str(e))

# =======================
# 9. Addestramento del classificatore RandomForest
# =======================
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
logging.info("Report di classificazione:\n" + json.dumps(report, indent=2))
logging.info("Accuracy: " + str(accuracy_score(y_test, y_pred)))

cv_scores = cross_val_score(clf, X_processed, y, cv=cv_strategy, scoring='accuracy')
logging.info("CV Accuracy scores: " + str(cv_scores))
logging.info("Media CV Accuracy: " + str(cv_scores.mean()))

# =======================
# 10. Visualizzazione: Matrice di Confusione (percentuali)
# =======================
cm = confusion_matrix(y_test, y_pred, labels=["Basso", "Medio", "Alto"])
cm_percent = np.round(100 * cm / cm.sum(axis=1, keepdims=True), 2)
logging.info("Matrice di Confusione (% per classe): " + str(cm_percent))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Basso", "Medio", "Alto"])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues)
ax.set_title("Matrice di Confusione - Classificazione del BERTScore_F1")
plt.tight_layout()
plt.show()

# =======================
# 11. Visualizzazione: Boxplot dei punteggi per modello
# =======================
plt.figure(figsize=(10, 6))
sns.boxplot(x="model", y="BERTScore_F1", data=df)
plt.title("Distribuzione di BERTScore_F1 per modello")
plt.xlabel("Modello")
plt.ylabel("BERTScore_F1")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =======================
# 12. Visualizzazione: Distribuzione delle classi per modello
# =======================
plt.figure(figsize=(10, 6))
sns.countplot(x="model", hue="Score_Class", data=df, order=sorted(df["model"].unique()))
plt.title("Distribuzione delle classi di performance per modello")
plt.xlabel("Modello")
plt.ylabel("Numero di occorrenze")
plt.xticks(rotation=45)
plt.legend(title="Classe di performance")
plt.tight_layout()
plt.show()

# =======================
# 13. Visualizzazione: Importanza delle Feature
# =======================
feature_names = numeric_features + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=[feature_names[i] for i in indices], y=importances[indices])
plt.title("Importanza delle Feature nel modello di classificazione")
plt.xlabel("Feature")
plt.ylabel("Importanza")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

logging.info("Elaborazione completata. I risultati sono stati salvati nel log e mostrati nei plot.")
