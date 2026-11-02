import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def main():
    print("=== Iniciando Pipeline de Treinamento - SIS-DCM ===")
    
    # Caminho do dataset local, pode ser substituído por um input futuramente, onde o usuário possa fazer upload do seu próprio arquivo
    dataset_path = 'data.csv'
    if not os.path.exists(dataset_path):
        print("Baixando dataset do Kaggle...")
        import kagglehub
        path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
        dataset_path = os.path.join(path, 'data.csv')

    # 1. Carregar os dados
    df = pd.read_csv(dataset_path)
    
    # 2. Pré-processamento e Limpeza
    columns_to_drop = ['id', 'Unnamed: 32']
    df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df_cleaned['diagnosis'] = df_cleaned['diagnosis'].map({'M': 1, 'B': 0})
    
    # 3. Divisão de dados
    X = df_cleaned.drop(columns=['diagnosis'])
    y = df_cleaned['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Padronização das features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Treinamento da Regressão Logística
    print("Treinando Regressão Logística...")
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    
    # 6. Treinamento do Random Forest
    print("Treinando Random Forest Classifier...")
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_clf.fit(X_train_scaled, y_train)
    y_pred_rf = rf_clf.predict(X_test_scaled)
    
    # 7. Avaliação dos resultados
    def exibir_metricas(nome, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"\n=== {nome} ===")
        print(f"Acurácia: {acc:.4f}")
        print(f"Sensibilidade (Recall): {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Matriz de Confusão:")
        print(confusion_matrix(y_true, y_pred))

    exibir_metricas("Regressão Logística", y_test, y_pred_lr)
    exibir_metricas("Random Forest", y_test, y_pred_rf)
    print("\n=== Pipeline finalizado com sucesso! ===")

if __name__ == '__main__':
    main()
