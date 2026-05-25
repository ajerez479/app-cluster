import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def train_and_save_pipeline():
    try:
        df = pd.read_csv('netflix_users.csv')
    except FileNotFoundError:
        print("Error: Por favor coloca tu archivo 'netflix_users.csv' en la misma carpeta.")
        return

    # Guardar copia del dataset original antes de modificaciones
    df_original = df.copy()

    # 2. Limpieza de datos
    df_clean = df.drop(columns=['User_ID'], errors='ignore')

    # Estructurar la edad en categorías
    if 'Age' in df_clean.columns:
        df_clean['Age_Category'] = pd.cut(
            df_clean['Age'],
            bins=[0, 12, 25, 60, int(df_clean['Age'].max())],
            labels=['ninio', 'joven', 'adulto', 'anciano']
        )
        df_clean = df_clean.drop(columns=['Age'])

    # Convertir 'Last_Login' a datetime y calcular días de inactividad
    if 'Last_Login' in df_clean.columns:
        df_clean['Last_Login'] = pd.to_datetime(df_clean['Last_Login'])
        df_clean['Inactivity_Days'] = (pd.to_datetime('today') - df_clean['Last_Login']).dt.days
        df_clean = df_clean.drop(columns=['Last_Login'])

    # Identificar las variables numerical y categorical
    numerical = df_clean.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
    categorical = df_clean.select_dtypes(include=['object']).columns.tolist()

    # Ordinal Encoding in Age_Category
    if 'Age_Category' in df_clean.columns:
        age_mapping = {'ninio': 0, 'joven': 1, 'adulto': 2, 'anciano': 3}
        df_clean['Age_Category'] = df_clean['Age_Category'].map(age_mapping)
        # Asegurar que no se tome como categórica pura para el encoder si ya es numérica
        if 'Age_Category' in categorical:
            categorical.remove('Age_Category')

    # Label Encoding
    encoder = LabelEncoder()
    for col in categorical:
        df_clean[col] = encoder.fit_transform(df_clean[col].astype(str))

    # Feature Scaling
    scaler = StandardScaler()
    if numerical:
        df_clean[numerical] = scaler.fit_transform(df_clean[numerical])

    # 3. PCA (Reducción de dimensionalidad para entrenamiento)
    pca = PCA(n_components=1)
    df_pca_res = pca.fit_transform(df_clean[numerical])
    
    df_pca = pd.DataFrame(data=df_pca_res, columns=['Principal Component 1'])

    # Integrar PCA y las categóricas
    df_pca = pd.concat([df_pca, df_clean[categorical].reset_index(drop=True)], axis=1)

    # 4. Entrenar el Modelo
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(df_pca)

    # --- GENERAR DATA PARA LA GRÁFICA (PCA 2D) ---
    pca_plot = PCA(n_components=2)
    df_pca_plot = pca_plot.fit_transform(df_clean[numerical])
    df_pca_plot = pd.DataFrame(
        data=df_pca_plot,
        columns=['Principal Component 1', 'Principal Component 2']
    )
    joblib.dump(df_pca_plot, 'df_pca_plot.joblib')
    # ---------------------------------------------

    # Calcular Métricas
    silhouette_avg = silhouette_score(df_pca, labels)
    davies_bouldin = davies_bouldin_score(df_pca, labels)

    # Preservar dataset con etiquetas para visualización
    df_clean_with_labels = df_clean.copy()
    df_clean_with_labels['Cluster'] = labels

    # 5. Guardar TODO en archivos joblib para Streamlit
    joblib.dump(df_original, 'df_original.joblib')
    joblib.dump(df_clean_with_labels, 'df_modificado.joblib')
    joblib.dump(kmeans, 'modelo_kmeans.joblib')
    
    # Guardamos las métricas e info en un diccionario
    info_modelo = {
        'nombre_modelo': 'KMeans',
        'usa_pca': 'Sí (1 componente)',
        'silhouette': silhouette_avg,
        'davies_bouldin': davies_bouldin
    }
    joblib.dump(info_modelo, 'info_modelo.joblib')
    
    print("¡Modelo entrenado y artefactos guardados con éxito!")

if __name__ == "__main__":
    train_and_save_pipeline()