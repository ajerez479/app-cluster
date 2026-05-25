import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Dashboard de Clustering - Netflix Users",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal con estilo limpio
st.title("📊 Análisis de Clustering & Segmentación de Usuarios")
st.markdown("---")

# Cargar los datos preentrenados usando st.cache_data para máxima velocidad
@st.cache_data
def cargar_artefactos():
    try:
        df_orig = joblib.load('df_original.joblib')
        df_mod = joblib.load('df_modificado.joblib')
        info = joblib.load('info_modelo.joblib')
        df_pca_plot = joblib.load('df_pca_plot.joblib') # Cargo la gráfica
        return df_orig, df_mod, info, df_pca_plot
    except FileNotFoundError:
        st.error("❌ No se encontraron los archivos. Ejecuta primero `python model.py`.")
        return None, None, None, None

df_original, df_modificado, info_modelo, df_pca_plot = cargar_artefactos()

if df_original is not None:
    # --- SECCIÓN 1: MÉTRICAS E INFO DEL MODELO ---
    st.header("⚙️ Información del Modelo y Rendimiento")
    
    # Diseño en columnas para las métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1: st.metric(label="Modelo Utilizado", value=info_modelo['nombre_modelo'])
    with col2: st.metric(label="¿Utilizó PCA?", value=info_modelo['usa_pca'])
    with col3: st.metric(label="Silhouette Score", value=f"{info_modelo['silhouette']:.4f}")
    with col4: st.metric(label="Davies-Bouldin Index", value=f"{info_modelo['davies_bouldin']:.4f}")
    
    st.markdown("---")

    # --- SECCIÓN 2: VISUALIZACIÓN DE DATASETS ---
    st.header("🔍 Visualización de Datos")

    # Pestañas (Tabs) para separar los enfoques de los datos
    tab1, tab2, tab3 = st.tabs([
        "📋 Dataset Original Legible", 
        "🛠️ Dataset Preprocesado (Modelo)", 
        "🎯 Comportamiento por Cluster"
    ])
    
    # Construir el dataframe original cruzado con las etiquetas del cluster
    df_lectura = df_original.copy()
    if 'Cluster' in df_modificado.columns:
        df_lectura['Cluster_Asignado'] = df_modificado['Cluster']

    with tab1:
        st.subheader("Vista previa de los datos originales (fijos)")
        st.markdown(f"Total de registros: `{df_original.shape[0]}` | Columnas originales: `{df_original.shape[1]}`")
        st.dataframe(df_lectura, use_container_width=True)
        
    with tab2:
        st.subheader("Vista previa de los datos tras limpieza, codificación y escalado")
        st.markdown(f"Total de registros: `{df_modificado.shape[0]}` | Columnas procesadas: `{df_modificado.shape[1]}`")
        st.dataframe(df_modificado, use_container_width=True)

    with tab3:
        st.subheader("Análisis del Comportamiento de Usuarios")
        st.markdown("Filtra el dataset por un cluster específico para inspeccionar directamente el comportamiento de ese grupo de usuarios.")
        
        if 'Cluster_Asignado' in df_lectura.columns:
            # Selector de Cluster interactivo
            cluster_seleccionado = st.selectbox("Selecciona un Cluster para inspeccionar:", sorted(df_lectura['Cluster_Asignado'].unique()))
            
            # Filtrar datos legibles por el cluster seleccionado
            df_filtrado = df_lectura[df_lectura['Cluster_Asignado'] == cluster_seleccionado]
            
            st.markdown(f"Registros en este grupo: `{df_filtrado.shape[0]}` ({df_filtrado.shape[0]/df_lectura.shape[0]*100:.1f}% del total)")
            st.dataframe(df_filtrado, use_container_width=True)
            
            # --- PERFILAMIENTO AUTOMÁTICO DE COMPORTAMIENTO ---
            st.markdown("#### 🕵️‍♂️ Perfil Resumen de este Grupo")
            col_inf1, col_inf2 = st.columns(2)
            
            with col_inf1:
                st.markdown("**Valores Promedio (Variables Numéricas):**")
                # Excluir identificadores para promediar el comportamiento real
                num_cols = df_original.select_dtypes(include=['number']).columns.drop('User_ID', errors='ignore')
                if not num_cols.empty:
                    st.dataframe(df_filtrado[num_cols].mean().rename("Promedio Real"), use_container_width=True)
                else:
                    st.info("No hay columnas numéricas para promediar.")
            
            with col_inf2:
                st.markdown("**Tendencias Principales (Variables Categóricas más frecuentes):**")
                cat_cols = df_original.select_dtypes(include=['object']).columns
                if not cat_cols.empty:
                    # Crear un diccionario con la moda de cada categoría
                    modas = {col: [df_filtrado[col].mode()[0]] for col in cat_cols if not df_filtrado[col].empty}
                    st.dataframe(pd.DataFrame(modas, index=["Frecuente"]).T, use_container_width=True)
                else:
                    st.info("No hay columnas categóricas en el dataset original.")

    # --- GRÁFICA PCA---
    st.header("📈 Visualización de Clusters (PCA 2D)")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df_pca_plot['Principal Component 1'], 
        df_pca_plot['Principal Component 2'], 
        c=df_modificado['Cluster'], 
        cmap='viridis'
    )
    ax.set_title(f'K-Means Clusters on PCA Components (k=2)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("---")
    # ------------------------------

    # --- SECCIÓN 3: GRÁFICO DE DISTRIBUCIÓN ---
    st.subheader("📊 Distribución de Registros por Cluster")
    if 'Cluster' in df_modificado.columns:
        cluster_counts = df_modificado['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Cantidad']
        st.bar_chart(data=cluster_counts, x='Cluster', y='Cantidad', use_container_width=True)