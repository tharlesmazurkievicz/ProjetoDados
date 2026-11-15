from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuração da Página
st.set_page_config(
    page_title="Predição de Churn em Telecomunicações",
    page_icon="📊",
    layout="wide"
)
# Caminhos do Projeto
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

# Busca automática pelo CSV dentro do projeto
arquivos_csv = list(BASE_DIR.rglob("*Telco*Churn*.csv")) + list(BASE_DIR.rglob("*.csv"))
arquivos_csv = [f for f in arquivos_csv if "assets" not in str(f) and "venv" not in str(f)]

if not arquivos_csv:
    st.error(f"Arquivo CSV da base não encontrado dentro de {BASE_DIR}.")
    st.stop()

DATA_PATH = arquivos_csv[0]

@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH)
    df_selecao = pd.read_csv(ASSETS_DIR / "tabela_selecao_variaveis.csv")
    df_resultados = pd.read_csv(ASSETS_DIR / "resultados_modelos.csv")
    artefatos = joblib.load(ASSETS_DIR / "artefatos_avaliacao.pkl")
    return df, df_selecao, df_resultados, artefatos

df, df_selecao, df_resultados, artefatos = carregar_dados()
# Cabeçalho Principal
st.title("📊 Predição de Evasão de Clientes (Churn) com Machine Learning")
st.markdown("**Projeto Extensionista — Tópicos Avançados em Big Data com Python** | Prof. Melquezedeque Lima")
st.divider()

# Organização nas 4 Partes Exigidas pela Diretriz
aba1, aba2, aba3, aba4 = st.tabs([
    "1. Problema e Contexto",
    "2. Metodologia",
    "3. Análise Exploratória (EDA)",
    "4. Resultados dos Modelos"
])

# ==============================================================================
# PARTE 1: PROBLEMA, CONTEXTO, OBJETIVO, BASE DE DADOS E VARIÁVEL-ALVO
# ==============================================================================
with aba1:
    st.header("1. Contextualização e Definição do Problema")
    
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.subheader("Problema Investigado")
        st.write(
            "No setor de telecomunicações, a perda de clientes para a concorrência — fenômeno conhecido "
            "como **Churn** — gera impacto direto na receita recorrente e eleva os custos operacionais, "
            "uma vez que adquirir um novo assinante custa significativamente mais do que reter um cliente ativo."
        )
        
        st.subheader("Objetivo do Projeto")
        st.write(
            "Desenvolver e comparar modelos de **Aprendizado de Máquina Supervisionado (Classificação Binária)** "
            "capazes de identificar clientes com alta probabilidade de cancelamento, permitindo ações preventivas "
            "de retenção baseadas em dados."
        )
        
        st.subheader("Base de Dados e Variável-Alvo")
        st.markdown("""
        - **Origem:** Base pública *Telco Customer Churn* (IBM Sample Data / Kaggle).
        - **Dimensão Bruta:** **7.043 registros** e **21 atributos** (dados demográficos, serviços contratados, tipo de contrato e faturamento).
        - **Variável-Alvo (`Churn`):** Indica se o cliente cancelou o serviço no último mês (`Yes` = 1.869 clientes / 26,5% | `No` = 5.174 clientes / 73,5%).
        """)

    with col2:
        st.subheader("Distribuição da Variável-Alvo (Churn)")
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Quantidade"]
        churn_counts["Status"] = churn_counts["Churn"].map({"No": "Ativo (No)", "Yes": "Cancelou (Yes)"})
        
        fig_target = px.pie(
            churn_counts, values="Quantidade", names="Status",
            color="Churn", color_discrete_map={"No": "#2b5c8f", "Yes": "#d95f02"},
            hole=0.45
        )
        fig_target.update_traces(textinfo="percent+value", textfont_size=14)
        st.plotly_chart(fig_target, use_container_width=True)

    with st.expander("Visualizar Amostra da Base de Dados Bruta (Primeiros 10 registros)"):
        st.dataframe(df.head(10), use_container_width=True)

# ==============================================================================
# PARTE 2: METODOLOGIA (FLUXOGRAMA VISUAL)
# ==============================================================================
with aba2:
    st.header("2. Fluxograma Metodológico")
    st.write(
        "Abaixo está representada a sequência completa de etapas executadas no projeto, desde a ingestão "
        "da base bruta até a avaliação comparativa dos classificadores otimizados."
    )
    
    fluxograma_path = ASSETS_DIR / "fluxograma_metodologia.png"
    if fluxograma_path.exists():
        st.image(str(fluxograma_path), caption="Pipeline Metodológico do Projeto (Draw.io)", use_container_width=True)
    else:
        st.info(
            "📌 **Aviso:** Salve a imagem do fluxograma criado no Draw.io/Lucidchart como "
            "`assets/fluxograma_metodologia.png` para exibi-la automaticamente aqui."
        )
        
    st.subheader("Resumo Técnico das Etapas Aplicadas")
    st.markdown("""
    1. **Diagnóstico e Limpeza:** Remoção de `customerID` (identificador único) e conversão de `TotalCharges` para numérico (revelando 11 valores ausentes em clientes com `tenure = 0`).
    2. **Seleção de Variáveis ($\alpha = 0,05$):** Aplicação de **ANOVA F-test** (numéricas) e **Qui-Quadrado $\chi^2$** (categóricas). Descarte fundamentado de `gender` ($p = 0,614$) e `PhoneService` ($p = 0,760$), mantendo **17 atributos preditores**.
    3. **Pré-Processamento sem Vazamento (*Data Leakage*):** Divisão estratificada (**80% treino / 20% teste**), imputação pela mediana, padronização (`StandardScaler`), codificação `OneHotEncoder` e balanceamento de classes com **SMOTE** aplicado exclusivamente dentro dos *folds* de treino.
    4. **Modelagem e Otimização:** Treinamento de 5 algoritmos (**Random Forest, Regressão Logística, SVM, KNN e Árvore de Decisão**) com **Validação Cruzada Estratificada (5-fold)** e **Grid Search** otimizando o `F1-Score`.
    """)

# ==============================================================================
# PARTE 3: ANÁLISE GRÁFICA DOS ATRIBUTOS VS. VARIÁVEL-ALVO
# ==============================================================================
with aba3:
    st.header("3. Análise Exploratória e Seleção de Variáveis")
    
    st.subheader("3.1 Resultado dos Testes Estatísticos (ANOVA F-test e Qui-Quadrado)")
    st.dataframe(
        df_selecao.style.format({"Estatística": "{:.2f}", "p_value": "{:.4e}"}),
        use_container_width=True
    )
    
    st.divider()
    st.subheader("3.2 Relação entre os Atributos Selecionados e o Churn")
    
    tipo_var = st.radio("Selecione a categoria de atributos para visualizar:", ["Variáveis Numéricas (3)", "Variáveis Categóricas (14)"], horizontal=True)
    
    if tipo_var == "Variáveis Numéricas (3)":
        num_vars = df_selecao[(df_selecao["Tipo"] == "Numérica") & (df_selecao["Decisão"] == "Mantida")]["Atributo"].tolist()
        cols = st.columns(3)
        for idx, var in enumerate(num_vars):
            img_path = ASSETS_DIR / f"eda_num_{var}.png"
            if img_path.exists():
                cols[idx].image(str(img_path), caption=f"{var} vs. Churn", use_container_width=True)
    else:
        cat_vars = df_selecao[(df_selecao["Tipo"] == "Categórica") & (df_selecao["Decisão"] == "Mantida")]["Atributo"].tolist()
        var_escolhida = st.selectbox("Escolha o atributo categórico:", cat_vars)
        img_path = ASSETS_DIR / f"eda_cat_{var_escolhida}.png"
        if img_path.exists():
            col_img, col_desc = st.columns([1.4, 1])
            with col_img:
                st.image(str(img_path), use_container_width=True)
            with col_desc:
                row_info = df_selecao[df_selecao["Atributo"] == var_escolhida].iloc[0]
                st.metric("Estatística Qui-Quadrado (χ²)", f"{row_info['Estatística']:.2f}")
                st.metric("p-value", f"{row_info['p_value']:.2e}")
                st.write(f"O atributo **{var_escolhida}** apresenta associação estatisticamente significativa com o cancelamento ($p < 0,05$).")

# ==============================================================================
# PARTE 4: RESULTADOS DOS MODELOS DE MACHINE LEARNING
# ==============================================================================
with aba4:
    st.header("4. Desempenho e Comparação dos Modelos Otimizados")
    
    st.subheader("4.1 Tabela Comparativa de Métricas (Conjunto de Teste - 20%)")
    st.dataframe(
        df_resultados.style.highlight_max(
            subset=["AUROC", "F1-Score", "Sensibilidade (Recall)", "Especificidade", "Precisão", "Acurácia"],
            color="#d4edda"
        ),
        use_container_width=True
    )
    
    st.divider()
    col_roc, col_cm = st.columns([1.2, 1])
    
    with col_roc:
        st.subheader("4.2 Curvas ROC Comparativas (AUROC)")
        fig_roc = go.Figure()
        for _, row in df_resultados.iterrows():
            nome = row["Modelo"]
            auroc = row["AUROC"]
            fpr, tpr, _ = artefatos["curvas_roc"][nome]
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nome} (AUROC = {auroc:.4f})"))
            
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório (0.50)", line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(
            xaxis_title="Taxa de Falsos Positivos (1 - Especificidade)",
            yaxis_title="Taxa de Verdadeiros Positivos (Sensibilidade)",
            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
            height=450
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col_cm:
        st.subheader("4.3 Matriz de Confusão por Modelo")
        modelo_sel = st.selectbox("Selecione o modelo:", df_resultados["Modelo"].tolist())
        cm = artefatos["matrizes"][modelo_sel]
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Classe Predita", y="Classe Real", color="Clientes"),
            x=["Não (No)", "Sim (Yes)"],
            y=["Não (No)", "Sim (Yes)"],
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)