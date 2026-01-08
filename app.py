import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="FinanceOps - Delta.ai Edition", layout="wide")

# =========================
# HELPERS & FORMATAÇÃO
# =========================
def brl(x: float) -> str:
    s = f"{x:,.2f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")

def pct(x: float) -> str:
    return f"{x:.1f}%"

# =========================
# ENGINE DE PRECIFICAÇÃO (Baseado nos CSVs fornecidos)
# =========================
def engine_precificacao_delta(fixas, pessoas, horas_nominais, margem_alvo, impostos=10.0, comissao=5.0):
    """
    Refaz a lógica conforme 'ROTEIRO PARA FORMAÇÃO DE PREÇOS.csv'
    e 'ÍNDICE COMERCIALIZAÇÃO E MARK U.csv'
    """
    # 2º Passo: Capacidade Produtiva (Produtividade de 85% conforme planilha)
    capacidade_real = (pessoas * horas_nominais) * 0.85
    
    # 3º Passo: Custo Hora
    custo_hora = fixas / capacidade_real if capacidade_real > 0 else 0
    
    # 6º Passo: Mark Up Multiplicador (Evita o erro de margem sobre custo)
    taxas_incidencia = impostos + comissao + margem_alvo
    if taxas_incidencia >= 100:
        markup = 10.0 # Trava de segurança
    else:
        markup = 100 / (100 - taxas_incidencia)
    
    # Preço Sugerido por Hora (Preço de Venda = Custo Direto * Markup)
    preco_venda_hora = custo_hora * markup
    
    return {
        "custo_hora": custo_hora,
        "markup": markup,
        "preco_venda_hora": preco_venda_hora,
        "capacidade_real": capacidade_real
    }

# =========================
# INTERFACE PRINCIPAL
# =========================
st.title("🚀 FinanceOps — MVP Delta.ai")
st.markdown("---")

# Sidebar: Configurações de Custos Reais (Inputs das Planilhas)
st.sidebar.header("📋 Dados da Operação")
custos_fixos_mensais = st.sidebar.number_input("Despesas Fixas Totais (Mês)", value=15000.0, step=500.0)
time_produtivo = st.sidebar.number_input("Nº de Pessoas (Mão de Obra Direta)", value=2, min_value=1)
horas_p_pessoa = st.sidebar.number_input("Horas Contratuais/Mês", value=160, step=10)

st.sidebar.header("💰 Estratégia Comercial")
margem_desejada = st.sidebar.slider("Margem de Lucro Alvo (%)", 10, 80, 40)
taxa_imposto = st.sidebar.number_input("Impostos (%)", value=10.0)

# Cálculo em tempo real
dados_preco = engine_precificacao_delta(
    custos_fixos_mensais, time_produtivo, horas_p_pessoa, margem_desejada, impostos=taxa_imposto
)

# =========================
# DASHBOARD DE RESULTADOS
# =========================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Custo Hora (Real)", brl(dados_preco["custo_hora"]))
    st.caption("Considerando 85% de produtividade")

with c2:
    st.metric("Mark Up Multiplicador", f"{dados_preco['markup']:.2f}x")
    st.caption("Proteção de margem aplicada")

with c3:
    st.metric("Preço de Venda (Sugestão/h)", brl(dados_preco["preco_venda_hora"]))
    st.info("Preço ideal para bater a meta")

with c4:
    # Simulação de Ponto de Equilíbrio (Break-even em horas)
    be_horas = custos_fixos_mensais / (dados_preco["preco_venda_hora"] * 0.8) # 80% margem contrib.
    st.metric("Ponto Equilíbrio (Horas)", f"{int(be_horas)}h")

st.markdown("---")

# =========================
# SIMULAÇÃO DE CENÁRIOS E RUNWAY
# =========================
st.subheader("📉 Projeção de Runway e Fluxo de Caixa")

col_input, col_chart = st.columns([1, 2])

with col_input:
    caixa_inicial = st.number_input("Caixa Atual (R$)", value=50000.0)
    vendas_estimadas_h = st.slider("Horas Vendidas/Mês", 10, int(dados_preco["capacidade_real"]), 80)
    
    # Cálculo de Receita e Burn
    receita_mensal = vendas_estimadas_h * dados_preco["preco_venda_hora"]
    burn_mensal = custos_fixos_mensais - (receita_mensal * 0.7) # simplificado: custos variáveis ~30%
    
    if burn_mensal > 0:
        runway = caixa_inicial / burn_mensal
        st.error(f"⚠️ Runway Estimado: {runway:.1f} meses")
    else:
        st.success("✅ Operação Lucrativa (Cash Flow Positive)")

with col_chart:
    # Criando gráfico de evolução de caixa para 12 meses
    meses = [f"Mês {i}" for i in range(1, 13)]
    caixa_evolucao = []
    caixa_temp = caixa_inicial
    for m in meses:
        caixa_temp -= burn_mensal
        caixa_evolucao.append(max(caixa_temp, 0))
    
    df_projeção = pd.DataFrame({"Mês": meses, "Saldo em Caixa": caixa_evolucao})
    st.area_chart(df_projeção.set_index("Mês"))

# =========================
# RELATÓRIO COPIÁVEL
# =========================
with st.expander("📝 Gerar Relatório de Precificação para Sócios"):
    report = f"""
    ESTRATÉGIA DE PRECIFICAÇÃO FINANCE OPS:
    --------------------------------------
    1. CUSTO ESTRUTURAL: {brl(custos_fixos_mensais)}
    2. CAPACIDADE REAL: {dados_preco['capacidade_real']:.1f} horas/mês
    3. CUSTO HORA TÉCNICO: {brl(dados_preco['custo_hora'])}
    4. MARKUP MULTIPLICADOR: {dados_preco['markup']:.2f}x
    
    RESULTADO:
    - Preço Sugerido: {brl(dados_preco['preco_venda_hora'])} /hora
    - Margem Líquida Prevista: {margem_desejada}%
    - Break-even: Vender {int(be_horas)} horas/mês.
    """
    st.code(report, language="text")
