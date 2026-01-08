import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="FinanceOps - MVP Delta.ai", layout="wide")

# ==========================================
# 1. HELPERS E FORMATAÇÃO
# ==========================================
def brl(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def pct(x: float) -> str:
    return f"{x:.2f}%"

# ==========================================
# 2. CORE FINANCEIRO (LÓGICA DAS PLANILHAS)
# ==========================================

def calcular_markup_multiplicador(impostos, comissao, margem_lucro):
    """Lógica da aba 'ÍNDICE COMERCIALIZAÇÃO E MARK U'"""
    soma_taxas = impostos + comissao + margem_lucro
    if soma_taxas >= 100: return 10.0  # Limite de segurança
    return 100 / (100 - soma_taxas)

def calcular_venda_prazo(valor_avista, parcelas, taxa_juros=0.0123):
    """Lógica da aba 'CÁLCULO PREÇO À PRAZO'"""
    if parcelas <= 1: return valor_avista
    # Fórmula de coeficiente de financiamento (Price)
    coeficiente = (taxa_juros * (1 + taxa_juros)**parcelas) / ((1 + taxa_juros)**parcelas - 1)
    valor_parcela = valor_avista * coeficiente
    return valor_parcela * parcelas

# ==========================================
# 3. INTERFACE E INPUTS (SIDEBAR)
# ==========================================
st.title("🚀 FinanceOps MVP — Sistema de Gestão Delta.ai")
st.markdown("---")

with st.sidebar:
    st.header("🏢 1. Estrutura de Custos")
    fixas_total = st.number_input("Despesas Fixas Mensais (R$)", value=15000.0, step=500.0)
    qtd_pessoas = st.number_input("Nº de Colaboradores Diretos", value=2, min_value=1)
    horas_mes = st.number_input("Horas Contratuais/Mês", value=160)
    
    st.header("📈 2. Premissas de Venda")
    margem_alvo = st.slider("Margem de Lucro Desejada (%)", 5, 80, 40)
    impostos = st.number_input("Impostos e Taxas (%)", value=10.0)
    comissao = st.number_input("Comissões de Venda (%)", value=5.0)
    
    st.header("💰 3. Fluxo de Caixa")
    caixa_atual = st.number_input("Saldo em Caixa (R$)", value=50000.0)
    churn_rate = st.slider("Churn Rate Mensal (%)", 0.0, 20.0, 5.0)

# ==========================================
# 4. PROCESSAMENTO DOS DADOS
# ==========================================

# Cálculo de Capacidade (85% de eficiência conforme CAPACIDADE PRODUTIVA.csv)
capacidade_real = (qtd_pessoas * horas_mes) * 0.85
custo_hora_tecnico = fixas_total / capacidade_real

# Cálculo de Preço via Markup
markup = calcular_markup_multiplicador(impostos, comissao, margem_alvo)
preco_sugerido_hora = custo_hora_tecnico * markup

# ==========================================
# 5. DASHBOARD PRINCIPAL (MÉTRICAS)
# ==========================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Custo Hora Real", brl(custo_hora_tecnico))
    st.caption("Base: 85% Produtividade")
with c2:
    st.metric("Markup Aplicado", f"{markup:.2f}x")
    st.caption("Fórmula: 100 / (100 - x)")
with c3:
    st.metric("Preço de Venda/h", brl(preco_sugerido_hora))
    st.success("Preço Mínimo Sugerido")
with c4:
    # Break-even em horas
    horas_ponto_equilibrio = fixas_total / (preco_sugerido_hora - (preco_sugerido_hora * (impostos+comissao)/100))
    st.metric("Break-even (Horas)", f"{int(horas_ponto_equilibrio)}h")

st.markdown("---")

# ==========================================
# 6. SIMULADOR DE VENDAS E PRAZOS
# ==========================================
st.subheader("🛒 Simulador de Negociação e Parcelamento")
col_v, col_p = st.columns(2)

with col_v:
    horas_projeto = st.number_input("Horas Estimadas para o Projeto/Serviço", value=40)
    valor_total_avista = horas_projeto * preco_sugerido_hora
    st.write(f"**Valor Total à Vista:** {brl(valor_total_avista)}")

with col_p:
    n_parcelas = st.select_slider("Parcelamento (Meses)", options=[1, 2, 3, 6, 10, 12, 24])
    # Juros de 1.23% extraído do arquivo 'CÁLCULO PREÇO À PRAZO.csv'
    valor_total_prazo = calcular_venda_prazo(valor_total_avista, n_parcelas, 0.0123)
    st.write(f"**Valor Total a Prazo:** {brl(valor_total_prazo)}")
    st.write(f"**Parcelas de:** {brl(valor_total_prazo/n_parcelas)}")

st.markdown("---")

# ==========================================
# 7. PROJEÇÃO DE 12 MESES (DRE + CAIXA)
# ==========================================
st.subheader("📅 Projeção de Performance (Próximos 12 meses)")

vendas_h_mes = st.slider("Expectativa de Vendas Mensais (Horas)", 10, int(capacidade_real), int(capacidade_real*0.6))

lista_meses = []
caixa_acumulado = caixa_atual
receita_total = vendas_h_mes * preco_sugerido_hora

for i in range(1, 13):
    # Aplicação de Churn na receita a partir do mês 2
    receita_ajustada = receita_total * ((1 - churn_rate/100)**(i-1))
    impostos_pagos = receita_ajustada * (impostos/100)
    comissoes_pagas = receita_ajustada * (comissao/100)
    
    margem_contribuicao = receita_ajustada - impostos_pagos - comissoes_pagas
    resultado_mes = margem_contribuicao - fixas_total
    caixa_acumulado += resultado_mes
    
    lista_meses.append({
        "Mês": f"Mês {i}",
        "Receita Bruta": receita_ajustada,
        "Custos/Impostos": impostos_pagos + comissoes_pagas,
        "Resultado Líquido": resultado_mes,
        "Saldo em Caixa": max(caixa_acumulado, 0)
    })

df_projeção = pd.DataFrame(lista_meses)

tab1, tab2 = st.tabs(["📊 Gráfico de Caixa", "📋 Tabela DRE Simplificada"])

with tab1:
    st.area_chart(df_projeção.set_index("Mês")["Saldo em Caixa"])
    if caixa_acumulado < 0:
        st.error(f"⚠️ Alerta: O caixa zera no {df_projeção[df_projeção['Saldo em Caixa'] <= 0]['Mês'].iloc[0]}")
    else:
        runway = "Infinito" if resultado_mes > 0 else f"{caixa_acumulado/abs(resultado_mes):.1f} meses"
        st.success(f"✅ Runway estimado: {runway}")

with tab2:
    st.dataframe(df_projeção.style.format({
        "Receita Bruta": brl, "Custos/Impostos": brl, 
        "Resultado Líquido": brl, "Saldo em Caixa": brl
    }), use_container_width=True)

# ==========================================
# 8. EXPORTAÇÃO (RELATÓRIO)
# ==========================================
st.markdown("---")
if st.button("📄 Gerar Relatório Executivo"):
    relatorio = f"""
    --- RELATÓRIO DE VIABILIDADE FINANCEIRA ---
    Data: {datetime.now().strftime('%d/%m/%Y')}
    
    1. PRECIFICAÇÃO:
       - Custo Hora: {brl(custo_hora_tecnico)}
       - Markup: {markup:.2f}x
       - Preço de Venda/h: {brl(preco_sugerido_hora)}
       
    2. OPERAÇÃO:
       - Break-even: {int(horas_ponto_equilibrio)} horas/mês
       - Capacidade Real: {capacidade_real} horas/mês
       
    3. PROJEÇÃO:
       - Receita Mensal Esperada: {brl(receita_total)}
       - Burn Rate (se houver): {brl(min(0, resultado_mes))}
       - Status de Caixa: {'Lucrativo' if resultado_mes > 0 else 'Em queima de caixa'}
    -------------------------------------------
    """
    st.code(relatorio, language="text")
