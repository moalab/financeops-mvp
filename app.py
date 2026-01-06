import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="FinanceOps MVP", layout="wide")

# -----------------------------
# Motor de simulação (MVP)
# -----------------------------
def simulate(
    receita_base: float,
    custos_fixos_base: float,
    custos_variaveis_base: float,
    crescimento_receita_pct: float,
    crescimento_custos_pct: float,
    caixa_inicial: float,
    horizonte_meses: int,
):
    meses = np.arange(1, horizonte_meses + 1)

    receita = np.zeros(horizonte_meses, dtype=float)
    custos_fixos = np.zeros(horizonte_meses, dtype=float)
    custos_variaveis = np.zeros(horizonte_meses, dtype=float)

    for i in range(horizonte_meses):
        if i == 0:
            receita[i] = receita_base
            custos_fixos[i] = custos_fixos_base
            custos_variaveis[i] = custos_variaveis_base
        else:
            receita[i] = receita[i - 1] * (1 + crescimento_receita_pct)
            custos_fixos[i] = custos_fixos[i - 1] * (1 + crescimento_custos_pct)
            custos_variaveis[i] = custos_variaveis[i - 1] * (1 + crescimento_custos_pct)

    custos_total = custos_fixos + custos_variaveis
    resultado = receita - custos_total
    margem_pct = np.where(receita > 0, (resultado / receita) * 100, 0.0)
    burn = np.where(resultado < 0, -resultado, 0.0)

    caixa = np.zeros(horizonte_meses + 1, dtype=float)
    caixa[0] = caixa_inicial
    for i in range(horizonte_meses):
        caixa[i + 1] = caixa[i] + resultado[i]
    caixa = caixa[1:]  # alinhado aos meses

    df = pd.DataFrame(
        {
            "Mês": meses,
            "Receita": receita,
            "Custos Fixos": custos_fixos,
            "Custos Variáveis": custos_variaveis,
            "Custos Totais": custos_total,
            "Resultado": resultado,
            "Margem (%)": margem_pct,
            "Burn": burn,
            "Caixa": caixa,
        }
    )

    breakeven_mes = int(df.loc[df["Resultado"] >= 0, "Mês"].min()) if (df["Resultado"] >= 0).any() else None
    caixa_negativo_mes = int(df.loc[df["Caixa"] < 0, "Mês"].min()) if (df["Caixa"] < 0).any() else None

    # runway simples (média burn últimos 3 meses)
    runway = None
    burn_ult = df["Burn"].tail(3).mean()
    if burn_ult and burn_ult > 0:
        runway = float(df["Caixa"].iloc[-1] / burn_ult)

    return df, breakeven_mes, caixa_negativo_mes, runway


def brl(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# -----------------------------
# UI: Navegação
# -----------------------------
st.title("FinanceOps — MVP (Streamlit)")

tabs = st.tabs(["1) Diagnóstico", "2) DRE", "3) Tendência", "4) Simulador", "5) Export"])

# -----------------------------
# Estado base (inputs do mês)
# -----------------------------
if "base" not in st.session_state:
    st.session_state.base = {
        "receita": 50000.0,
        "custos_fixos": 20000.0,
        "custos_variaveis": 10000.0,
        "caixa": 100000.0,
    }

# -----------------------------
# Tab 1: Diagnóstico
# -----------------------------
with tabs[0]:
    st.subheader("Diagnóstico financeiro (mês atual)")

    c1, c2, c3, c4 = st.columns(4)
    receita = c1.number_input("Receita (R$)", min_value=0.0, value=st.session_state.base["receita"], step=1000.0)
    custos_fixos = c2.number_input("Custos fixos (R$)", min_value=0.0, value=st.session_state.base["custos_fixos"], step=500.0)
    custos_variaveis = c3.number_input("Custos variáveis (R$)", min_value=0.0, value=st.session_state.base["custos_variaveis"], step=500.0)
    caixa = c4.number_input("Caixa atual (R$)", min_value=0.0, value=st.session_state.base["caixa"], step=1000.0)

    st.session_state.base.update(
        {"receita": receita, "custos_fixos": custos_fixos, "custos_variaveis": custos_variaveis, "caixa": caixa}
    )

    resultado = receita - (custos_fixos + custos_variaveis)
    margem = (resultado / receita * 100) if receita > 0 else 0.0
    burn = -resultado if resultado < 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Resultado", brl(resultado))
    k2.metric("Margem", f"{margem:.1f}%")
    k3.metric("Burn", brl(burn))
    k4.metric("Custos Totais", brl(custos_fixos + custos_variaveis))

# -----------------------------
# Tab 2: DRE simples
# -----------------------------
with tabs[1]:
    st.subheader("DRE simplificada (mês atual)")

    receita = st.session_state.base["receita"]
    custos_fixos = st.session_state.base["custos_fixos"]
    custos_variaveis = st.session_state.base["custos_variaveis"]
    custos_totais = custos_fixos + custos_variaveis
    resultado = receita - custos_totais

    dre = pd.DataFrame(
        {
            "Linha": ["Receita", "Custos Fixos", "Custos Variáveis", "Custos Totais", "Resultado"],
            "Valor": [receita, -custos_fixos, -custos_variaveis, -custos_totais, resultado],
        }
    )

    st.dataframe(dre.style.format({"Valor": lambda v: brl(v)}), use_container_width=True)

# -----------------------------
# Tab 3: Tendência (mock simples para MVP)
# -----------------------------
with tabs[2]:
    st.subheader("Tendência (rápido)")
    st.caption("No MVP, esta aba mostra a tendência da simulação. Na V2, entra histórico real mensal importado de planilha/CSV.")

    st.info("Dica: use o Simulador para gerar séries e visualizar tendência de Receita, Custos, Resultado e Caixa.")

# -----------------------------
# Tab 4: Simulador
# -----------------------------
with tabs[3]:
    st.subheader("Simulador de Cenários (CORE)")

    left, right = st.columns([1, 2])

    with left:
        horizonte = st.selectbox("Horizonte (meses)", [12, 24, 36], index=1)
        crescimento_receita = st.slider("Crescimento mensal da receita (%)", -50, 80, 5) / 100
        crescimento_custos = st.slider("Crescimento mensal dos custos (%)", -20, 50, 2) / 100

        run = st.button("Rodar simulação", type="primary")

    if run:
        df, breakeven, caixa_neg, runway = simulate(
            receita_base=st.session_state.base["receita"],
            custos_fixos_base=st.session_state.base["custos_fixos"],
            custos_variaveis_base=st.session_state.base["custos_variaveis"],
            crescimento_receita_pct=crescimento_receita,
            crescimento_custos_pct=crescimento_custos,
            caixa_inicial=st.session_state.base["caixa"],
            horizonte_meses=horizonte,
        )
        st.session_state["last_sim"] = df
        st.session_state["last_kpis"] = {"breakeven": breakeven, "caixa_neg": caixa_neg, "runway": runway}

    if "last_sim" in st.session_state:
        df = st.session_state["last_sim"]
        k = st.session_state["last_kpis"]

        with right:
            a, b, c, d = st.columns(4)
            a.metric("Receita final", brl(df["Receita"].iloc[-1]))
            b.metric("Resultado final", brl(df["Resultado"].iloc[-1]))
            c.metric("Caixa final", brl(df["Caixa"].iloc[-1]))
            d.metric("Margem final", f"{df['Margem (%)'].iloc[-1]:.1f}%")

            if k["caixa_neg"]:
                st.error(f"Caixa fica negativo no mês {k['caixa_neg']}.")
            else:
                st.success("Caixa permanece positivo no horizonte.")

            if k["breakeven"]:
                st.success(f"Break-even a partir do mês {k['breakeven']}.")
            else:
                st.warning("Não atinge break-even no horizonte.")

            if k["runway"] is not None:
                st.info(f"Runway estimada (burn médio 3m): {k['runway']:.1f} meses.")

            st.markdown("### Gráficos")
            st.line_chart(df.set_index("Mês")[["Receita", "Custos Totais", "Resultado"]])
            st.line_chart(df.set_index("Mês")[["Caixa", "Burn"]])

            with st.expander("Tabela completa"):
                st.dataframe(
                    df.style.format(
                        {
                            "Receita": lambda v: brl(v),
                            "Custos Fixos": lambda v: brl(v),
                            "Custos Variáveis": lambda v: brl(v),
                            "Custos Totais": lambda v: brl(v),
                            "Resultado": lambda v: brl(v),
                            "Burn": lambda v: brl(v),
                            "Caixa": lambda v: brl(v),
                            "Margem (%)": "{:.2f}%",
                        }
                    ),
                    use_container_width=True,
                )

            st.markdown("### Resumo executivo automático")
            resumo = [
                f"• Receita base: {brl(st.session_state.base['receita'])}",
                f"• Crescimento receita: {crescimento_receita*100:.1f}% ao mês",
                f"• Crescimento custos: {crescimento_custos*100:.1f}% ao mês",
                f"• Resultado final: {brl(df['Resultado'].iloc[-1])}",
                f"• Caixa final: {brl(df['Caixa'].iloc[-1])}",
            ]
            if k["breakeven"]:
                resumo.append(f"• Break-even: mês {k['breakeven']}.")
            else:
                resumo.append("• Break-even: não ocorre no horizonte.")
            if k["caixa_neg"]:
                resumo.append(f"• Risco crítico: caixa negativo no mês {k['caixa_neg']}.")
            else:
                resumo.append("• Caixa: sem mês negativo no horizonte.")
            st.write("\n".join(resumo))

# -----------------------------
# Tab 5: Export
# -----------------------------
with tabs[4]:
    st.subheader("Exportar resultados")

    if "last_sim" not in st.session_state:
        st.warning("Rode uma simulação primeiro.")
    else:
        df = st.session_state["last_sim"]
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV da simulação", data=csv, file_name="simulacao_financeops.csv", mime="text/csv")

        st.caption("V2: export PDF (relatório) + export Excel formatado + salvar cenários em banco.")
