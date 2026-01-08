# app.py
# FinanceOps — UX-first MVP (Streamlit)
# Navegação por módulos (sidebar), “resultado em 60s”, presets, 3 cenários antes de eventos,
# ações com impacto imediato e relatório copiável.

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

st.set_page_config(page_title="FinanceOps", layout="wide")

# =========================
# Helpers
# =========================
def brl(x: float) -> str:
    s = f"{x:,.2f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")

def pct(x: float) -> str:
    return f"{x:.1f}%"

def clamp_min(valor, minimo):
    return max(valor, minimo)

def seasonal_multiplier(seasonality_on: bool, seasonality_list, mes_abs: int) -> float:
    if not seasonality_on:
        return 1.0
    idx = (mes_abs - 1) % 12
    return float(seasonality_list[idx])

def apply_ramp(value: float, month_relative: int, ramp_months: int) -> float:
    if ramp_months <= 0:
        return value
    factor = clamp_min(month_relative / ramp_months, 1.0)
    return value * factor

# =========================
# Data structures (events)
# =========================
@dataclass
class Hiring:
    start_month: int
    monthly_cost: float
    ramp_months: int = 0
    revenue_impact: float = 0.0  # opcional

@dataclass
class Investment:
    month: int
    value: float
    kind: str  # "OPEX" | "CAPEX" | "APORTE"
    amort_months: int = 0  # para CAPEX

@dataclass
class CostCut:
    start_month: int
    fixed_reduction_pct: float = 0.0   # 0..1
    variable_reduction_pct: float = 0.0
    duration_months: int = 0  # 0 = indefinido

# =========================
# Simulation engine
# =========================
def simulate_financeops(
    horizon_months: int,
    revenue_base: float,
    revenue_growth_m: float,
    fixed_cost_base: float,
    var_cost_base: float,
    cost_growth_m: float,
    cash_initial: float,
    tax_pct: float = 0.0,
    cogs_pct: float = 0.0,
    seasonality_on: bool = False,
    seasonality_12: list | None = None,
    hirings: list[Hiring] | None = None,
    investments: list[Investment] | None = None,
    cuts: list[CostCut] | None = None,
):
    hirings = hirings or []
    investments = investments or []
    cuts = cuts or []
    seasonality_12 = seasonality_12 or [1.0] * 12

    months = np.arange(1, horizon_months + 1)

    receita_bruta = np.zeros(horizon_months, dtype=float)
    impostos = np.zeros(horizon_months, dtype=float)
    cogs = np.zeros(horizon_months, dtype=float)
    receita_liq = np.zeros(horizon_months, dtype=float)

    fixed_cost = np.zeros(horizon_months, dtype=float)
    var_cost = np.zeros(horizon_months, dtype=float)
    event_cost = np.zeros(horizon_months, dtype=float)

    resultado = np.zeros(horizon_months, dtype=float)
    margem_pct = np.zeros(horizon_months, dtype=float)
    burn = np.zeros(horizon_months, dtype=float)

    cash = np.zeros(horizon_months + 1, dtype=float)
    cash[0] = cash_initial

    # CAPEX amortization schedule
    capex_monthly_add = np.zeros(horizon_months, dtype=float)
    for inv in investments:
        if inv.kind == "CAPEX" and inv.amort_months and inv.amort_months > 0:
            start = inv.month
            for m in range(start, min(horizon_months, start + inv.amort_months) + 1):
                capex_monthly_add[m - 1] += inv.value / inv.amort_months

    def is_cut_active(cut: CostCut, m: int) -> bool:
        if m < cut.start_month:
            return False
        if cut.duration_months and cut.duration_months > 0:
            return m <= (cut.start_month + cut.duration_months - 1)
        return True

    for i, m in enumerate(months):
        # base revenue
        if m == 1:
            base_rev = revenue_base
        else:
            base_rev = receita_bruta[i - 1] * (1 + revenue_growth_m)

        base_rev *= seasonal_multiplier(seasonality_on, seasonality_12, m)

        # base costs
        if m == 1:
            fc = fixed_cost_base
            vc = var_cost_base
        else:
            fc = fixed_cost[i - 1] * (1 + cost_growth_m)
            vc = var_cost[i - 1] * (1 + cost_growth_m)

        # apply cost cuts
        fc_mult = 1.0
        vc_mult = 1.0
        for cut in cuts:
            if is_cut_active(cut, m):
                fc_mult *= (1 - cut.fixed_reduction_pct)
                vc_mult *= (1 - cut.variable_reduction_pct)
        fc *= fc_mult
        vc *= vc_mult

        # events: hirings
        ev_cost = 0.0
        ev_rev = 0.0
        for h in hirings:
            if m >= h.start_month:
                rel = m - h.start_month + 1
                ev_cost += apply_ramp(h.monthly_cost, rel, h.ramp_months)
                if h.revenue_impact:
                    ev_rev += apply_ramp(h.revenue_impact, rel, h.ramp_months)

        # investments
        aporte_mes = 0.0
        for inv in investments:
            if inv.month == m:
                if inv.kind == "OPEX":
                    ev_cost += inv.value
                elif inv.kind == "APORTE":
                    aporte_mes += inv.value

        ev_cost += capex_monthly_add[i]

        # finalize revenue
        receita_bruta[i] = base_rev + ev_rev
        impostos[i] = receita_bruta[i] * tax_pct
        cogs[i] = receita_bruta[i] * cogs_pct
        receita_liq[i] = receita_bruta[i] - impostos[i] - cogs[i]

        fixed_cost[i] = fc
        var_cost[i] = vc
        event_cost[i] = ev_cost

        total_cost = fc + vc + ev_cost
        resultado[i] = receita_liq[i] - total_cost

        margem_pct[i] = (resultado[i] / receita_bruta[i] * 100) if receita_bruta[i] > 0 else 0.0
        burn[i] = -resultado[i] if resultado[i] < 0 else 0.0

        cash[i + 1] = cash[i] + resultado[i] + aporte_mes

    cash_series = cash[1:]

    df = pd.DataFrame({
        "Mês": months,
        "Receita Bruta": receita_bruta,
        "Impostos": impostos,
        "COGS": cogs,
        "Receita Líquida": receita_liq,
        "Custos Fixos": fixed_cost,
        "Custos Variáveis": var_cost,
        "Custos Eventos": event_cost,
        "Custos Totais": fixed_cost + var_cost + event_cost,
        "Resultado": resultado,
        "Margem (%)": margem_pct,
        "Burn": burn,
        "Caixa": cash_series,
    })

    breakeven = int(df.loc[df["Resultado"] >= 0, "Mês"].min()) if (df["Resultado"] >= 0).any() else None
    caixa_neg = int(df.loc[df["Caixa"] < 0, "Mês"].min()) if (df["Caixa"] < 0).any() else None

    burn_ult = df["Burn"].tail(3).mean()
    runway = float(df["Caixa"].iloc[-1] / burn_ult) if burn_ult and burn_ult > 0 else None

    return df, breakeven, caixa_neg, runway

# =========================
# Session state defaults
# =========================
def init_state():
    if "drivers" not in st.session_state:
        st.session_state.drivers = {
            "horizon": 24,
            "revenue_base": 50000.0,
            "fixed_cost_base": 20000.0,
            "var_cost_base": 10000.0,
            "cash_initial": 100000.0,
            "revenue_growth_pct": 5.0,
            "cost_growth_pct": 2.0,
            "tax_pct": 0.0,
            "cogs_pct": 0.0,
            "seasonality_on": False,
            "seasonality_12": [1.0] * 12,
            "three_scenarios": True,
            # deltas em pontos percentuais
            "opt_revenue_delta": 3.0,
            "pess_revenue_delta": -3.0,
            "opt_cost_delta": -1.0,
            "pess_cost_delta": 1.0,
        }
    if "hirings" not in st.session_state:
        st.session_state.hirings = []
    if "investments" not in st.session_state:
        st.session_state.investments = []
    if "cuts" not in st.session_state:
        st.session_state.cuts = []
    if "results" not in st.session_state:
        st.session_state.results = None  # cache da última simulação (cenários)

init_state()

d = st.session_state.drivers

# =========================
# UI: Sidebar navigation + summary
# =========================
st.sidebar.title("FinanceOps")

page = st.sidebar.radio(
    "Navegação",
    ["Estado atual", "Hipóteses", "Cenários", "Ações (eventos)", "Relatório"],
    index=0
)

st.sidebar.divider()

# Completion status
def filled_status():
    base_ok = (d["revenue_base"] > 0) and (d["fixed_cost_base"] >= 0) and (d["var_cost_base"] >= 0)
    hypo_ok = True  # sliders sempre têm valor
    return base_ok, hypo_ok

base_ok, hypo_ok = filled_status()

st.sidebar.markdown("### Status")
st.sidebar.write(f"• Estado atual: {'✅' if base_ok else '⏳'}")
st.sidebar.write(f"• Hipóteses: {'✅' if hypo_ok else '⏳'}")
st.sidebar.write(f"• Cenários: {'✅' if st.session_state.results else '⏳'}")

st.sidebar.divider()

# Sticky summary (“painel de controle”)
st.sidebar.markdown("### Resumo do cenário")
st.sidebar.caption("Sempre visível enquanto você navega.")
st.sidebar.write(f"Receita: **{brl(d['revenue_base'])}**")
st.sidebar.write(f"Fixos: **{brl(d['fixed_cost_base'])}**")
st.sidebar.write(f"Variáveis: **{brl(d['var_cost_base'])}**")
st.sidebar.write(f"Caixa: **{brl(d['cash_initial'])}**")
st.sidebar.write(f"Cresc. Receita: **{pct(d['revenue_growth_pct'])}/m**")
st.sidebar.write(f"Cresc. Custos: **{pct(d['cost_growth_pct'])}/m**")
st.sidebar.write(f"Horizonte: **{d['horizon']} meses**")

# =========================
# Top header
# =========================
st.title("📊 FinanceOps — Copiloto Financeiro (MVP)")

# =========================
# Page: Estado atual (resultado em 60s)
# =========================
if page == "Estado atual":
    st.subheader("1) Estado atual — preencha o mínimo e já veja valor")

    left, right = st.columns([1, 1])

    with left:
        st.info("Meta: você sair daqui com **Resultado, Margem, Burn e um sinal de risco** em menos de 60 segundos.")

        d["revenue_base"] = st.number_input(
            "Receita do mês (R$)",
            min_value=0.0, step=1000.0, value=float(d["revenue_base"]),
            help="Use o último mês real. Se estiver instável, use a média dos últimos 3 meses."
        )
        d["fixed_cost_base"] = st.number_input(
            "Custos fixos (R$)",
            min_value=0.0, step=500.0, value=float(d["fixed_cost_base"]),
            help="Custos que existem mesmo se você vender zero: time fixo, aluguel, ferramentas base, contabilidade, etc."
        )
        d["var_cost_base"] = st.number_input(
            "Custos variáveis (R$)",
            min_value=0.0, step=500.0, value=float(d["var_cost_base"]),
            help="Custos que crescem com vendas: taxas, comissões, mídia proporcional, insumos, frete."
        )
        d["cash_initial"] = st.number_input(
            "Caixa disponível (R$)",
            min_value=0.0, step=1000.0, value=float(d["cash_initial"]),
            help="Saldo de conta + reserva de liquidez. Isso alimenta runway e risco de caixa negativo."
        )

        with st.expander("Como preencher (rápido)"):
            st.markdown(
                "- **Receita**: último mês ou média 3 meses.\n"
                "- **Fixos**: tudo que você pagaria mesmo vendendo zero.\n"
                "- **Variáveis**: taxas/comissões/insumos.\n"
                "- **Caixa**: dinheiro disponível hoje."
            )

    # instant diagnostics
    receita = d["revenue_base"]
    custos_totais = d["fixed_cost_base"] + d["var_cost_base"]
    resultado = receita - custos_totais
    margem = (resultado / receita * 100) if receita > 0 else 0.0
    burn = -resultado if resultado < 0 else 0.0

    with right:
        st.markdown("### Diagnóstico instantâneo")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resultado", brl(resultado))
        c2.metric("Margem", f"{margem:.1f}%")
        c3.metric("Burn", brl(burn))
        c4.metric("Custos Totais", brl(custos_totais))

        # Simple product alerts
        if receita > 0:
            fix_ratio = d["fixed_cost_base"] / receita
            if fix_ratio >= 0.8:
                st.error("Rigidez alta: custos fixos ≥ 80% da receita. Você fica sem manobra.")
            elif fix_ratio >= 0.6:
                st.warning("Rigidez moderada: custos fixos ≥ 60% da receita. Atenção em meses ruins.")
            else:
                st.success("Boa flexibilidade: custos fixos abaixo de 60% da receita.")
        if resultado < 0:
            if burn > 0:
                runway = d["cash_initial"] / burn if burn > 0 else None
                if runway is not None:
                    st.warning(f"Runway simples (se nada mudar): ~**{runway:.1f} meses**.")
        else:
            st.success("Você está positivo no mês atual. Próximo passo: testar cenários.")

        st.caption("Próximo: vá em **Hipóteses** e depois **Cenários** para ver o futuro.")

# =========================
# Page: Hipóteses (presets + sliders)
# =========================
elif page == "Hipóteses":
    st.subheader("2) Hipóteses — ajuste o motor sem virar planilha")

    st.info("Escolha um preset para começar (1 clique) e depois refine. Isso reduz chute e acelera validação.")

    p1, p2, p3 = st.columns(3)

    def set_preset(name):
        if name == "Conservador":
            d["revenue_growth_pct"] = 2.0
            d["cost_growth_pct"] = 2.0
        elif name == "Realista":
            d["revenue_growth_pct"] = 5.0
            d["cost_growth_pct"] = 2.0
        elif name == "Agressivo":
            d["revenue_growth_pct"] = 10.0
            d["cost_growth_pct"] = 3.0
        st.session_state.results = None  # invalidate cache

    if p1.button("Preset: Conservador"):
        set_preset("Conservador")
    if p2.button("Preset: Realista"):
        set_preset("Realista")
    if p3.button("Preset: Agressivo"):
        set_preset("Agressivo")

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        d["horizon"] = st.selectbox("Horizonte (meses)", [12, 24, 36], index=[12, 24, 36].index(d["horizon"]))
    with c2:
        d["revenue_growth_pct"] = st.slider(
            "Crescimento mensal da receita (%)",
            min_value=-50.0, max_value=80.0, step=0.5,
            value=float(d["revenue_growth_pct"]),
            help="Ex.: 5% a.m. = multiplica por 1,05 todo mês."
        )
    with c3:
        d["cost_growth_pct"] = st.slider(
            "Crescimento mensal dos custos (%)",
            min_value=-20.0, max_value=50.0, step=0.5,
            value=float(d["cost_growth_pct"]),
            help="Reajustes e expansão. Se vai contratar, prefira modelar em Ações."
        )

    st.markdown("### Ajustes (opcionais)")
    o1, o2, o3 = st.columns(3)
    with o1:
        d["tax_pct"] = st.slider(
            "Imposto médio sobre receita (%)",
            0.0, 25.0, float(d["tax_pct"]), 0.5,
            help="Se não souber, deixe 0%. MVP usa alíquota média."
        )
    with o2:
        d["cogs_pct"] = st.slider(
            "COGS / custo direto (%)",
            0.0, 80.0, float(d["cogs_pct"]), 0.5,
            help="Se você tem custo direto por entrega/venda."
        )
    with o3:
        d["three_scenarios"] = st.toggle(
            "Comparar 3 cenários (recomendado)",
            value=bool(d["three_scenarios"]),
            help="Mostra Base/Otimista/Pessimista automaticamente."
        )

    with st.expander("Entenda (sem economês)"):
        st.markdown(
            "- **Crescimento da receita**: o quanto você espera melhorar mês a mês.\n"
            "- **Crescimento dos custos**: inflação + expansão. Se você vai contratar, modele como evento.\n"
            "- **Imposto/COGS**: opcional; só aumenta realismo.\n"
            "- **3 cenários**: te protege do autoengano (principalmente no pessimista)."
        )

# =========================
# Page: Cenários (wow + leitura)
# =========================
elif page == "Cenários":
    st.subheader("3) Cenários — veja risco e amplitude (sem rodar 20 vezes)")

    if not base_ok:
        st.warning("Preencha **Estado atual** primeiro (receita/custos/caixa).")
    else:
        left, right = st.columns([1, 2])

        with left:
            st.info("Aqui o jogo muda: você enxerga **quando o caixa quebra** e **quando chega no break-even**.")
            if d["three_scenarios"]:
                st.markdown("### Ajuste dos cenários")
                d["opt_revenue_delta"] = st.slider("Otimista: +pp crescimento receita", 0.0, 20.0, float(d["opt_revenue_delta"]), 0.5)
                d["pess_revenue_delta"] = st.slider("Pessimista: +pp crescimento receita", -20.0, 0.0, float(d["pess_revenue_delta"]), 0.5)
                d["opt_cost_delta"] = st.slider("Otimista: +pp crescimento custos", -10.0, 0.0, float(d["opt_cost_delta"]), 0.5)
                d["pess_cost_delta"] = st.slider("Pessimista: +pp crescimento custos", 0.0, 10.0, float(d["pess_cost_delta"]), 0.5)

            run = st.button("🚀 Rodar cenários", type="primary")

        def run_one(label, rg_pct, cg_pct):
            df, be, neg, runway = simulate_financeops(
                horizon_months=int(d["horizon"]),
                revenue_base=float(d["revenue_base"]),
                revenue_growth_m=float(rg_pct) / 100,
                fixed_cost_base=float(d["fixed_cost_base"]),
                var_cost_base=float(d["var_cost_base"]),
                cost_growth_m=float(cg_pct) / 100,
                cash_initial=float(d["cash_initial"]),
                tax_pct=float(d["tax_pct"]) / 100,
                cogs_pct=float(d["cogs_pct"]) / 100,
                seasonality_on=bool(d["seasonality_on"]),
                seasonality_12=d["seasonality_12"],
                hirings=st.session_state.hirings,
                investments=st.session_state.investments,
                cuts=st.session_state.cuts,
            )
            return {"label": label, "df": df, "breakeven": be, "cash_neg": neg, "runway": runway}

        if run or (st.session_state.results is None):
            base_rg = float(d["revenue_growth_pct"])
            base_cg = float(d["cost_growth_pct"])

            scenarios = [run_one("Base", base_rg, base_cg)]

            if d["three_scenarios"]:
                opt_rg = base_rg + float(d["opt_revenue_delta"])
                pess_rg = base_rg + float(d["pess_revenue_delta"])
                opt_cg = base_cg + float(d["opt_cost_delta"])
                pess_cg = base_cg + float(d["pess_cost_delta"])
                scenarios.append(run_one("Otimista", opt_rg, opt_cg))
                scenarios.append(run_one("Pessimista", pess_rg, pess_cg))

            st.session_state.results = scenarios

        scenarios = st.session_state.results

        with right:
            # Summary table
            rows = []
            for s in scenarios:
                df = s["df"]
                rows.append({
                    "Cenário": s["label"],
                    "Caixa final": df["Caixa"].iloc[-1],
                    "Resultado final": df["Resultado"].iloc[-1],
                    "Break-even (mês)": s["breakeven"] if s["breakeven"] else "—",
                    "Caixa negativo (mês)": s["cash_neg"] if s["cash_neg"] else "—",
                    "Runway (meses)": round(s["runway"], 1) if s["runway"] is not None else "—",
                })
            s_df = pd.DataFrame(rows)

            st.markdown("### Leitura rápida (board-ready)")
            st.dataframe(
                s_df.style.format({
                    "Caixa final": lambda v: brl(v) if isinstance(v, (int, float)) else v,
                    "Resultado final": lambda v: brl(v) if isinstance(v, (int, float)) else v,
                }),
                use_container_width=True
            )

            # Charts
            chart_cash = pd.DataFrame({"Mês": scenarios[0]["df"]["Mês"]}).set_index("Mês")
            chart_res = pd.DataFrame({"Mês": scenarios[0]["df"]["Mês"]}).set_index("Mês")
            for s in scenarios:
                chart_cash[s["label"]] = s["df"].set_index("Mês")["Caixa"]
                chart_res[s["label"]] = s["df"].set_index("Mês")["Resultado"]

            st.caption("Caixa por cenário")
            st.line_chart(chart_cash)

            st.caption("Resultado mensal por cenário")
            st.line_chart(chart_res)

            # Product-style alerts based on pessimistic
            pess = next((x for x in scenarios if x["label"] == "Pessimista"), None)
            base = next((x for x in scenarios if x["label"] == "Base"), scenarios[0])

            st.markdown("### Alertas do produto")
            if pess and pess["cash_neg"]:
                st.error(f"No **Pessimista**, o caixa fica negativo no mês **{pess['cash_neg']}**. Seu plano está frágil.")
            elif base["cash_neg"]:
                st.warning(f"No **Base**, o caixa fica negativo no mês **{base['cash_neg']}**. Você precisa ajustar preço/custo/crescimento ou usar ações.")
            else:
                st.success("Sem caixa negativo no horizonte (no cenário base). Próximo passo: testar Ações para acelerar ou proteger.")

            st.caption("Próximo: vá em **Ações (eventos)** para simular decisões reais (contratar, investir, aporte, corte).")

# =========================
# Page: Ações (eventos) — decisões com impacto imediato
# =========================
elif page == "Ações (eventos)":
    st.subheader("4) Ações (eventos) — simule decisões reais")

    st.info("Aqui você para de mexer em % e começa a tomar decisão: contratar, investir, receber aporte, cortar custo.")

    # Quick view: counts
    a1, a2, a3 = st.columns(3)
    a1.metric("Contratações", len(st.session_state.hirings))
    a2.metric("Investimentos/Aportes", len(st.session_state.investments))
    a3.metric("Cortes", len(st.session_state.cuts))

    st.divider()

    colL, colR = st.columns([1, 1])

    with colL:
        st.markdown("## ➕ Adicionar ação")
        action_type = st.selectbox("Tipo de ação", ["Contratação", "Investimento / Aporte", "Corte de custo"])

        if action_type == "Contratação":
            st.caption("Quando contratar muda o custo fixo (e opcionalmente gera receita).")
            c1, c2, c3, c4 = st.columns(4)
            start = c1.number_input("Mês início", 1, d["horizon"], 3, 1)
            cost = c2.number_input("Custo mensal total (R$)", 0.0, 1e9, 8000.0, 500.0)
            ramp = c3.number_input("Ramp (meses)", 0, 12, 1, 1)
            rev_imp = c4.number_input("Impacto em receita (R$/mês)", 0.0, 1e9, 0.0, 500.0)
            if st.button("Salvar contratação", type="primary"):
                st.session_state.hirings.append(Hiring(int(start), float(cost), int(ramp), float(rev_imp)))
                st.session_state.results = None

        elif action_type == "Investimento / Aporte":
            st.caption("OPEX vira custo no mês. CAPEX pode amortizar. APORTE entra no caixa.")
            c1, c2, c3, c4 = st.columns(4)
            m = c1.number_input("Mês", 1, d["horizon"], 2, 1)
            v = c2.number_input("Valor (R$)", 0.0, 1e12, 10000.0, 1000.0)
            kind = c3.selectbox("Tipo", ["OPEX", "CAPEX", "APORTE"])
            amort = c4.number_input("Amortização (meses)", 0, 36, 0, 1)
            if st.button("Salvar investimento/aporte", type="primary"):
                st.session_state.investments.append(Investment(int(m), float(v), kind, int(amort)))
                st.session_state.results = None

        else:
            st.caption("Reduz custos fixos/variáveis a partir de um mês (por tempo definido ou até o fim).")
            c1, c2, c3, c4 = st.columns(4)
            sm = c1.number_input("Mês início", 1, d["horizon"], 4, 1)
            fr = c2.slider("Redução fixos (%)", 0.0, 50.0, 0.0, 0.5)
            vr = c3.slider("Redução variáveis (%)", 0.0, 50.0, 0.0, 0.5)
            dur = c4.number_input("Duração (meses) — 0=até fim", 0, 36, 0, 1)
            if st.button("Salvar corte", type="primary"):
                st.session_state.cuts.append(CostCut(int(sm), fr/100, vr/100, int(dur)))
                st.session_state.results = None

        st.divider()

        if st.button("🧹 Limpar TODAS as ações"):
            st.session_state.hirings = []
            st.session_state.investments = []
            st.session_state.cuts = []
            st.session_state.results = None

    with colR:
        st.markdown("## 📌 Ações registradas")
        if st.session_state.hirings:
            st.caption("Contratações")
            st.dataframe(pd.DataFrame([asdict(x) for x in st.session_state.hirings]), use_container_width=True)
        if st.session_state.investments:
            st.caption("Investimentos/Aportes")
            st.dataframe(pd.DataFrame([asdict(x) for x in st.session_state.investments]), use_container_width=True)
        if st.session_state.cuts:
            st.caption("Cortes")
            st.dataframe(pd.DataFrame([asdict(x) for x in st.session_state.cuts]), use_container_width=True)

        st.divider()

        st.markdown("## ⚡ Ver impacto agora")
        if st.button("Recalcular cenários com ações", type="primary"):
            st.session_state.results = None
            st.success("Ações aplicadas. Vá em **Cenários** para ver o impacto.")

        st.caption("A melhor UX aqui é: adicionar uma ação → recalcular → ver o caixa mudar. Sem planilha, sem drama.")

# =========================
# Page: Relatório (copiável + CSV)
# =========================
else:
    st.subheader("5) Relatório — pronto pra compartilhar")

    if not st.session_state.results:
        st.warning("Rode os **Cenários** primeiro para gerar o relatório.")
    else:
        scenarios = st.session_state.results
        base = next((x for x in scenarios if x["label"] == "Base"), scenarios[0])
        df_base = base["df"]

        # Executive summary
        st.markdown("### Resumo executivo (Base)")
        be = base["breakeven"]
        neg = base["cash_neg"]
        runway = base["runway"]

        bullets = [
            f"• Receita atual: {brl(d['revenue_base'])}",
            f"• Custos fixos: {brl(d['fixed_cost_base'])} | Custos variáveis: {brl(d['var_cost_base'])}",
            f"• Caixa inicial: {brl(d['cash_initial'])}",
            f"• Crescimento receita (base): {pct(d['revenue_growth_pct'])}/m | Crescimento custos (base): {pct(d['cost_growth_pct'])}/m",
            f"• Horizonte: {d['horizon']} meses",
            f"• Receita final (base): {brl(df_base['Receita Bruta'].iloc[-1])}",
            f"• Resultado final (base): {brl(df_base['Resultado'].iloc[-1])}",
            f"• Caixa final (base): {brl(df_base['Caixa'].iloc[-1])}",
        ]
        if be:
            bullets.append(f"• Break-even: mês {be}.")
        else:
            bullets.append("• Break-even: não ocorre no horizonte.")
        if neg:
            bullets.append(f"• Risco crítico: caixa negativo no mês {neg}.")
        else:
            bullets.append("• Caixa: sem mês negativo no horizonte.")
        if runway is not None:
            bullets.append(f"• Runway estimada (burn médio 3m): {runway:.1f} meses.")

        report_text = "\n".join(bullets)
        st.text_area("Texto (copiar e colar)", report_text, height=220)

        st.markdown("### Gráficos (Base)")
        st.line_chart(df_base.set_index("Mês")[["Receita Bruta", "Custos Totais", "Resultado"]])
        st.line_chart(df_base.set_index("Mês")[["Caixa", "Burn"]])

        with st.expander("Tabela completa (Base)"):
            st.dataframe(
                df_base.style.format({
                    "Receita Bruta": lambda v: brl(v),
                    "Impostos": lambda v: brl(v),
                    "COGS": lambda v: brl(v),
                    "Receita Líquida": lambda v: brl(v),
                    "Custos Fixos": lambda v: brl(v),
                    "Custos Variáveis": lambda v: brl(v),
                    "Custos Eventos": lambda v: brl(v),
                    "Custos Totais": lambda v: brl(v),
                    "Resultado": lambda v: brl(v),
                    "Burn": lambda v: brl(v),
                    "Caixa": lambda v: brl(v),
                    "Margem (%)": "{:.2f}%"
                }),
                use_container_width=True
            )

        st.markdown("### Export")
        csv = df_base.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV do cenário Base", data=csv, file_name="financeops_base.csv", mime="text/csv")

        st.caption("V2 natural: PDF + salvar cenários (SQLite) + login. Mas MVP já está vendável.")

