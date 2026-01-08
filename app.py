# app.py — FinanceOps (UX Copilot MVP)
# - Navegação por módulos + progresso
# - Resultado em 60s
# - "Saiba mais" por campo
# - Sazonalidade com presets
# - 3 cenários (antes de eventos)
# - Eventos com impacto imediato (delta no caixa final)
# - Visual mais suave via CSS (sem libs extras)

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

st.set_page_config(page_title="FinanceOps", layout="wide")

# --------------------------
# Minimal CSS to "de-square"
# --------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2.2rem;}
      [data-testid="stSidebar"] {padding-top: 0.8rem;}
      .soft-card {border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; padding: 14px 14px; background: rgba(255,255,255,0.03);}
      .soft-pill {display:inline-block; padding: 4px 10px; border-radius: 999px; background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.08); font-size: 12px;}
      .muted {opacity: 0.85;}
      .kpi-row div[data-testid="stMetric"] {background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 10px 12px; border-radius: 16px;}
      .small {font-size: 12px; opacity:0.85;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Helpers
# =========================
def brl(x: float) -> str:
    s = f"{x:,.2f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")

def pct(x: float) -> str:
    return f"{x:.1f}%"

def seasonal_multiplier(seasonality_on: bool, seasonality_list, mes_abs: int) -> float:
    if not seasonality_on:
        return 1.0
    idx = (mes_abs - 1) % 12
    return float(seasonality_list[idx])

def apply_ramp(value: float, month_relative: int, ramp_months: int) -> float:
    if ramp_months <= 0:
        return value
    factor = min(month_relative / ramp_months, 1.0)
    return value * factor

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# =========================
# Events data structures
# =========================
@dataclass
class Hiring:
    start_month: int
    monthly_cost: float
    ramp_months: int = 0
    revenue_impact: float = 0.0

@dataclass
class Investment:
    month: int
    value: float
    kind: str  # "OPEX" | "CAPEX" | "APORTE"
    amort_months: int = 0

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
    seasonality_12: Optional[List[float]] = None,
    hirings: Optional[List[Hiring]] = None,
    investments: Optional[List[Investment]] = None,
    cuts: Optional[List[CostCut]] = None,
):
    hirings = hirings or []
    investments = investments or []
    cuts = cuts or []
    seasonality_12 = seasonality_12 or [1.0] * 12

    months = np.arange(1, horizon_months + 1)

    receita_bruta = np.zeros(horizon_months)
    impostos = np.zeros(horizon_months)
    cogs = np.zeros(horizon_months)
    receita_liq = np.zeros(horizon_months)

    fixos = np.zeros(horizon_months)
    variaveis = np.zeros(horizon_months)
    custos_eventos = np.zeros(horizon_months)

    resultado = np.zeros(horizon_months)
    margem = np.zeros(horizon_months)
    burn = np.zeros(horizon_months)

    caixa = np.zeros(horizon_months + 1)
    caixa[0] = cash_initial

    # CAPEX amortization
    capex_monthly_add = np.zeros(horizon_months)
    for inv in investments:
        if inv.kind == "CAPEX" and inv.amort_months and inv.amort_months > 0:
            start = inv.month
            for m in range(start, min(horizon_months, start + inv.amort_months) + 1):
                capex_monthly_add[m - 1] += inv.value / inv.amort_months

    def cut_active(cut: CostCut, m: int) -> bool:
        if m < cut.start_month:
            return False
        if cut.duration_months and cut.duration_months > 0:
            return m <= (cut.start_month + cut.duration_months - 1)
        return True

    for i, m in enumerate(months):
        # Revenue base
        if m == 1:
            rev = revenue_base
        else:
            rev = receita_bruta[i - 1] * (1 + revenue_growth_m)

        # Seasonality
        rev *= seasonal_multiplier(seasonality_on, seasonality_12, m)

        # Costs base
        if m == 1:
            fc = fixed_cost_base
            vc = var_cost_base
        else:
            fc = fixos[i - 1] * (1 + cost_growth_m)
            vc = variaveis[i - 1] * (1 + cost_growth_m)

        # Apply cuts
        fc_mult, vc_mult = 1.0, 1.0
        for cut in cuts:
            if cut_active(cut, m):
                fc_mult *= (1 - cut.fixed_reduction_pct)
                vc_mult *= (1 - cut.variable_reduction_pct)
        fc *= fc_mult
        vc *= vc_mult

        # Events
        ev_cost = 0.0
        ev_rev = 0.0

        for h in hirings:
            if m >= h.start_month:
                rel = m - h.start_month + 1
                ev_cost += apply_ramp(h.monthly_cost, rel, h.ramp_months)
                if h.revenue_impact:
                    ev_rev += apply_ramp(h.revenue_impact, rel, h.ramp_months)

        aporte_mes = 0.0
        for inv in investments:
            if inv.month == m:
                if inv.kind == "OPEX":
                    ev_cost += inv.value
                elif inv.kind == "APORTE":
                    aporte_mes += inv.value

        ev_cost += capex_monthly_add[i]

        receita_bruta[i] = rev + ev_rev
        impostos[i] = receita_bruta[i] * tax_pct
        cogs[i] = receita_bruta[i] * cogs_pct
        receita_liq[i] = receita_bruta[i] - impostos[i] - cogs[i]

        fixos[i] = fc
        variaveis[i] = vc
        custos_eventos[i] = ev_cost

        total_cost = fc + vc + ev_cost
        resultado[i] = receita_liq[i] - total_cost
        margem[i] = (resultado[i] / receita_bruta[i] * 100) if receita_bruta[i] > 0 else 0.0
        burn[i] = -resultado[i] if resultado[i] < 0 else 0.0

        caixa[i + 1] = caixa[i] + resultado[i] + aporte_mes

    df = pd.DataFrame({
        "Mês": months,
        "Receita Bruta": receita_bruta,
        "Impostos": impostos,
        "COGS": cogs,
        "Receita Líquida": receita_liq,
        "Custos Fixos": fixos,
        "Custos Variáveis": variaveis,
        "Custos Eventos": custos_eventos,
        "Custos Totais": fixos + variaveis + custos_eventos,
        "Resultado": resultado,
        "Margem (%)": margem,
        "Burn": burn,
        "Caixa": caixa[1:],
    })

    breakeven = int(df.loc[df["Resultado"] >= 0, "Mês"].min()) if (df["Resultado"] >= 0).any() else None
    caixa_neg = int(df.loc[df["Caixa"] < 0, "Mês"].min()) if (df["Caixa"] < 0).any() else None

    burn_ult = df["Burn"].tail(3).mean()
    runway = float(df["Caixa"].iloc[-1] / burn_ult) if burn_ult and burn_ult > 0 else None

    return df, breakeven, caixa_neg, runway

# =========================
# State init
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
            "seasonality_12": [1.0]*12,
            "three_scenarios": True,
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
        st.session_state.results = None
    if "last_base_cash_final" not in st.session_state:
        st.session_state.last_base_cash_final = None  # para delta em Ações

init_state()
d = st.session_state.drivers

# =========================
# UX: readiness & progress
# =========================
def readiness() -> Dict[str, bool]:
    state_ok = (safe_float(d["revenue_base"]) > 0) and (safe_float(d["fixed_cost_base"]) >= 0) and (safe_float(d["var_cost_base"]) >= 0)
    hypo_ok = True
    return {
        "Estado atual": state_ok,
        "Hipóteses": hypo_ok,
        "Cenários": st.session_state.results is not None,
    }

ready = readiness()
progress = sum(1 for v in ready.values() if v) / len(ready)

# =========================
# Sidebar: navigation + summary
# =========================
st.sidebar.markdown("## FinanceOps")
st.sidebar.markdown('<span class="soft-pill">MVP público • UX Copilot</span>', unsafe_allow_html=True)
st.sidebar.write("")

page = st.sidebar.radio(
    "Onde você quer mexer agora?",
    ["Estado atual", "Hipóteses", "Cenários", "Ações (eventos)", "Relatório"],
    index=0
)

st.sidebar.markdown("### Progresso")
st.sidebar.progress(progress)
st.sidebar.write(f'<span class="small">Prontidão: {int(progress*100)}%</span>', unsafe_allow_html=True)

with st.sidebar.expander("Checklist (rápido)", expanded=False):
    for k, v in ready.items():
        st.write(f"{'✅' if v else '⏳'} {k}")

st.sidebar.divider()

with st.sidebar.expander("Resumo do cenário (sempre visível)", expanded=True):
    st.write(f"Receita: **{brl(d['revenue_base'])}**")
    st.write(f"Fixos: **{brl(d['fixed_cost_base'])}**")
    st.write(f"Variáveis: **{brl(d['var_cost_base'])}**")
    st.write(f"Caixa: **{brl(d['cash_initial'])}**")
    st.write(f"Cresc. Receita: **{pct(d['revenue_growth_pct'])}/m**")
    st.write(f"Cresc. Custos: **{pct(d['cost_growth_pct'])}/m**")
    st.write(f"Horizonte: **{d['horizon']} meses**")
    st.write(f"Sazonalidade: **{'Ligada' if d['seasonality_on'] else 'Desligada'}**")

# =========================
# Header
# =========================
st.title("📊 FinanceOps — Copiloto financeiro")
st.caption("Menos planilha. Mais decisão. (E menos autoengano no cenário pessimista.)")

# =========================
# Page: Estado atual
# =========================
if page == "Estado atual":
    st.subheader("Estado atual — resultado em 60 segundos")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.write("Preencha o mínimo. O app te devolve diagnóstico imediato.")
        c1, c2 = st.columns(2)

        with c1:
            d["revenue_base"] = st.number_input(
                "Receita do mês (R$)",
                min_value=0.0, step=1000.0, value=float(d["revenue_base"]),
                help="Último mês real ou média 3 meses."
            )
            with st.expander("Saiba mais: Receita do mês"):
                st.write("Use o valor do mês mais recente. Se houver oscilação, use a média dos últimos 3 meses. "
                         "O objetivo aqui é criar uma base razoável, não acertar o futuro com precisão mística.")

        with c2:
            d["cash_initial"] = st.number_input(
                "Caixa disponível (R$)",
                min_value=0.0, step=1000.0, value=float(d["cash_initial"]),
                help="Saldo de conta + reserva de liquidez."
            )
            with st.expander("Saiba mais: Caixa"):
                st.write("É o dinheiro que sustenta o negócio enquanto você ajusta receita e custos. "
                         "Caixa é oxigênio. Sem ele, o resto vira filosofia.")

        c3, c4 = st.columns(2)
        with c3:
            d["fixed_cost_base"] = st.number_input(
                "Custos fixos (R$)",
                min_value=0.0, step=500.0, value=float(d["fixed_cost_base"]),
                help="Acontecem mesmo com receita baixa."
            )
            with st.expander("Saiba mais: Custos fixos"):
                st.write("Tudo que você pagaria mesmo vendendo zero: salários fixos, aluguel, ferramentas base, contabilidade. "
                         "É a parte rígida do seu modelo.")

        with c4:
            d["var_cost_base"] = st.number_input(
                "Custos variáveis (R$)",
                min_value=0.0, step=500.0, value=float(d["var_cost_base"]),
                help="Crescem quando você vende."
            )
            with st.expander("Saiba mais: Custos variáveis"):
                st.write("Taxas, comissões, mídia proporcional, insumos, fretes. "
                         "Se você é serviço puro, pode ser baixo (mas nunca é zero de verdade).")

        st.markdown("</div>", unsafe_allow_html=True)

    # Instant diagnosis
    receita = safe_float(d["revenue_base"])
    custos_totais = safe_float(d["fixed_cost_base"]) + safe_float(d["var_cost_base"])
    resultado = receita - custos_totais
    margem = (resultado / receita * 100) if receita > 0 else 0.0
    burn = -resultado if resultado < 0 else 0.0

    with right:
        st.markdown("### Diagnóstico instantâneo")
        r1, r2, r3, r4 = st.columns(4)
        with st.container():
            st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
            r1.metric("Resultado", brl(resultado))
            r2.metric("Margem", f"{margem:.1f}%")
            r3.metric("Burn", brl(burn))
            r4.metric("Custos Totais", brl(custos_totais))
            st.markdown("</div>", unsafe_allow_html=True)

        # Product-style insights
        st.markdown("### Insights")
        if receita > 0:
            rigidez = safe_float(d["fixed_cost_base"]) / receita
            if rigidez >= 0.8:
                st.error("Rigidez alta: **fixos ≥ 80% da receita**. Você está andando num trampolim… sem mola.")
            elif rigidez >= 0.6:
                st.warning("Rigidez moderada: **fixos ≥ 60% da receita**. Mês ruim vira estresse rápido.")
            else:
                st.success("Boa flexibilidade: fixos abaixo de 60% da receita.")
        if resultado < 0 and burn > 0:
            runway_simples = safe_float(d["cash_initial"]) / burn
            st.warning(f"Runway simples (se nada mudar): ~**{runway_simples:.1f} meses**.")
        elif resultado >= 0:
            st.success("Você está positivo no mês atual. Agora o jogo é: **consegue sustentar isso no tempo?**")

        st.caption("Próximo passo: vá em **Hipóteses** → depois **Cenários**.")

# =========================
# Page: Hipóteses + Sazonalidade
# =========================
elif page == "Hipóteses":
    st.subheader("Hipóteses — ajuste o motor sem sofrimento")

    # Presets (1 click)
    st.markdown("### Presets (1 clique)")
    p1, p2, p3 = st.columns(3)

    def set_preset(name: str):
        if name == "Conservador":
            d["revenue_growth_pct"] = 2.0
            d["cost_growth_pct"] = 2.0
        elif name == "Realista":
            d["revenue_growth_pct"] = 5.0
            d["cost_growth_pct"] = 2.0
        elif name == "Agressivo":
            d["revenue_growth_pct"] = 10.0
            d["cost_growth_pct"] = 3.0
        st.session_state.results = None

    if p1.button("Conservador"):
        set_preset("Conservador")
    if p2.button("Realista"):
        set_preset("Realista")
    if p3.button("Agressivo"):
        set_preset("Agressivo")

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        d["horizon"] = st.selectbox("Horizonte (meses)", [12, 24, 36], index=[12,24,36].index(d["horizon"]))
        with st.expander("Saiba mais: Horizonte"):
            st.write("12 meses = curto prazo; 24 = planejamento; 36 = estratégia. Quanto maior, mais importante usar cenários.")

    with c2:
        d["revenue_growth_pct"] = st.slider(
            "Crescimento mensal da receita (%)",
            -50.0, 80.0, float(d["revenue_growth_pct"]), 0.5,
            help="Ex.: 5% a.m. = multiplica por 1,05 todo mês."
        )
        with st.expander("Saiba mais: Crescimento da receita"):
            st.write("Se você não tem histórico, use um valor conservador e confie mais no cenário pessimista. "
                     "Crescimento é hipótese, não decreto.")

    with c3:
        d["cost_growth_pct"] = st.slider(
            "Crescimento mensal dos custos (%)",
            -20.0, 50.0, float(d["cost_growth_pct"]), 0.5,
            help="Inflação + expansão."
        )
        with st.expander("Saiba mais: Crescimento dos custos"):
            st.write("Se você vai contratar, prefira modelar isso em **Ações**. É mais realista do que inflar %.")

    st.markdown("### Ajustes opcionais (mais realismo)")
    o1, o2, o3 = st.columns(3)
    with o1:
        d["tax_pct"] = st.slider("Imposto médio sobre receita (%)", 0.0, 25.0, float(d["tax_pct"]), 0.5)
        with st.expander("Saiba mais: Imposto"):
            st.write("MVP usa alíquota média. Se você não souber, deixe 0% e só use depois para refinamento.")

    with o2:
        d["cogs_pct"] = st.slider("COGS / custo direto (%)", 0.0, 80.0, float(d["cogs_pct"]), 0.5)
        with st.expander("Saiba mais: COGS"):
            st.write("Se você tem custo direto por venda/entrega (produto/insumo/frete), isso melhora muito a receita líquida.")

    with o3:
        d["three_scenarios"] = st.toggle("Comparar 3 cenários (recomendado)", value=bool(d["three_scenarios"]))
        with st.expander("Saiba mais: 3 cenários"):
            st.write("Base/Otimista/Pessimista automaticamente. É seu cinto de segurança contra decisão otimista demais.")

    st.divider()

    # Sazonalidade (toggle + presets + 12 multipliers)
    st.markdown("### Sazonalidade (opcional)")
    d["seasonality_on"] = st.toggle(
        "Meu negócio tem sazonalidade (meses melhores/piores)",
        value=bool(d["seasonality_on"]),
        help="Liga multiplicadores mensais (jan..dez). 1.10 = +10%, 0.90 = -10%."
    )

    if d["seasonality_on"]:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.write("Defina o “perfil do ano” (rápido) e ajuste se quiser.")
        s1, s2, s3, s4 = st.columns(4)

        def set_season_preset(kind: str):
            base = [1.0]*12
            if kind == "Alta em Nov/Dez":
                base[10] = 1.15
                base[11] = 1.25
            elif kind == "Alta no Verão":
                base[0] = 1.10
                base[1] = 1.10
                base[11] = 1.10
            elif kind == "Baixa no meio do ano":
                base[5] = 0.92
                base[6] = 0.90
                base[7] = 0.93
            d["seasonality_12"] = base
            st.session_state.results = None

        if s1.button("Preset: Ano plano"):
            set_season_preset("Plano")
        if s2.button("Preset: Alta em Nov/Dez"):
            set_season_preset("Alta em Nov/Dez")
        if s3.button("Preset: Alta no Verão"):
            set_season_preset("Alta no Verão")
        if s4.button("Preset: Baixa no meio do ano"):
            set_season_preset("Baixa no meio do ano")

        months = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
        season = d["seasonality_12"]
        grid = st.columns(6)
        for i, mname in enumerate(months):
            with grid[i % 6]:
                season[i] = st.number_input(
                    f"{mname}",
                    min_value=0.70, max_value=1.30,
                    value=float(season[i]), step=0.01,
                    help="Multiplicador do mês (1.00 = normal)."
                )
        d["seasonality_12"] = season

        with st.expander("Saiba mais: como usar sazonalidade sem se ferrar"):
            st.write("Se você não tem histórico, use sazonalidade leve (±10%). "
                     "Sazonalidade forte sem dado vira ficção bem escrita.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Page: Cenários
# =========================
elif page == "Cenários":
    st.subheader("Cenários — risco e amplitude (board-ready)")

    if safe_float(d["revenue_base"]) <= 0:
        st.warning("Preencha **Estado atual** primeiro.")
    else:
        left, right = st.columns([1, 2])

        with left:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.write("Rodar cenários = enxergar onde o caixa quebra e onde o break-even aparece.")
            if d["three_scenarios"]:
                st.markdown("**Ajuste rápido (desvios do Base)**")
                d["opt_revenue_delta"] = st.slider("Otimista: +pp cresc. receita", 0.0, 20.0, float(d["opt_revenue_delta"]), 0.5)
                d["pess_revenue_delta"] = st.slider("Pessimista: +pp cresc. receita", -20.0, 0.0, float(d["pess_revenue_delta"]), 0.5)
                d["opt_cost_delta"] = st.slider("Otimista: +pp cresc. custos", -10.0, 0.0, float(d["opt_cost_delta"]), 0.5)
                d["pess_cost_delta"] = st.slider("Pessimista: +pp cresc. custos", 0.0, 10.0, float(d["pess_cost_delta"]), 0.5)

            run = st.button("🚀 Rodar cenários", type="primary")
            st.markdown("</div>", unsafe_allow_html=True)

        def run_one(label: str, rg_pct: float, cg_pct: float) -> Dict[str, Any]:
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

        if run or st.session_state.results is None:
            base_rg = float(d["revenue_growth_pct"])
            base_cg = float(d["cost_growth_pct"])

            scenarios = [run_one("Base", base_rg, base_cg)]
            if d["three_scenarios"]:
                scenarios.append(run_one("Otimista", base_rg + float(d["opt_revenue_delta"]), base_cg + float(d["opt_cost_delta"])))
                scenarios.append(run_one("Pessimista", base_rg + float(d["pess_revenue_delta"]), base_cg + float(d["pess_cost_delta"])))

            st.session_state.results = scenarios
            # salva caixa final do Base para deltas em ações
            base_df = next(x for x in scenarios if x["label"] == "Base")["df"]
            st.session_state.last_base_cash_final = float(base_df["Caixa"].iloc[-1])

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

            st.markdown("### Leitura rápida")
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

            # Copilot insights
            st.markdown("### Copilot (o que isso significa)")
            base = next(x for x in scenarios if x["label"] == "Base")
            pess = next((x for x in scenarios if x["label"] == "Pessimista"), None)

            if pess and pess["cash_neg"]:
                st.error(f"No **Pessimista**, seu caixa quebra no mês **{pess['cash_neg']}**. "
                         f"Tradução: você precisa de proteção (corte/aporte) ou de crescimento mais forte.")
            elif base["cash_neg"]:
                st.warning(f"No **Base**, seu caixa quebra no mês **{base['cash_neg']}**. "
                           f"Próximo passo: testar **Ações**.")
            else:
                st.success("No cenário base, não há mês de caixa negativo no horizonte. "
                           "Agora use **Ações** para acelerar (contratar/investir) com segurança.")

# =========================
# Page: Ações (eventos) — impact now
# =========================
elif page == "Ações (eventos)":
    st.subheader("Ações — decisões reais, impacto imediato")

    if safe_float(d["revenue_base"]) <= 0:
        st.warning("Preencha **Estado atual** primeiro.")
    else:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.write("Aqui você para de mexer em porcentagem e começa a mexer em estratégia.")
        st.markdown("</div>", unsafe_allow_html=True)

        tabs = st.tabs(["Contratar", "Investir/Aporte", "Cortar custos", "Ações registradas"])

        # --- CONTRATAR
        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            start = c1.number_input("Mês início", 1, d["horizon"], 3, 1)
            cost = c2.number_input("Custo mensal total (R$)", 0.0, 1e12, 8000.0, 500.0)
            ramp = c3.number_input("Ramp (meses)", 0, 12, 1, 1)
            rev_imp = c4.number_input("Impacto em receita (R$/mês)", 0.0, 1e12, 0.0, 500.0)
            with st.expander("Saiba mais: contratação"):
                st.write("Use custo total (salário + encargos). Ramp suaviza entrada. "
                         "Impacto em receita é opcional e deve ser conservador.")
            if st.button("Salvar contratação", type="primary"):
                st.session_state.hirings.append(Hiring(int(start), float(cost), int(ramp), float(rev_imp)))
                st.session_state.results = None
                st.success("Contratação adicionada.")

        # --- INVESTIR/APORTE
        with tabs[1]:
            c1, c2, c3, c4 = st.columns(4)
            m = c1.number_input("Mês", 1, d["horizon"], 2, 1)
            v = c2.number_input("Valor (R$)", 0.0, 1e12, 10000.0, 1000.0)
            kind = c3.selectbox("Tipo", ["OPEX", "CAPEX", "APORTE"])
            amort = c4.number_input("Amortização (meses)", 0, 36, 0, 1)
            with st.expander("Saiba mais: OPEX/CAPEX/APORTE"):
                st.write("**OPEX** vira custo no mês. **CAPEX** pode ser amortizado em N meses. "
                         "**APORTE** entra no caixa (não afeta resultado, mas salva o jogo).")
            if st.button("Salvar investimento/aporte", type="primary"):
                st.session_state.investments.append(Investment(int(m), float(v), kind, int(amort)))
                st.session_state.results = None
                st.success("Investimento/aporte adicionado.")

        # --- CORTAR CUSTOS
        with tabs[2]:
            c1, c2, c3, c4 = st.columns(4)
            sm = c1.number_input("Mês início", 1, d["horizon"], 4, 1)
            fr = c2.slider("Redução fixos (%)", 0.0, 50.0, 0.0, 0.5)
            vr = c3.slider("Redução variáveis (%)", 0.0, 50.0, 0.0, 0.5)
            dur = c4.number_input("Duração (meses) — 0=até fim", 0, 36, 0, 1)
            with st.expander("Saiba mais: corte"):
                st.write("Corte em fixos dá mais efeito no caixa, mas pode doer. "
                         "Use duração se for algo temporário (ex.: renegociação por 6 meses).")
            if st.button("Salvar corte", type="primary"):
                st.session_state.cuts.append(CostCut(int(sm), fr/100, vr/100, int(dur)))
                st.session_state.results = None
                st.success("Corte adicionado.")

        # --- REGISTRADAS
        with tabs[3]:
            colL, colR = st.columns([1, 1])
            with colL:
                st.caption("Contratações")
                st.dataframe(pd.DataFrame([asdict(x) for x in st.session_state.hirings]) if st.session_state.hirings else pd.DataFrame(),
                             use_container_width=True)
                st.caption("Investimentos/Aportes")
                st.dataframe(pd.DataFrame([asdict(x) for x in st.session_state.investments]) if st.session_state.investments else pd.DataFrame(),
                             use_container_width=True)
            with colR:
                st.caption("Cortes")
                st.dataframe(pd.DataFrame([asdict(x) for x in st.session_state.cuts]) if st.session_state.cuts else pd.DataFrame(),
                             use_container_width=True)

                st.divider()
                if st.button("🧹 Limpar TODAS as ações"):
                    st.session_state.hirings = []
                    st.session_state.investments = []
                    st.session_state.cuts = []
                    st.session_state.results = None
                    st.success("Ações limpas.")

        st.divider()

        # Impact now: run Base only and show delta vs last_base_cash_final
        run_now = st.button("⚡ Recalcular (Base) e ver impacto agora", type="primary")
        if run_now:
            df_new, be, neg, runway = simulate_financeops(
                horizon_months=int(d["horizon"]),
                revenue_base=float(d["revenue_base"]),
                revenue_growth_m=float(d["revenue_growth_pct"]) / 100,
                fixed_cost_base=float(d["fixed_cost_base"]),
                var_cost_base=float(d["var_cost_base"]),
                cost_growth_m=float(d["cost_growth_pct"]) / 100,
                cash_initial=float(d["cash_initial"]),
                tax_pct=float(d["tax_pct"]) / 100,
                cogs_pct=float(d["cogs_pct"]) / 100,
                seasonality_on=bool(d["seasonality_on"]),
                seasonality_12=d["seasonality_12"],
                hirings=st.session_state.hirings,
                investments=st.session_state.investments,
                cuts=st.session_state.cuts,
            )
            cash_final_new = float(df_new["Caixa"].iloc[-1])
            last = st.session_state.last_base_cash_final
            delta = (cash_final_new - last) if last is not None else None

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Caixa final (novo)", brl(cash_final_new), (brl(delta) if delta is not None else None))
            k2.metric("Break-even", str(be) if be else "—")
            k3.metric("Caixa negativo", str(neg) if neg else "—")
            k4.metric("Runway", f"{runway:.1f}m" if runway is not None else "—")

            st.caption("Quer comparar Base/Otimista/Pessimista com as ações? Vá em **Cenários** e rode novamente.")
            st.line_chart(df_new.set_index("Mês")[["Caixa", "Resultado"]])

# =========================
# Page: Relatório
# =========================
else:
    st.subheader("Relatório — pronto pra copiar/colar e exportar")

    if st.session_state.results is None:
        st.warning("Rode **Cenários** primeiro.")
    else:
        scenarios = st.session_state.results
        base = next(x for x in scenarios if x["label"] == "Base")
        df_base = base["df"]

        be = base["breakeven"]
        neg = base["cash_neg"]
        runway = base["runway"]

        bullets = [
            f"• Receita atual: {brl(d['revenue_base'])}",
            f"• Custos fixos: {brl(d['fixed_cost_base'])} | Custos variáveis: {brl(d['var_cost_base'])}",
            f"• Caixa inicial: {brl(d['cash_initial'])}",
            f"• Cresc. receita (base): {pct(d['revenue_growth_pct'])}/m | Cresc. custos (base): {pct(d['cost_growth_pct'])}/m",
            f"• Horizonte: {d['horizon']} meses",
            f"• Caixa final (base): {brl(df_base['Caixa'].iloc[-1])}",
            f"• Resultado final (base): {brl(df_base['Resultado'].iloc[-1])}",
        ]
        bullets.append(f"• Break-even: mês {be}." if be else "• Break-even: não ocorre no horizonte.")
        bullets.append(f"• Risco crítico: caixa negativo no mês {neg}." if neg else "• Caixa: sem mês negativo no horizonte.")
        if runway is not None:
            bullets.append(f"• Runway estimada (burn médio 3m): {runway:.1f} meses.")

        report_text = "\n".join(bullets)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.text_area("Texto executivo (copiar e colar)", report_text, height=220)
            csv = df_base.to_csv(index=False).encode("utf-8")
            st.download_button("Baixar CSV (Base)", data=csv, file_name="financeops_base.csv", mime="text/csv")
        with c2:
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
