import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

st.set_page_config(page_title="FinanceOps MVP - Wizard", layout="wide")

# =========================
# Helpers
# =========================
def brl(x: float) -> str:
    s = f"{x:,.2f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")

def pct(x: float) -> str:
    return f"{x:.2f}%"

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
    revenue_impact: float = 0.0  # optional

@dataclass
class Investment:
    month: int
    value: float
    kind: str  # "OPEX" | "CAPEX" | "APORTE"
    amort_months: int = 0  # only for CAPEX

@dataclass
class CostCut:
    start_month: int
    fixed_reduction_pct: float = 0.0
    variable_reduction_pct: float = 0.0
    duration_months: int = 0  # 0 = indefinido


# =========================
# Simulation engine
# =========================
def simulate_financeops(
    horizon_months: int,
    # drivers
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
    # events
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

    # precompute CAPEX amortization schedules
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

        # apply cost cuts (if any)
        fc_mult = 1.0
        vc_mult = 1.0
        for cut in cuts:
            if is_cut_active(cut, m):
                fc_mult *= (1 - cut.fixed_reduction_pct)
                vc_mult *= (1 - cut.variable_reduction_pct)

        fc *= fc_mult
        vc *= vc_mult

        # events: hirings add costs and can add revenue impact
        ev_cost = 0.0
        ev_rev = 0.0
        for h in hirings:
            if m >= h.start_month:
                rel = m - h.start_month + 1
                ev_cost += apply_ramp(h.monthly_cost, rel, h.ramp_months)
                if h.revenue_impact and h.revenue_impact != 0:
                    ev_rev += apply_ramp(h.revenue_impact, rel, h.ramp_months)

        # investments
        aporte_mes = 0.0
        for inv in investments:
            if inv.month == m:
                if inv.kind == "OPEX":
                    ev_cost += inv.value
                elif inv.kind == "APORTE":
                    aporte_mes += inv.value
                # CAPEX is handled via amortization schedule

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

        if receita_bruta[i] > 0:
            margem_pct[i] = (resultado[i] / receita_bruta[i]) * 100
        else:
            margem_pct[i] = 0.0

        burn[i] = -resultado[i] if resultado[i] < 0 else 0.0

        cash[i + 1] = cash[i] + resultado[i] + aporte_mes

    cash_series = cash[1:]

    df = pd.DataFrame({
        "Mês": months,
        "Receita Bruta": receita_bruta,
        "Impostos": impostos,
        "COGS": cogs,
        "Receita Líquida (após imposto/COGS)": receita_liq,
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


def scenario_pack(base_value, delta_opt, delta_pess, floor=None):
    opt = base_value + delta_opt
    pess = base_value + delta_pess
    if floor is not None:
        opt = max(opt, floor)
        pess = max(pess, floor)
    return opt, pess


# =========================
# Session state defaults
# =========================
if "step" not in st.session_state:
    st.session_state.step = 1

if "drivers" not in st.session_state:
    st.session_state.drivers = {
        "horizon": 24,
        "revenue_base": 50000.0,
        "revenue_growth_pct": 5.0,
        "fixed_cost_base": 20000.0,
        "var_cost_base": 10000.0,
        "cost_growth_pct": 2.0,
        "cash_initial": 100000.0,
        "tax_pct": 0.0,
        "cogs_pct": 0.0,
        "seasonality_on": False,
        "seasonality_12": [1.0]*12,
        "three_scenarios": True,
        "opt_revenue_delta": 3.0,   # +3pp no crescimento receita
        "pess_revenue_delta": -3.0, # -3pp no crescimento receita
        "opt_cost_delta": -1.0,     # -1pp crescimento custos
        "pess_cost_delta": 1.0,     # +1pp crescimento custos
    }

if "hirings" not in st.session_state:
    st.session_state.hirings = []  # list[Hiring]
if "investments" not in st.session_state:
    st.session_state.investments = []  # list[Investment]
if "cuts" not in st.session_state:
    st.session_state.cuts = []  # list[CostCut]


# =========================
# UI: Wizard header
# =========================
st.title("FinanceOps — Wizard (MVP)")

step_titles = {
    1: "1) Receita",
    2: "2) Custos",
    3: "3) Caixa e Prazos",
    4: "4) Eventos (Opcional)",
    5: "5) Resultados",
}

cols = st.columns([2, 1, 1, 1, 1, 2])
cols[0].markdown(f"### {step_titles[st.session_state.step]}")

with cols[-1]:
    st.caption("Dica: este wizard foi desenhado pra reduzir atrito. Você preenche drivers, o motor calcula o resto.")

nav1, nav2, nav3 = st.columns([1, 1, 4])
with nav1:
    if st.button("⬅️ Voltar", disabled=(st.session_state.step == 1)):
        st.session_state.step -= 1
        st.rerun()
with nav2:
    if st.button("Avançar ➡️", disabled=(st.session_state.step == 5)):
        st.session_state.step += 1
        st.rerun()

st.divider()

d = st.session_state.drivers

# =========================
# Step 1: Receita
# =========================
if st.session_state.step == 1:
    st.subheader("Receita — drivers essenciais")

    st.info(
        "Preencha o mínimo para o simulador projetar sua trajetória. "
        "Você não precisa de planilha: só do número atual e da sua melhor hipótese de crescimento."
    )

    c1, c2 = st.columns(2)
    with c1:
        d["revenue_base"] = st.number_input(
            "Receita mensal atual (R$)",
            min_value=0.0,
            value=float(d["revenue_base"]),
            step=1000.0,
            help="Quanto você faturou (ou estima faturar) no mês mais recente. Use receita recorrente + pontual, se fizer sentido."
        )
        st.caption("📌 Use o último mês real ou uma média dos últimos 3 meses se tiver oscilação.")

    with c2:
        d["revenue_growth_pct"] = st.slider(
            "Crescimento mensal da receita (%)",
            min_value=-50.0,
            max_value=80.0,
            value=float(d["revenue_growth_pct"]),
            step=0.5,
            help="Crescimento médio esperado mês a mês. Ex.: 5% a.m. = multiplicar por 1,05 todo mês."
        )
        st.caption("📌 Se você não sabe, escolha um valor conservador e use 3 cenários na etapa de Resultados.")

    st.markdown("### Sazonalidade (opcional)")
    d["seasonality_on"] = st.toggle(
        "Meu negócio tem sazonalidade (meses melhores e piores)",
        value=bool(d["seasonality_on"]),
        help="Se ligado, você define 12 multiplicadores (jan..dez). Ex.: 1,10 = 10% acima da média; 0,90 = 10% abaixo."
    )

    if d["seasonality_on"]:
        st.warning("Sazonalidade é opcional. Se você não tem histórico, deixe desligado.")
        months = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
        grid = st.columns(6)
        season = d["seasonality_12"]
        for i, mname in enumerate(months):
            with grid[i % 6]:
                season[i] = st.number_input(
                    f"{mname}",
                    min_value=0.5,
                    max_value=1.5,
                    value=float(season[i]),
                    step=0.01,
                    help="Multiplicador do mês. 1,00 = normal. 1,20 = +20%. 0,85 = -15%."
                )
        d["seasonality_12"] = season

# =========================
# Step 2: Custos
# =========================
elif st.session_state.step == 2:
    st.subheader("Custos — drivers essenciais")

    st.info(
        "Aqui você separa o que é custo fixo (sobrevive sem vender) e variável (cresce quando você vende). "
        "Isso melhora muito a qualidade do forecast."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        d["fixed_cost_base"] = st.number_input(
            "Custos fixos atuais (R$)",
            min_value=0.0,
            value=float(d["fixed_cost_base"]),
            step=500.0,
            help="Custos que acontecem mesmo com receita baixa: salários fixos, aluguel, software base, contabilidade, etc."
        )
        st.caption("📌 Se tiver dúvida, some tudo que você pagaria mesmo se vendesse zero.")

    with c2:
        d["var_cost_base"] = st.number_input(
            "Custos variáveis atuais (R$)",
            min_value=0.0,
            value=float(d["var_cost_base"]),
            step=500.0,
            help="Custos que escalam com o faturamento: comissões, taxas, mídia proporcional, insumos, fretes, etc."
        )
        st.caption("📌 Se você é serviço, variável pode ser quase zero (ou comissões).")

    with c3:
        d["cost_growth_pct"] = st.slider(
            "Crescimento mensal dos custos (%)",
            min_value=-20.0,
            max_value=50.0,
            value=float(d["cost_growth_pct"]),
            step=0.5,
            help="Como você espera que os custos cresçam mês a mês. Ex.: reajustes, expansão do time, inflação, etc."
        )
        st.caption("📌 Se você pretende aumentar time, dá pra modelar na etapa Eventos (mais realista).")

    st.markdown("### Ajustes (opcionais)")
    o1, o2 = st.columns(2)
    with o1:
        d["tax_pct"] = st.slider(
            "Imposto sobre receita (%)",
            min_value=0.0,
            max_value=25.0,
            value=float(d["tax_pct"]),
            step=0.5,
            help="Se você quiser aproximar receita líquida, informe uma alíquota média (Simples/ISS etc.). Se não souber, deixe 0%."
        )
        st.caption("📌 MVP: é alíquota média, sem contabilidade avançada.")
    with o2:
        d["cogs_pct"] = st.slider(
            "COGS / custo direto (%)",
            min_value=0.0,
            max_value=80.0,
            value=float(d["cogs_pct"]),
            step=0.5,
            help="Se você vende produto/tem custo direto por entrega, informe % médio. Se for serviço puro, pode deixar 0%."
        )
        st.caption("📌 Isso ajuda a separar 'custo do que eu vendo' do 'custo de operar'.")

# =========================
# Step 3: Caixa e prazos
# =========================
elif st.session_state.step == 3:
    st.subheader("Caixa e prazos (DSO/DPO)")

    st.info(
        "O caixa define quanto tempo você aguenta. DSO/DPO entram como contexto para evoluirmos depois "
        "para um fluxo de caixa mais realista (com atraso de recebimento/pagamento)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        d["cash_initial"] = st.number_input(
            "Caixa atual (R$)",
            min_value=0.0,
            value=float(d["cash_initial"]),
            step=1000.0,
            help="Quanto você tem disponível (conta, reserva, aplicações de liquidez). Isso alimenta runway e risco de caixa negativo."
        )
        st.caption("📌 Se o caixa está apertado, 3 cenários te protegem de autoengano.")

    with c2:
        dso = st.number_input(
            "Prazo médio de recebimento (DSO) — dias",
            min_value=0,
            value=30,
            step=1,
            help="Em média, quantos dias você demora para receber após faturar. Ex.: cartão pode ser 30; B2B pode ser 45/60."
        )
        st.caption("📌 MVP: ainda não desloca receita no tempo (vamos evoluir depois).")

    with c3:
        dpo = st.number_input(
            "Prazo médio de pagamento (DPO) — dias",
            min_value=0,
            value=30,
            step=1,
            help="Em média, quantos dias você leva para pagar fornecedores. Um DPO maior melhora o caixa no curto prazo."
        )
        st.caption("📌 MVP: vira base para evolução do fluxo de caixa real.")

# =========================
# Step 4: Eventos
# =========================
elif st.session_state.step == 4:
    st.subheader("Eventos (opcional) — decisões que mudam o futuro")

    st.info(
        "Eventos tiram seu forecast da 'reta' e colocam decisões reais: contratar, investir, cortar, receber aporte. "
        "Se você não tiver nada planejado, pode pular."
    )

    d["horizon"] = st.selectbox("Horizonte da simulação (meses)", [12, 24, 36], index=[12,24,36].index(d["horizon"]))

    st.markdown("## Contratações")
    with st.expander("Adicionar contratação", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        start = c1.number_input("Mês de início", min_value=1, max_value=d["horizon"], value=3, step=1,
                                help="Mês em que o custo começa a aparecer.")
        cost = c2.number_input("Custo mensal total (R$)", min_value=0.0, value=8000.0, step=500.0,
                               help="Salário + encargos + benefícios (aprox.).")
        ramp = c3.number_input("Ramp (meses)", min_value=0, max_value=12, value=1, step=1,
                               help="Se 2, entra 50% no 1º mês e 100% no 2º (simplificado).")
        rev_imp = c4.number_input("Impacto em receita (R$/mês)", min_value=0.0, value=0.0, step=500.0,
                                  help="Opcional: ex. vendedor gera receita após entrar. MVP: soma direto.")
        if st.button("➕ Inserir contratação"):
            st.session_state.hirings.append(Hiring(int(start), float(cost), int(ramp), float(rev_imp)))

    if st.session_state.hirings:
        dfh = pd.DataFrame([asdict(x) for x in st.session_state.hirings])
        st.dataframe(dfh, use_container_width=True)
        if st.button("🧹 Limpar contratações"):
            st.session_state.hirings = []

    st.markdown("## Investimentos / Aportes")
    with st.expander("Adicionar investimento/aporte", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        imonth = c1.number_input("Mês", min_value=1, max_value=d["horizon"], value=2, step=1,
                                 help="Mês em que ocorre o evento.")
        val = c2.number_input("Valor (R$)", min_value=0.0, value=10000.0, step=1000.0,
                              help="Valor do investimento (ou aporte).")
        kind = c3.selectbox("Tipo", ["OPEX", "CAPEX", "APORTE"],
                            help="OPEX vira custo no mês. CAPEX pode amortizar. APORTE entra no caixa.")
        amort = c4.number_input("Amortização (meses)", min_value=0, max_value=36, value=0, step=1,
                                help="Só para CAPEX: distribui o custo em N meses. Se 0, ignora.")
        if st.button("➕ Inserir investimento/aporte"):
            st.session_state.investments.append(Investment(int(imonth), float(val), kind, int(amort)))

    if st.session_state.investments:
        dfi = pd.DataFrame([asdict(x) for x in st.session_state.investments])
        st.dataframe(dfi, use_container_width=True)
        if st.button("🧹 Limpar investimentos/aportes"):
            st.session_state.investments = []

    st.markdown("## Cortes de custo")
    with st.expander("Adicionar corte", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        cstart = c1.number_input("Mês início", min_value=1, max_value=d["horizon"], value=4, step=1,
                                 help="Quando o corte passa a valer.")
        fr = c2.slider("Redução fixos (%)", 0.0, 50.0, 0.0, 0.5,
                       help="Ex.: 10% reduz custos fixos a partir do mês definido.")
        vr = c3.slider("Redução variáveis (%)", 0.0, 50.0, 0.0, 0.5,
                       help="Ex.: renegociar taxas e reduzir variável.")
        dur = c4.number_input("Duração (meses) — 0 = indefinido", min_value=0, max_value=36, value=0, step=1,
                              help="0 significa que segue até o fim do horizonte.")
        if st.button("➕ Inserir corte"):
            st.session_state.cuts.append(CostCut(int(cstart), fr/100, vr/100, int(dur)))

    if st.session_state.cuts:
        dfc = pd.DataFrame([asdict(x) for x in st.session_state.cuts])
        st.dataframe(dfc, use_container_width=True)
        if st.button("🧹 Limpar cortes"):
            st.session_state.cuts = []

# =========================
# Step 5: Resultados
# =========================
else:
    st.subheader("Resultados — simulação e leitura executiva")

    left, right = st.columns([1, 2])

    with left:
        d["horizon"] = st.selectbox("Horizonte (meses)", [12, 24, 36], index=[12,24,36].index(d["horizon"]))
        d["three_scenarios"] = st.toggle(
            "Rodar 3 cenários (Base/Otimista/Pessimista)",
            value=bool(d["three_scenarios"]),
            help="Gera variações automáticas nas premissas para mostrar risco e amplitude."
        )

        if d["three_scenarios"]:
            st.markdown("### Ajuste dos cenários")
            st.caption("Você define como o otimista e o pessimista se desviam do cenário base.")
            d["opt_revenue_delta"] = st.slider("Otimista: +pp crescimento receita", 0.0, 20.0, float(d["opt_revenue_delta"]), 0.5)
            d["pess_revenue_delta"] = st.slider("Pessimista: +pp crescimento receita", -20.0, 0.0, float(d["pess_revenue_delta"]), 0.5)
            d["opt_cost_delta"] = st.slider("Otimista: +pp crescimento custos", -10.0, 0.0, float(d["opt_cost_delta"]), 0.5)
            d["pess_cost_delta"] = st.slider("Pessimista: +pp crescimento custos", 0.0, 10.0, float(d["pess_cost_delta"]), 0.5)

        run = st.button("🚀 Rodar simulação", type="primary")

    def run_one_scenario(rev_growth_pct, cost_growth_pct, label):
        df, be, neg, runway = simulate_financeops(
            horizon_months=int(d["horizon"]),
            revenue_base=float(d["revenue_base"]),
            revenue_growth_m=float(rev_growth_pct)/100,
            fixed_cost_base=float(d["fixed_cost_base"]),
            var_cost_base=float(d["var_cost_base"]),
            cost_growth_m=float(cost_growth_pct)/100,
            cash_initial=float(d["cash_initial"]),
            tax_pct=float(d["tax_pct"])/100,
            cogs_pct=float(d["cogs_pct"])/100,
            seasonality_on=bool(d["seasonality_on"]),
            seasonality_12=d["seasonality_12"],
            hirings=st.session_state.hirings,
            investments=st.session_state.investments,
            cuts=st.session_state.cuts,
        )
        return {"label": label, "df": df, "breakeven": be, "cash_neg": neg, "runway": runway}

    if run:
        base_rg = float(d["revenue_growth_pct"])
        base_cg = float(d["cost_growth_pct"])

        scenarios = [run_one_scenario(base_rg, base_cg, "Base")]

        if d["three_scenarios"]:
            opt_rg, pess_rg = scenario_pack(base_rg, d["opt_revenue_delta"], d["pess_revenue_delta"])
            opt_cg, pess_cg = scenario_pack(base_cg, d["opt_cost_delta"], d["pess_cost_delta"])
            scenarios.append(run_one_scenario(opt_rg, opt_cg, "Otimista"))
            scenarios.append(run_one_scenario(pess_rg, pess_cg, "Pessimista"))

        st.session_state["scenarios"] = scenarios

    if "scenarios" not in st.session_state:
        st.warning("Clique em **Rodar simulação** para ver os resultados.")
    else:
        scenarios = st.session_state["scenarios"]

        with right:
            # KPIs summary
            st.markdown("### KPIs por cenário")
            summary_rows = []
            for s in scenarios:
                df = s["df"]
                summary_rows.append({
                    "Cenário": s["label"],
                    "Receita final": df["Receita Bruta"].iloc[-1],
                    "Resultado final": df["Resultado"].iloc[-1],
                    "Caixa final": df["Caixa"].iloc[-1],
                    "Break-even (mês)": s["breakeven"] if s["breakeven"] else "—",
                    "Caixa negativo (mês)": s["cash_neg"] if s["cash_neg"] else "—",
                    "Runway (meses)": round(s["runway"], 1) if s["runway"] is not None else "—",
                })
            s_df = pd.DataFrame(summary_rows)
            st.dataframe(
                s_df.style.format({
                    "Receita final": lambda v: brl(v) if isinstance(v, (float, int)) else v,
                    "Resultado final": lambda v: brl(v) if isinstance(v, (float, int)) else v,
                    "Caixa final": lambda v: brl(v) if isinstance(v, (float, int)) else v,
                }),
                use_container_width=True
            )

            # Charts
            st.markdown("### Gráficos (comparativo)")
            # create wide df for charts
            chart_cash = pd.DataFrame({"Mês": scenarios[0]["df"]["Mês"]}).set_index("Mês")
            chart_res = pd.DataFrame({"Mês": scenarios[0]["df"]["Mês"]}).set_index("Mês")
            for s in scenarios:
                chart_cash[s["label"]] = s["df"].set_index("Mês")["Caixa"]
                chart_res[s["label"]] = s["df"].set_index("Mês")["Resultado"]

            st.caption("Caixa por cenário")
            st.line_chart(chart_cash)

            st.caption("Resultado mensal por cenário")
            st.line_chart(chart_res)

            # Detail table for base scenario
            base = next(x for x in scenarios if x["label"] == "Base")
            df_base = base["df"]

            with st.expander("Tabela completa — cenário Base"):
                st.dataframe(
                    df_base.style.format({
                        "Receita Bruta": lambda v: brl(v),
                        "Impostos": lambda v: brl(v),
                        "COGS": lambda v: brl(v),
                        "Receita Líquida (após imposto/COGS)": lambda v: brl(v),
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

            st.markdown("### Resumo executivo (Base)")
            be = base["breakeven"]
            neg = base["cash_neg"]
            runway = base["runway"]

            resumo = [
                f"• Receita base: {brl(float(d['revenue_base']))}",
                f"• Crescimento receita (base): {pct(float(d['revenue_growth_pct']))} ao mês",
                f"• Custos fixos: {brl(float(d['fixed_cost_base']))} | Custos variáveis: {brl(float(d['var_cost_base']))}",
                f"• Crescimento custos (base): {pct(float(d['cost_growth_pct']))} ao mês",
                f"• Caixa inicial: {brl(float(d['cash_initial']))}",
                f"• Receita final (base): {brl(df_base['Receita Bruta'].iloc[-1])}",
                f"• Caixa final (base): {brl(df_base['Caixa'].iloc[-1])}",
            ]
            if be:
                resumo.append(f"• Break-even: mês {be}.")
            else:
                resumo.append("• Break-even: não ocorre no horizonte.")
            if neg:
                resumo.append(f"• Risco crítico: caixa negativo no mês {neg}.")
            else:
                resumo.append("• Caixa: sem mês negativo no horizonte.")
            if runway is not None:
                resumo.append(f"• Runway estimada (burn médio 3m): {runway:.1f} meses.")

            st.write("\n".join(resumo))

            st.markdown("### Export (Base)")
            csv = df_base.to_csv(index=False).encode("utf-8")
            st.download_button("Baixar CSV do cenário Base", data=csv, file_name="financeops_base.csv", mime="text/csv")
