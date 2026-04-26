"""
====================================================================
  Defence Budget Prediction System – India
  Subject  : Fundamental Data Analysis
  Model    : Linear Regression (sklearn)
  UI       : Streamlit
====================================================================
  Assumptions & Limitations
  --------------------------
  • Data covers 2015-2024 (10 years) – a small sample; predictions
    beyond 2027 carry high uncertainty.
  • GDP values are in Trillion USD; Budget values in Billion USD.
  • Linear Regression assumes a linear relationship between features
    and the target variable; real-world defence budgets can be
    influenced by geopolitical events not captured here.
  • "Previous Year Budget" is the single strongest predictor due to
    budget inertia (governments rarely cut defence spending abruptly).
====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ──────────────────────────────────────────────
# 1.  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="India Defence Budget Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 2.  CUSTOM CSS  (dark military theme)
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google font ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Share+Tech+Mono&family=Inter:wght@300;400;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0b0f14;
    --surface:  #131920;
    --border:   #1e2d3d;
    --accent:   #00d4a1;
    --accent2:  #0099ff;
    --danger:   #ff4b4b;
    --warn:     #f0a500;
    --text:     #c9d6e3;
    --subtext:  #607080;
}

/* ── Global ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Headings ── */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; }
h1 { color: var(--accent) !important; letter-spacing: 2px; }
h2 { color: var(--accent2) !important; }
h3 { color: var(--text) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] label {
    color: var(--text) !important;
    font-size: 0.82rem !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: var(--subtext) !important;
    font-size: 0.75rem !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
}

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #00d4a1 0%, #0099ff 100%) !important;
    color: #000 !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    font-size: 0.9rem !important;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85 !important; }

/* ── Input widgets ── */
input[type="number"], .stNumberInput input {
    background: var(--surface) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #0b1a14 0%, #0b1525 100%);
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
    margin: 18px 0;
    box-shadow: 0 0 40px rgba(0,212,161,0.12);
}
.result-box .label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: var(--subtext);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.result-box .value {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    color: var(--accent);
    font-weight: 700;
}
.result-box .sub {
    font-size: 0.85rem;
    color: var(--subtext);
    margin-top: 6px;
}

/* ── Trend indicator ── */
.trend-up   { color: #00d4a1 !important; font-weight: 700; }
.trend-down { color: var(--danger) !important; font-weight: 700; }

/* ── Table ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Info / warning banners ── */
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 3.  DATA  (real-world sourced: SIPRI + IMF/World Bank)
# ──────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Historical data sourced from:
      • SIPRI Military Expenditure Database (defence budget USD, % of GDP)
      • IMF / World Bank (GDP current USD)
    All figures are rounded to 2 decimal places.
    GDP unit   : Trillion USD
    Budget unit: Billion  USD
    """
    data = {
        "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "GDP_Trillion_USD": [2.10, 2.29, 2.65, 2.72, 2.87, 2.67, 3.18, 3.39, 3.73, 4.11],
        "Defence_Budget_Billion_USD": [47.91, 51.13, 52.51, 57.91, 61.02, 64.09, 66.52, 76.60, 83.59, 92.66],
        "Defence_pct_GDP": [2.28, 2.23, 1.98, 2.13, 2.13, 2.40, 2.09, 2.26, 2.24, 2.25],
    }
    df = pd.DataFrame(data)

    # ── Lag feature: previous year's budget ──
    df["Prev_Budget_Billion_USD"] = df["Defence_Budget_Billion_USD"].shift(1)

    # ── Drop 2015 row (NaN lag) ──
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ──────────────────────────────────────────────
# 4.  MODEL TRAINING
# ──────────────────────────────────────────────
@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Train a Linear Regression model.
    Features  : Year, GDP, Defence%GDP, Previous Year Budget
    Target    : Defence Budget (Billion USD)
    Split     : 80% train / 20% test (random_state=42 for reproducibility)
    """
    feature_cols = ["Year", "GDP_Trillion_USD", "Defence_pct_GDP", "Prev_Budget_Billion_USD"]
    X = df[feature_cols]
    y = df["Defence_Budget_Billion_USD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae   = mean_absolute_error(y_test, y_pred_test)
    r2    = r2_score(y_test, y_pred_test)

    # Full-dataset predictions for chart
    y_pred_all = model.predict(X)

    return model, X_train, X_test, y_train, y_test, y_pred_test, y_pred_all, mae, r2, feature_cols


# ──────────────────────────────────────────────
# 5.  MATPLOTLIB THEME HELPER
# ──────────────────────────────────────────────
def apply_dark_theme(ax, fig):
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#131920")
    ax.tick_params(colors="#607080", labelsize=9)
    ax.xaxis.label.set_color("#607080")
    ax.yaxis.label.set_color("#607080")
    ax.title.set_color("#c9d6e3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d3d")
    ax.grid(color="#1e2d3d", linestyle="--", linewidth=0.6, alpha=0.7)


# ──────────────────────────────────────────────
# 6.  CHARTS
# ──────────────────────────────────────────────
def plot_historical(df: pd.DataFrame, y_pred_all: np.ndarray):
    """Line chart: Actual vs Model-Fitted budget over 2016-2024."""
    fig, ax = plt.subplots(figsize=(9, 4))
    apply_dark_theme(ax, fig)

    years_plot = df["Year"].values
    actual     = df["Defence_Budget_Billion_USD"].values

    ax.plot(years_plot, actual, "o-", color="#00d4a1", lw=2.2,
            markersize=7, label="Actual Budget", zorder=3)
    ax.plot(years_plot, y_pred_all, "s--", color="#0099ff", lw=1.8,
            markersize=5, label="Model Fit", zorder=2, alpha=0.85)
    ax.fill_between(years_plot, actual, y_pred_all, alpha=0.08, color="#00d4a1")

    ax.set_title("India Defence Budget – Actual vs Model Fit (USD Billion)", pad=14,
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Budget (Billion USD)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))
    ax.set_xticks(years_plot)
    ax.legend(facecolor="#131920", edgecolor="#1e2d3d", labelcolor="#c9d6e3",
              fontsize=9)
    plt.tight_layout()
    return fig


def plot_future(df: pd.DataFrame, future_df: pd.DataFrame, future_preds: np.ndarray):
    """Extended forecast chart – dynamically extends to whatever year the user selected."""
    end_year   = int(future_df["Year"].iloc[-1])
    n_forecast = len(future_df)

    # Wider figure for long forecasts
    fig_w = max(10, 8 + n_forecast * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    apply_dark_theme(ax, fig)

    hist_years  = df["Year"].values
    hist_budget = df["Defence_Budget_Billion_USD"].values
    fut_years   = future_df["Year"].values

    ax.plot(hist_years, hist_budget, "o-", color="#00d4a1", lw=2.2,
            markersize=7, label="Historical (Actual)", zorder=3)
    ax.plot(fut_years, future_preds, "D--", color="#f0a500", lw=2,
            markersize=5 if n_forecast > 8 else 8,
            label=f"Forecast (2025–{end_year})", zorder=3)

    # Shade forecast region
    ax.axvspan(2024.5, end_year + 0.5, alpha=0.06, color="#f0a500")
    ax.axvline(2024.5, color="#1e2d3d", lw=1.2, ls=":")

    # Annotate forecast points — skip labels when too many to avoid clutter
    label_every = max(1, n_forecast // 8)
    for i, (yr, val) in enumerate(zip(fut_years, future_preds)):
        if i % label_every == 0 or i == n_forecast - 1:
            ax.annotate(f"${val:.0f}B", xy=(yr, val),
                        xytext=(0, 14), textcoords="offset points",
                        ha="center", fontsize=7.5, color="#f0a500",
                        fontfamily="monospace")

    ax.set_title(f"India Defence Budget – Historical + Forecast 2025–{end_year} (USD Billion)",
                 pad=14, fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Budget (Billion USD)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))

    # Smart x-tick spacing so labels never overlap
    all_years = np.append(hist_years, fut_years)
    total_span = end_year - int(hist_years[0])
    tick_step  = 1 if total_span <= 15 else (2 if total_span <= 22 else 5)
    tick_years = [y for y in all_years if (y - int(hist_years[0])) % tick_step == 0]
    ax.set_xticks(tick_years)
    ax.tick_params(axis='x', rotation=45 if n_forecast > 8 else 0)

    ax.legend(facecolor="#131920", edgecolor="#1e2d3d", labelcolor="#c9d6e3",
              fontsize=9)
    plt.tight_layout()
    return fig


def plot_coefficients(model, feature_cols):
    """Bar chart of Linear Regression coefficients."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    apply_dark_theme(ax, fig)

    labels = ["Year", "GDP\n(T USD)", "Defence\n% GDP", "Prev Budget\n(B USD)"]
    coefs  = model.coef_
    colors = ["#00d4a1" if c > 0 else "#ff4b4b" for c in coefs]

    bars = ax.barh(labels, coefs, color=colors, edgecolor="#0b0f14", height=0.55)
    ax.axvline(0, color="#607080", lw=0.8)

    for bar, val in zip(bars, coefs):
        ax.text(val + (max(coefs) * 0.02), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8,
                color="#c9d6e3", fontfamily="monospace")

    ax.set_title("Feature Coefficients (Linear Regression)", pad=10,
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Coefficient Value")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# 7.  MAIN APP
# ──────────────────────────────────────────────
def main():
    # ── Load data & train model ──
    df = load_data()
    (model, X_train, X_test, y_train, y_test,
     y_pred_test, y_pred_all, mae, r2, feature_cols) = train_model(df)

    # ── Dynamic forecast: always computed AFTER inputs are read ──
    # (actual computation happens below, after input_year is known)
    last_gdp_growth = 0.095   # ~9.5% nominal GDP growth assumption
    last_def_pct    = df["Defence_pct_GDP"].iloc[-1]
    last_budget     = df["Defence_Budget_Billion_USD"].iloc[-1]
    last_gdp        = df["GDP_Trillion_USD"].iloc[-1]

    def build_forecast(end_year: int):
        """Forecast every year from 2025 up to end_year using chained lag predictions."""
        rows = []
        prev_b = last_budget
        prev_g = last_gdp
        for yr in range(2025, end_year + 1):
            gdp = round(prev_g * (1 + last_gdp_growth), 3)
            row = {"Year": yr, "GDP_Trillion_USD": gdp,
                   "Defence_pct_GDP": last_def_pct,
                   "Prev_Budget_Billion_USD": prev_b}
            rows.append(row)
            prev_g = gdp
            prev_b = model.predict(pd.DataFrame([row]))[0]
        fdf   = pd.DataFrame(rows)
        preds = model.predict(fdf[feature_cols])
        return fdf, preds

    # ── HEADER ──
    st.markdown(
        "<h1 style='text-align:center;'>🛡️ INDIA DEFENCE BUDGET PREDICTOR</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#607080; font-family:Share Tech Mono,monospace; "
        "font-size:0.8rem; letter-spacing:2px;'>MACHINE LEARNING · LINEAR REGRESSION · 2015–FORECAST</p>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── KPI CARDS ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (Test)", f"${mae:.2f}B", help="Mean Absolute Error on test set")
    c2.metric("R² Score",   f"{r2:.4f}",    help="Coefficient of determination (1.0 = perfect)")
    c3.metric("Training Samples", f"{len(X_train)}")
    c4.metric("Test Samples",     f"{len(X_test)}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── TWO COLUMNS: Sidebar prediction | Charts ──
    left, right = st.columns([1, 2])

    # ── SIDEBAR-STYLE PANEL ──
    with left:
        st.markdown("### 🎯 Custom Prediction")
        st.markdown(
            "<p style='font-size:0.78rem; color:#607080; font-family:Share Tech Mono,monospace;'>"
            "Enter economic parameters to get a model prediction.</p>",
            unsafe_allow_html=True
        )

        input_year    = st.number_input("Year",                   min_value=2024, max_value=2040, value=2025, step=1)
        input_gdp     = st.number_input("GDP (Trillion USD)",      min_value=1.0,  max_value=20.0, value=4.50, step=0.05, format="%.2f")
        input_def_pct = st.number_input("Defence % of GDP",       min_value=0.5,  max_value=10.0, value=2.25, step=0.05, format="%.2f")
        input_prev    = st.number_input("Previous Year Budget (B USD)", min_value=10.0, max_value=500.0, value=92.66, step=0.5, format="%.2f")

        predict_clicked = st.button("🔮  PREDICT BUDGET")

        # ── Build dynamic forecast up to input_year (runs on every interaction) ──
        future_df, future_preds = build_forecast(input_year)

        if predict_clicked:
            inp = pd.DataFrame([{
                "Year": input_year,
                "GDP_Trillion_USD": input_gdp,
                "Defence_pct_GDP": input_def_pct,
                "Prev_Budget_Billion_USD": input_prev,
            }])
            pred_val = model.predict(inp)[0]
            change   = pred_val - input_prev
            pct_chg  = (change / input_prev) * 100

            trend_class = "trend-up" if change >= 0 else "trend-down"
            trend_icon  = "▲" if change >= 0 else "▼"

            st.markdown(f"""
            <div class="result-box">
                <div class="label">Predicted Defence Budget</div>
                <div class="value">${pred_val:.2f}B</div>
                <div class="sub">
                    <span class="{trend_class}">{trend_icon} {abs(pct_chg):.1f}% vs prev year</span>
                    &nbsp;·&nbsp; Year {input_year}
                </div>
            </div>
            """, unsafe_allow_html=True)

            inr_approx = pred_val * 83.5  # rough USD→INR conversion (1 USD ≈ ₹83.5)
            st.info(f"≈ ₹{inr_approx:,.0f} Crore  (at ~₹83.5/USD)")

            if change >= 0:
                st.success(f"📈 Budget projected to INCREASE by ${abs(change):.2f}B ({abs(pct_chg):.1f}%)")
            else:
                st.warning(f"📉 Budget projected to DECREASE by ${abs(change):.2f}B ({abs(pct_chg):.1f}%)")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"### 📌 Auto-Forecast 2025–{input_year}")
        forecast_table = pd.DataFrame({
            "Year":         future_df["Year"].values,
            "GDP (T $)":    future_df["GDP_Trillion_USD"].values.round(2),
            "Budget (B $)": future_preds.round(2),
        })
        st.dataframe(forecast_table, hide_index=True, use_container_width=True)

    # ── CHARTS — rebuilt dynamically every time input_year changes ──
    with right:
        st.markdown("### 📊 Historical Budget – Actual vs Model Fit")
        st.pyplot(plot_historical(df, y_pred_all), use_container_width=True)

        st.markdown(f"### 🔭 Extended Forecast (2025–{input_year})")
        st.pyplot(plot_future(df, future_df, future_preds), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── FEATURE IMPORTANCE & DATASET ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🧠 Feature Coefficients")
        st.pyplot(plot_coefficients(model, feature_cols), use_container_width=True)
        st.caption("Positive bars increase the predicted budget; negative bars decrease it.")

    with col_b:
        st.markdown("### 📋 Dataset Used for Training")
        display_df = df.rename(columns={
            "GDP_Trillion_USD":          "GDP (T $)",
            "Defence_Budget_Billion_USD":"Budget (B $)",
            "Defence_pct_GDP":           "Def% GDP",
            "Prev_Budget_Billion_USD":   "Prev Budget (B $)",
        })
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        st.caption("Sources: SIPRI Military Expenditure Database · IMF / World Bank")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── MODEL EXPLANATION ──
    with st.expander("🔬 Model Logic & Assumptions"):
        st.markdown("""
**Algorithm:** Linear Regression  
`Budget = β₀ + β₁·Year + β₂·GDP + β₃·Def%GDP + β₄·PrevBudget`

**Why Linear Regression?**  
- Interpretable coefficients — great for a college project viva  
- Works well when the target variable has a roughly linear trend over time  
- Fast to train; easy to validate with MAE and R²

**Key Assumptions**
1. India's defence budget grows broadly in line with GDP.  
2. The *previous year budget* (lag-1) captures budget inertia — the strongest single predictor.  
3. Defence spending as a % of GDP remains near the historical mean (~2.2%).  
4. GDP grows at ~9.5% nominally per year for the 2025–2027 forecast window.

**Limitations**
- Small dataset (9 training points after lag) → coefficients are sensitive to outliers.  
- Linear model cannot capture sudden geopolitical shocks (e.g., border conflicts, pandemics).  
- Currency fluctuations (INR/USD) are not modelled explicitly.  
- For production use, consider Random Forest or XGBoost with more features.

**Data Sources**
| Source | Data |
|--------|------|
| SIPRI Military Expenditure Database | Defence budget (USD), % of GDP |
| IMF World Economic Outlook / World Bank | GDP at current USD |
        """)

    # ── FOOTER ──
    st.markdown(
        "<p style='text-align:center; font-size:0.72rem; color:#2a3a4a; "
        "font-family:Share Tech Mono,monospace; margin-top:24px;'>"
        "DEFENCE BUDGET PREDICTION SYSTEM · FUNDAMENTAL DATA ANALYSIS · "
        "LINEAR REGRESSION MODEL · DATA: SIPRI + IMF/WORLD BANK"
        "</p>",
        unsafe_allow_html=True
    )


# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()