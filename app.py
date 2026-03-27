
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, roc_auc_score

st.set_page_config(
    page_title="E-commerce Sales Optimization Engine",
    page_icon="📈",
    layout="wide"
)

RAW_FILE = "data/E-commerce Sales Optimization Engine.xlsx"
CLEAN_FILE = "data/cleaned_supermarket_data.csv"

@st.cache_data
def load_raw_data():
    return pd.read_excel(RAW_FILE)

@st.cache_data
def clean_data(df: pd.DataFrame):
    clean = df.drop_duplicates().copy()
    clean["Postal Code"] = clean["Postal Code"].astype(str).str.zfill(5)
    clean["Profit Margin %"] = np.where(clean["Sales"] != 0, clean["Profit"] / clean["Sales"] * 100, np.nan)
    clean["Profitability Status"] = np.where(clean["Profit"] < 0, "Loss-Making", "Profit-Making")
    clean["Discount Band"] = pd.cut(
        clean["Discount"],
        bins=[-0.001, 0, 0.1, 0.2, 0.3, 1.0],
        labels=["No Discount", "0-10%", "10-20%", "20-30%", "30%+"]
    )
    clean["Sales Band"] = pd.qcut(clean["Sales"].rank(method="first"), 4, labels=["Low", "Medium", "High", "Very High"])
    return clean

def money(x):
    return f"${x:,.0f}"

def pct(x):
    return f"{x:.1f}%"

def insight_box(title, bullets):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for bullet in bullets:
            st.write(f"- {bullet}")

def top_entity(df, group_col, metric):
    temp = df.groupby(group_col, as_index=False)[metric].sum().sort_values(metric, ascending=False)
    if temp.empty:
        return None, 0
    row = temp.iloc[0]
    return row[group_col], row[metric]

def bottom_entity(df, group_col, metric):
    temp = df.groupby(group_col, as_index=False)[metric].sum().sort_values(metric, ascending=True)
    if temp.empty:
        return None, 0
    row = temp.iloc[0]
    return row[group_col], row[metric]

def get_filtered_df(df):
    st.sidebar.header("Dashboard Filters")
    region = st.sidebar.multiselect("Region", sorted(df["Region"].dropna().unique()), default=sorted(df["Region"].dropna().unique()))
    segment = st.sidebar.multiselect("Segment", sorted(df["Segment"].dropna().unique()), default=sorted(df["Segment"].dropna().unique()))
    category = st.sidebar.multiselect("Category", sorted(df["Category"].dropna().unique()), default=sorted(df["Category"].dropna().unique()))
    ship = st.sidebar.multiselect("Ship Mode", sorted(df["Ship Mode"].dropna().unique()), default=sorted(df["Ship Mode"].dropna().unique()))
    disc_range = st.sidebar.slider("Discount range", 0.0, float(df["Discount"].max()), (0.0, float(df["Discount"].max())))
    sales_range = st.sidebar.slider("Sales range", 0.0, float(df["Sales"].max()), (0.0, float(df["Sales"].max())))
    filtered = df[
        df["Region"].isin(region) &
        df["Segment"].isin(segment) &
        df["Category"].isin(category) &
        df["Ship Mode"].isin(ship) &
        df["Discount"].between(disc_range[0], disc_range[1]) &
        df["Sales"].between(sales_range[0], sales_range[1])
    ].copy()
    return filtered

raw_df = load_raw_data()
clean_df = clean_data(raw_df)
filtered_df = get_filtered_df(clean_df)

st.title("E-commerce Sales Optimization Engine")
st.caption("A Streamlit dashboard for descriptive and diagnostic analysis of pricing, discounts, product mix, and profitability.")

tabs = st.tabs([
    "1. Business Objective & Strategy",
    "2. Data Audit",
    "3. Commercial Performance",
    "4. Discount & Margin Diagnostics",
    "5. Correlation & Driver Analysis",
    "6. Product Mix & Portfolio Risk",
    "7. Predictive Signals",
    "8. Recommendations"
])

with tabs[0]:
    st.subheader("Project Context")
    st.markdown("""
This dashboard is built as a decision-support engine for an e-commerce business that wants to grow **profitably**, not just grow **sales**.

The core management problem is simple:
- Sales can rise while margins quietly deteriorate.
- Discounts can stimulate demand while weakening contribution.
- Product mix can create revenue without creating value.

The purpose of this dashboard is to diagnose where profit is created, where it is destroyed, and what management should do next.
""")
    col1, col2 = st.columns(2)
    with col1:
        insight_box("Business objective", [
            "Identify which categories, sub-categories, segments, regions, and shipping modes truly create commercial value.",
            "Understand whether discounts are helping demand in a healthy way or simply buying low-quality revenue.",
            "Separate high-sales / low-profit areas from high-profit / scalable areas so product mix decisions become sharper."
        ])
    with col2:
        insight_box("Analytical strategy", [
            "Use descriptive analysis to summarize the commercial structure of the dataset.",
            "Use diagnostic analysis to explain why profitability changes across categories, discounts, regions, and segments.",
            "Use simple predictive signals as an extension to test whether profit and loss patterns are learnable from transactional variables."
        ])
    st.subheader("Questions this dashboard answers")
    st.markdown("""
1. Which categories and sub-categories drive the highest sales and profit?  
2. Which product groups are margin-destructive despite strong revenue?  
3. How strongly does discounting relate to profit deterioration?  
4. Which customer segments and regions are commercially strongest?  
5. Does shipping mode reveal a meaningful profitability pattern?  
6. Can profit or loss-making behaviour be identified from available variables?
""")

with tabs[1]:
    st.subheader("Raw Data Overview")
    a, b, c, d = st.columns(4)
    a.metric("Raw rows", f"{len(raw_df):,}")
    b.metric("Raw columns", raw_df.shape[1])
    c.metric("Missing values", int(raw_df.isna().sum().sum()))
    d.metric("Duplicate rows", int(raw_df.duplicated().sum()))
    st.dataframe(raw_df.head(50), use_container_width=True)

    st.subheader("Cleaning and transformation log")
    cleaning_log = pd.DataFrame({
        "Step": [
            "Duplicate removal",
            "Postal Code formatting",
            "Profit Margin % creation",
            "Profitability Status creation",
            "Discount Band creation",
            "Sales Band creation"
        ],
        "What was done": [
            f"Removed {int(raw_df.duplicated().sum())} exact duplicate rows.",
            "Converted Postal Code from numeric to string to preserve leading zeros and treat it as an identifier, not a measure.",
            "Created Profit Margin % = Profit / Sales * 100 for margin-based analysis.",
            "Created a binary commercial status label: Profit-Making vs Loss-Making.",
            "Created discount buckets to understand non-linear discount behaviour.",
            "Created sales quartile bands for portfolio comparison."
        ],
        "Why it matters": [
            "Duplicate transactions can overstate sales, profit, and order patterns.",
            "Postal codes are labels. Keeping them numeric can distort interpretation and any downstream modelling.",
            "Profit alone is not enough. Margin reveals how efficiently revenue becomes value.",
            "This allows clean classification of destructive vs healthy commercial patterns.",
            "Management decisions are easier when discount behaviour is grouped into interpretable bands.",
            "This helps compare low-ticket and high-ticket commercial behaviour."
        ]
    })
    st.dataframe(cleaning_log, use_container_width=True)

    st.subheader("Cleaned Data Overview")
    a, b, c, d = st.columns(4)
    a.metric("Cleaned rows", f"{len(clean_df):,}")
    b.metric("Rows removed", f"{len(raw_df) - len(clean_df):,}")
    c.metric("Cleaned columns", clean_df.shape[1])
    d.metric("Current filtered rows", f"{len(filtered_df):,}")
    st.dataframe(clean_df.head(50), use_container_width=True)

    insight_box("Data quality interpretation", [
        "The dataset is structurally clean because it has no missing values. This means diagnostic analysis is not distorted by imputation choices.",
        f"The only direct structural issue was duplication. Removing {len(raw_df) - len(clean_df)} rows prevents inflated business performance estimates.",
        "Derived fields such as Profit Margin %, Discount Band, and Profitability Status make the data more decision-ready for management."
    ])

with tabs[2]:
    st.subheader("Commercial Performance Snapshot")
    total_sales = filtered_df["Sales"].sum()
    total_profit = filtered_df["Profit"].sum()
    total_qty = filtered_df["Quantity"].sum()
    margin = (total_profit / total_sales * 100) if total_sales else 0
    loss_share = (filtered_df["Profit"] < 0).mean() * 100 if len(filtered_df) else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sales", money(total_sales))
    m2.metric("Profit", money(total_profit))
    m3.metric("Quantity", f"{int(total_qty):,}")
    m4.metric("Profit margin", pct(margin))
    m5.metric("Loss-making rows", pct(loss_share))

    sales_cat = filtered_df.groupby("Category", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    fig = px.bar(sales_cat.sort_values("Sales", ascending=False), x="Category", y="Sales", text_auto=".2s", title="Sales by Category")
    st.plotly_chart(fig, use_container_width=True)

    top_sales_cat, top_sales_val = top_entity(filtered_df, "Category", "Sales")
    top_profit_cat, top_profit_val = top_entity(filtered_df, "Category", "Profit")
    low_profit_cat, low_profit_val = bottom_entity(filtered_df, "Category", "Profit")
    insight_box("Insight: category view", [
        f"The chart displays total revenue generated by each category under the current filters.",
        f"{top_sales_cat} contributes the highest sales at {money(top_sales_val)}.",
        f"{top_profit_cat} is the strongest profit contributor at {money(top_profit_val)}, while {low_profit_cat} is the weakest on profit at {money(low_profit_val)}.",
        "A category can lead in sales without leading in profit. That gap is where management should look for discount pressure, poor product mix, or cost inefficiency."
    ])

    subcat = filtered_df.groupby("Sub-Category", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    fig = px.bar(subcat.sort_values("Profit", ascending=False), x="Sub-Category", y="Profit", color="Profit", title="Profit by Sub-Category")
    st.plotly_chart(fig, use_container_width=True)
    top_sub_sales, top_sub_sales_val = top_entity(filtered_df, "Sub-Category", "Sales")
    top_sub_profit, top_sub_profit_val = top_entity(filtered_df, "Sub-Category", "Profit")
    low_sub_profit, low_sub_profit_val = bottom_entity(filtered_df, "Sub-Category", "Profit")
    insight_box("Insight: sub-category view", [
        f"This graph shows which sub-categories create or destroy value after transactions are aggregated.",
        f"{top_sub_sales} is the top revenue sub-category at {money(top_sub_sales_val)}.",
        f"{top_sub_profit} contributes the highest profit at {money(top_sub_profit_val)}, while {low_sub_profit} destroys the most profit at {money(low_sub_profit_val)}.",
        "Sub-category analysis is where hidden portfolio problems usually appear, because broad category averages can mask weak commercial pockets."
    ])

    col1, col2 = st.columns(2)
    with col1:
        seg = filtered_df.groupby("Segment", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        fig = px.bar(seg.sort_values("Profit", ascending=False), x="Segment", y="Profit", color="Segment", title="Profit by Customer Segment")
        st.plotly_chart(fig, use_container_width=True)
        top_seg, top_seg_profit = top_entity(filtered_df, "Segment", "Profit")
        insight_box("Insight: segment view", [
            "This chart compares the economic quality of customer segments, not just their size.",
            f"{top_seg} is the strongest profit segment at {money(top_seg_profit)}.",
            "A valuable segment is one that combines strong sales with healthy margin, not one that only absorbs discounts."
        ])
    with col2:
        reg = filtered_df.groupby("Region", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        fig = px.bar(reg.sort_values("Profit", ascending=False), x="Region", y="Profit", color="Region", title="Profit by Region")
        st.plotly_chart(fig, use_container_width=True)
        top_reg, top_reg_profit = top_entity(filtered_df, "Region", "Profit")
        low_reg, low_reg_profit = bottom_entity(filtered_df, "Region", "Profit")
        insight_box("Insight: regional view", [
            "This graph reveals whether commercial performance is geographically concentrated or uneven.",
            f"{top_reg} is the best-performing region on profit at {money(top_reg_profit)}.",
            f"{low_reg} is the weakest region at {money(low_reg_profit)}.",
            "Underperforming regions deserve a mix review and discount discipline review before any growth spend is increased."
        ])

    ship = filtered_df.groupby("Ship Mode", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    ship["Profit Margin %"] = np.where(ship["Sales"] != 0, ship["Profit"] / ship["Sales"] * 100, 0)
    fig = px.bar(ship.sort_values("Profit Margin %", ascending=False), x="Ship Mode", y="Profit Margin %", color="Ship Mode", title="Profit Margin by Shipping Mode")
    st.plotly_chart(fig, use_container_width=True)
    top_ship, top_ship_margin = ship.sort_values("Profit Margin %", ascending=False).iloc[0][["Ship Mode", "Profit Margin %"]]
    insight_box("Insight: shipping mode", [
        "Shipping mode may reflect service economics or order behaviour rather than customer preference alone.",
        f"{top_ship} currently has the highest profit margin at {pct(top_ship_margin)}.",
        "If a shipping mode shows high sales but weak margin, it may signal hidden fulfilment or discount trade-offs."
    ])

with tabs[3]:
    st.subheader("Discount and Margin Diagnostics")
    disc_profit = filtered_df.groupby("Discount Band", as_index=False).agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Transactions=("Profit", "size")
    ).dropna()
    disc_profit["Profit Margin %"] = np.where(disc_profit["Sales"] != 0, disc_profit["Profit"] / disc_profit["Sales"] * 100, 0)
    fig = px.bar(disc_profit, x="Discount Band", y="Profit Margin %", color="Discount Band", title="Profit Margin by Discount Band")
    st.plotly_chart(fig, use_container_width=True)

    corr = filtered_df[["Sales", "Quantity", "Discount", "Profit", "Profit Margin %"]].corr(numeric_only=True).round(3)
    discount_profit_corr = corr.loc["Discount", "Profit"] if "Discount" in corr.index else np.nan
    high_discount_share = (filtered_df["Discount"] >= 0.2).mean() * 100 if len(filtered_df) else 0
    insight_box("Insight: discount band analysis", [
        "This chart groups transactions into discount buckets to show whether higher markdowns are commercially sustainable.",
        f"The correlation between Discount and Profit is {discount_profit_corr:.3f}. A strongly negative value suggests discounts are closely associated with weaker profit outcomes.",
        f"{pct(high_discount_share)} of filtered transactions sit at 20% discount or above. This matters because aggressive discounting can quickly compress margin even when revenue is preserved.",
        "If higher discount bands consistently show weaker margin, the business should stop treating discounting as a universal growth lever."
    ])

    col1, col2 = st.columns(2)
    with col1:
        sample_df = filtered_df.copy()
        if len(sample_df) > 3000:
            sample_df = sample_df.sample(3000, random_state=42)
        fig = px.scatter(
            sample_df,
            x="Discount", y="Profit",
            color="Category", size="Sales",
            hover_data=["Sub-Category", "Segment", "Region"],
            title="Discount vs Profit"
        )
        st.plotly_chart(fig, use_container_width=True)
        insight_box("Insight: discount versus profit", [
            "Each point is a transaction. Bubble size reflects sales value, while colour shows category.",
            "If points shift downward as discount increases, that indicates a profit erosion pattern.",
            "Large sales bubbles in negative-profit areas are especially important because they signal revenue that looks attractive but destroys value."
        ])
    with col2:
        fig = px.scatter(
            sample_df,
            x="Sales", y="Profit",
            color="Profitability Status",
            hover_data=["Category", "Sub-Category", "Segment", "Region"],
            title="Sales vs Profit"
        )
        st.plotly_chart(fig, use_container_width=True)
        negative_high_sales = filtered_df[(filtered_df["Sales"] > filtered_df["Sales"].median()) & (filtered_df["Profit"] < 0)].shape[0]
        insight_box("Insight: sales versus profit", [
            "This graph checks whether high sales are reliably translating into healthy profit.",
            f"There are {negative_high_sales:,} filtered transactions with above-median sales but negative profit.",
            "This is the clearest sign that topline growth alone is not a safe management metric."
        ])

with tabs[4]:
    st.subheader("Correlation and Driver Analysis")
    corr_df = filtered_df[["Sales", "Quantity", "Discount", "Profit", "Profit Margin %"]].corr(numeric_only=True)
    heat = px.imshow(
        corr_df.round(2),
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap of Key Numeric Variables"
    )
    st.plotly_chart(heat, use_container_width=True)

    insight_box("How to read this heatmap", [
        "Positive values mean two variables tend to move in the same direction. Negative values mean they move in opposite directions.",
        "A positive Sales-Profit correlation is normal, but it is not enough on its own because revenue can still be low quality.",
        "A negative Discount-Profit or Discount-Margin relationship is strategically important because it points to possible markdown-led margin damage."
    ])

    state_perf = filtered_df.groupby("State", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    state_perf["Profit Margin %"] = np.where(state_perf["Sales"] != 0, state_perf["Profit"] / state_perf["Sales"] * 100, 0)
    fig = px.scatter(
        state_perf,
        x="Sales", y="Profit",
        size=state_perf["Sales"].abs(),
        color="Profit Margin %",
        hover_name="State",
        title="State-Level Sales vs Profit"
    )
    st.plotly_chart(fig, use_container_width=True)

    best_state = state_perf.sort_values("Profit", ascending=False).iloc[0]
    worst_state = state_perf.sort_values("Profit", ascending=True).iloc[0]
    insight_box("Insight: state-level diagnostic", [
        "This chart reveals whether state-level commercial scale is translating into state-level profit.",
        f"{best_state['State']} is the top profit state at {money(best_state['Profit'])}.",
        f"{worst_state['State']} is the weakest profit state at {money(worst_state['Profit'])}.",
        "Large states with weak profit are prime candidates for mix correction, pricing discipline, and discount review."
    ])

    driver = filtered_df.groupby(["Category", "Sub-Category"], as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    driver["Profit Margin %"] = np.where(driver["Sales"] != 0, driver["Profit"] / driver["Sales"] * 100, 0)
    fig = px.scatter(
        driver,
        x="Sales", y="Profit Margin %",
        color="Category",
        size="Profit",
        hover_name="Sub-Category",
        title="Sub-Category Scale vs Profitability"
    )
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Insight: scale versus profitability", [
        "This view separates large sub-categories that scale efficiently from those that scale badly.",
        "Sub-categories in the upper-right area combine revenue with healthy margin and are the best candidates for selective growth.",
        "Sub-categories with high sales but low or negative margin need corrective action before any further volume push."
    ])

with tabs[5]:
    st.subheader("Product Mix and Portfolio Risk")
    mix = filtered_df.groupby(["Category", "Sub-Category"], as_index=False).agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
    mix["Profit Margin %"] = np.where(mix["Sales"] != 0, mix["Profit"] / mix["Sales"] * 100, 0)
    sales_med = mix["Sales"].median() if len(mix) else 0
    margin_med = mix["Profit Margin %"].median() if len(mix) else 0

    def quadrant(row):
        if row["Sales"] >= sales_med and row["Profit Margin %"] >= margin_med:
            return "Scale Up"
        if row["Sales"] >= sales_med and row["Profit Margin %"] < margin_med:
            return "Fix Margin"
        if row["Sales"] < sales_med and row["Profit Margin %"] >= margin_med:
            return "Protect Niche"
        return "Review / Rationalize"

    mix["Portfolio Action"] = mix.apply(quadrant, axis=1)
    fig = px.scatter(
        mix, x="Sales", y="Profit Margin %", color="Portfolio Action",
        size=mix["Profit"].abs() + 1,
        hover_name="Sub-Category",
        title="Product Mix Portfolio Matrix"
    )
    fig.add_vline(x=sales_med, line_dash="dash")
    fig.add_hline(y=margin_med, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    quadrant_counts = mix["Portfolio Action"].value_counts().to_dict()
    insight_box("How to use the portfolio matrix", [
        "The vertical split separates larger-revenue sub-categories from smaller ones. The horizontal split separates stronger-margin sub-categories from weaker ones.",
        f"Current mix snapshot: {quadrant_counts}.",
        "Scale Up means large and healthy. Fix Margin means commercially important but margin-stressed. Protect Niche means smaller but efficient. Review / Rationalize means neither large nor attractive enough under current conditions."
    ])

    loss_mix = mix.sort_values("Profit").head(10)
    fig = px.bar(loss_mix, x="Sub-Category", y="Profit", color="Category", title="Top 10 Margin-Destructive Sub-Categories")
    st.plotly_chart(fig, use_container_width=True)
    insight_box("Insight: margin-destructive portfolio areas", [
        "This chart isolates the sub-categories that hurt profitability the most after aggregation.",
        "These are the highest-priority areas for pricing, assortment, discount, or fulfilment intervention.",
        "Management should not automatically delist these areas, but should first test whether margin damage is caused by discounting, region mix, or low-quality demand."
    ])

with tabs[6]:
    st.subheader("Predictive Signals")
    st.markdown("This section is included as an analytical extension. The main project focus remains descriptive and diagnostic analysis.")
    model_df = filtered_df.copy()
    if len(model_df) < 100:
        st.warning("Not enough filtered rows for stable modelling. Widen the filters in the sidebar.")
    else:
        features = ["Ship Mode", "Segment", "City", "State", "Region", "Category", "Sub-Category", "Sales", "Quantity", "Discount"]
        X = model_df[features]
        y_reg = model_df["Profit"]
        y_cls = (model_df["Profit"] < 0).astype(int)

        cat_cols = ["Ship Mode", "Segment", "City", "State", "Region", "Category", "Sub-Category"]
        num_cols = ["Sales", "Quantity", "Discount"]

        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        reg_pipe = Pipeline([
            ("prep", pre),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        ])
        reg_pipe.fit(X_train, y_train)
        pred = reg_pipe.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
        cls_pipe = Pipeline([
            ("prep", pre),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"))
        ])
        cls_pipe.fit(X_train_c, y_train_c)
        pred_c = cls_pipe.predict(X_test_c)
        prob_c = cls_pipe.predict_proba(X_test_c)[:, 1]
        acc = accuracy_score(y_test_c, pred_c)
        f1 = f1_score(y_test_c, pred_c)
        auc = roc_auc_score(y_test_c, prob_c)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Profit model R²", f"{r2:.2f}")
        c2.metric("Profit model MAE", f"{mae:.2f}")
        c3.metric("Loss classifier accuracy", f"{acc:.2f}")
        c4.metric("Loss classifier F1", f"{f1:.2f}")
        c5.metric("Loss classifier ROC-AUC", f"{auc:.2f}")

        insight_box("Interpretation of predictive signals", [
            "The regression model checks whether profit can be estimated from available transaction features with useful accuracy.",
            "The classification model checks whether loss-making patterns are systematic enough to be identified in advance.",
            "If the classification score is strong, the business can build simple early-warning rules to flag risky orders, risky discount structures, or weak product mix combinations."
        ])

with tabs[7]:
    st.subheader("Management Recommendations")
    category_profit = filtered_df.groupby("Category", as_index=False)["Profit"].sum().sort_values("Profit")
    weak_cat = category_profit.iloc[0]["Category"] if len(category_profit) else "N/A"
    subcat_profit = filtered_df.groupby("Sub-Category", as_index=False)["Profit"].sum().sort_values("Profit")
    weak_sub = subcat_profit.iloc[0]["Sub-Category"] if len(subcat_profit) else "N/A"
    strong_sub = subcat_profit.iloc[-1]["Sub-Category"] if len(subcat_profit) else "N/A"

    recs = [
        f"Protect profit before chasing more volume. The weakest category on profit in the current view is **{weak_cat}**, so growth decisions there should be margin-led rather than sales-led.",
        f"Run an immediate pricing and discount review for **{weak_sub}**, because it is the most margin-destructive sub-category in the filtered data.",
        f"Prioritize selective growth in stronger sub-categories such as **{strong_sub}**, where the business appears to convert revenue into value more efficiently.",
        "Treat discounts as a precision tool, not a blanket growth tool. High-discount orders should be reviewed by category, region, and segment before scaling.",
        "Use the portfolio matrix operationally: scale healthy winners, repair important but weak-margin lines, and rationalize low-scale weak-margin areas."
    ]
    insight_box("Recommended strategy actions", recs)

    st.subheader("Suggested operating model for the Sales Optimization Engine")
    st.markdown("""
- **Weekly review:** Track category, sub-category, segment, region, and shipping profitability.  
- **Discount governance:** Approve deeper discounts only in areas where they still protect healthy margin.  
- **Portfolio review:** Use the matrix to decide what to scale, fix, protect, or reconsider.  
- **Risk flags:** Treat high-sales negative-profit transactions as management exceptions.  
- **Decision rule:** Reward profitable growth, not sales growth in isolation.
""")
