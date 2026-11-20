import streamlit as st
import pandas as pd
import plotly.express as px
import re

st.set_page_config(page_title="TransUnion Fraud Rule Explorer", layout="wide")
st.title("TransUnion Fraud Rule Explorer")
st.write("Compare optimized fraud-detection rule sets across evolutionary algorithms.")

uploaded_files = st.file_uploader(
    "Upload NSGA-II, SPEA2, and MOEA/D results",
    type="csv", accept_multiple_files=True
)


def standardize_results(df):
    df = df.rename(columns={
        "val_precision": "precision", "val_recall": "recall",
        "val_f1": "f1", "rule_set": "rule_set_string"
    })
    if "solution" not in df.columns:
        df["solution"] = [f"R{i}" for i in range(1, len(df) + 1)]
    return df



# Verified feature categories from the team-provided variable breakdown.
# Fraud_Label is the target, not an input rule condition.
FEATURE_CATEGORIES = {
    # Transaction
    "Transaction_Amount": "Transaction",
    "Transaction_Hour": "Transaction",
    "Time_Since_Last_Transaction": "Transaction",
    "Amount_Deviation_From_Avg": "Transaction",
    # Transaction Velocity
    "Transactions_Last_24h": "Transaction Velocity",
    "Transactions_Last_7d": "Transaction Velocity",
    "Velocity_1h": "Transaction Velocity",
    "Velocity_24h": "Transaction Velocity",
    "Velocity_Transactions_24h": "Transaction Velocity",
    # Account / Customer
    "Account_Age_Days": "Account / Customer",
    "Days_Since_Last_Address_Change": "Account / Customer",
    "Account_Type": "Account / Customer",
    "Customer_Segment": "Account / Customer",
    "Num_Cards": "Account / Customer",
    "Credit_Limit_Utilization": "Account / Customer",
    # Device / Session
    "Device_Risk_Score": "Device / Session",
    "Device_Type": "Device / Session",
    "Session_Duration": "Device / Session",
    "Failed_Login_Count": "Device / Session",
    # Location / IP
    "Distance_From_Home": "Location / IP",
    "Country_Code": "Location / IP",
    "Geo_Location_Risk": "Location / IP",
    "IP_Risk_Score": "Location / IP",
    # Merchant
    "Merchant_Category": "Merchant",
    "Merchant_Risk_Score": "Merchant",
    "Merchant_Size": "Merchant",
    "Num_Unique_Merchants_7d": "Merchant",
    # Payment / Card
    "Payment_Method": "Payment / Card",
    "Card_Presence": "Payment / Card",
    "Card_Type": "Payment / Card",
    "Authentication_Type": "Payment / Card",
    # Historical / Behavioral Risk
    "Chargeback_History": "Historical / Behavioral Risk",
    "Previous_Fraud_Reports": "Historical / Behavioral Risk",
    "Behavioral_Anomaly_Score": "Historical / Behavioral Risk",
    "Card_Usage_Frequency": "Historical / Behavioral Risk",
    # Other
    "Email_Domain_Age_Days": "Other",
    "Transaction_Channel": "Other",
    "Day_Of_Week": "Other",
    "Email_Provider": "Other",
    # Target
    "Fraud_Label": "Target",
}

def strip_outer_parentheses(text):
    """Remove parentheses only when they wrap the whole expression."""
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        wraps_all = True
        for i, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return text
                if depth == 0 and i != len(text) - 1:
                    wraps_all = False
                    break
        if not wraps_all or depth != 0:
            break
        text = text[1:-1].strip()
    return text


def split_top_level(text, connector):
    """Split on a logical connector only outside parentheses and quoted strings."""
    result, start, depth, quote = [], 0, 0, None
    i = 0
    token = f" {connector} "
    while i < len(text):
        char = text[i]
        if quote:
            if char == quote and (i == 0 or text[i - 1] != "\\"):
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and text[i:i + len(token)].upper() == token:
            result.append(text[start:i].strip())
            i += len(token)
            start = i
            continue
        i += 1
    result.append(text[start:].strip())
    return result


def rule_set_table(expression):
    """Show one row per OR-separated rule, preserving its AND conditions."""
    text = strip_outer_parentheses(str(expression))
    rows = []
    for rule_number, rule_text in enumerate(split_top_level(text, "OR"), start=1):
        rule_text = strip_outer_parentheses(rule_text)
        conditions = split_top_level(rule_text, "AND")
        condition_labels = []
        categories = []
        for condition in conditions:
            condition = strip_outer_parentheses(condition)
            match = re.fullmatch(
                r"\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(<=|>=|==|!=|=|<|>)\s*(.*?)\s*",
                condition,
            )
            if match:
                feature, operator, value = match.groups()
                condition_labels.append(f"{feature} {operator} {value}")
                category = FEATURE_CATEGORIES.get(feature, "Unmapped")
            else:
                # Preserve unrecognized conditions rather than changing their meaning.
                condition_labels.append(condition)
                category = "Unmapped"
            if category not in categories:
                categories.append(category)
        rows.append({
            "Rule (OR)": f"Rule {rule_number}",
            "Conditions (AND)": "\nAND\n".join(condition_labels),
            "Feature categories": ", ".join(categories),
        })
    return pd.DataFrame(rows)

if uploaded_files:
    required_columns = {
        "solution", "precision", "recall", "f1", "n_rules",
        "total_conditions", "rule_set_string"
    }
    algorithm_data = {}
    errors = []

    for file in uploaded_files:
        filename = file.name.lower()
        if "nsga" in filename:
            algorithm = "NSGA-II"
        elif "spea" in filename:
            algorithm = "SPEA2"
        elif any(x in filename for x in ("moead", "moea_d", "moea-d")):
            algorithm = "MOEA/D"
        else:
            errors.append(
                f"Could not identify algorithm in '{file.name}'. "
                "Include NSGA, SPEA, or MOEAD in the filename."
            )
            continue

        if algorithm in algorithm_data:
            errors.append(f"Multiple files for {algorithm}; using the first valid file.")
            continue

        try:
            df = standardize_results(pd.read_csv(file))
            missing = required_columns - set(df.columns)
            if missing:
                errors.append(f"{file.name} is missing columns: {', '.join(sorted(missing))}")
                continue
            for col in ("precision", "recall", "f1", "n_rules", "total_conditions"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["algorithm"] = algorithm
            algorithm_data[algorithm] = df
        except Exception as exc:
            errors.append(f"Error reading {file.name}: {exc}")

    for error in errors:
        st.warning(error)

    if algorithm_data:
        st.success(
            f"Successfully loaded {len(algorithm_data)} algorithm(s): "
            + ", ".join(algorithm_data)
        )
        all_results = pd.concat(algorithm_data.values(), ignore_index=True)

        st.subheader("Performance Preferences")
        c1, c2, c3 = st.columns(3)
        with c1:
            min_precision = st.slider("Minimum Precision (%)", 0, 100, 50)
        with c2:
            min_recall = st.slider("Minimum Recall (%)", 0, 100, 40)
        with c3:
            max_conditions = st.slider("Maximum Conditions", 1, 160, 50)

        filtered = all_results.loc[
            (all_results["precision"] * 100 >= min_precision)
            & (all_results["recall"] * 100 >= min_recall)
            & (all_results["total_conditions"] <= max_conditions)
        ].copy().reset_index(drop=True)

        st.subheader("Matching Rule Sets — All Algorithms")
        if filtered.empty:
            st.info("No rule sets match these preferences. Try adjusting the sliders.")
        else:
            display = filtered[[
                "algorithm", "solution", "precision", "recall", "f1",
                "n_rules", "total_conditions"
            ]].copy()
            for col in ("precision", "recall", "f1"):
                display[col] = (display[col] * 100).round(1)
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.write(f"Found {len(filtered)} matching rule sets across {filtered['algorithm'].nunique()} algorithm(s).")

            st.subheader("Precision vs. Recall")
            plot_data = filtered.copy()
            for col in ("precision", "recall", "f1"):
                plot_data[f"{col}_pct"] = plot_data[col] * 100
            plot_data = plot_data.dropna(subset=["precision_pct", "recall_pct"])
            chart_selection = None
            if not plot_data.empty:
                st.caption("Click a point to load its rule set in the inspector below.")
                fig = px.scatter(
                    plot_data, x="recall_pct", y="precision_pct",
                    color="algorithm", symbol="algorithm",
                    size="total_conditions",
                    size_max=22,
                    custom_data=["algorithm", "solution"],
                    hover_data={
                        "solution": True, "f1_pct": ":.1f",
                        "n_rules": True, "total_conditions": True,
                        "recall_pct": ":.1f", "precision_pct": ":.1f"
                    },
                    labels={
                        "recall_pct": "Recall (%)", "precision_pct": "Precision (%)",
                        "algorithm": "Algorithm", "f1_pct": "F1 (%)",
                        "n_rules": "Number of rules", "total_conditions": "Total conditions"
                    },
                    title="Precision–Recall Trade-offs (Filtered Rule Sets)"
                )
                fig.update_layout(legend_title_text="Algorithm")
                chart_selection = st.plotly_chart(
                    fig, use_container_width=True,
                    key="fraud_rule_scatter",
                    on_select="rerun", selection_mode="points"
                )
            else:
                st.info("No valid precision/recall pairs available for the chart.")

            st.subheader("Inspect a Rule Set")
            # Plotly returns the custom_data fields for the selected point.
            # Match by algorithm + solution, not Plotly's per-trace point index.
            selected_index = 0
            if chart_selection is not None:
                points = chart_selection.selection.points
                if points:
                    custom = points[-1].get("customdata", [])
                    if len(custom) >= 2:
                        algorithm, solution = str(custom[0]), str(custom[1])
                        matches = filtered.index[
                            (filtered["algorithm"].astype(str) == algorithm)
                            & (filtered["solution"].astype(str) == solution)
                        ].tolist()
                        if matches:
                            selected_index = matches[0]

            selected_row = st.selectbox(
                "Select an algorithm and solution",
                options=list(range(len(filtered))),
                index=selected_index,
                format_func=lambda i: (
                    f"{filtered.iloc[i]['algorithm']} — "
                    f"{filtered.iloc[i]['solution']} (row {i + 1})"
                )
            )
            rule = filtered.iloc[selected_row]
            st.markdown("**Rule Set Breakdown**")
            st.caption("Each row is a separate OR alternative. Conditions within that row are joined by AND. Feature categories follow the team’s variable breakdown.")
            breakdown = rule_set_table(rule["rule_set_string"])
            st.dataframe(
                breakdown, use_container_width=True, hide_index=True,
                row_height=70,
                column_config={
                    "Rule (OR)": st.column_config.TextColumn(width="small"),
                    "Conditions (AND)": st.column_config.TextColumn(width="large"),
                    "Feature categories": st.column_config.TextColumn(width="medium"),
                },
            )
            with st.expander("View original rule-set expression"):
                st.code(str(rule["rule_set_string"]), language=None)
            st.caption(
                "Transaction-level inspection is not yet enabled. It requires the "
                "underlying transaction records and a validated rule evaluator."
            )
    else:
        st.error("No valid algorithm result files were loaded.")
else:
    st.info("Upload one or more algorithm result CSV files to begin.")
