import streamlit as st
import pandas as pd
import plotly.express as px

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
            if not plot_data.empty:
                fig = px.scatter(
                    plot_data, x="recall_pct", y="precision_pct",
                    color="algorithm", symbol="algorithm",
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
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid precision/recall pairs available for the chart.")

            st.subheader("Inspect a Rule Set")
            selected_row = st.selectbox(
                "Select an algorithm and solution",
                options=list(range(len(filtered))),
                format_func=lambda i: (
                    f"{filtered.iloc[i]['algorithm']} — "
                    f"{filtered.iloc[i]['solution']} (row {i + 1})"
                )
            )
            rule = filtered.iloc[selected_row]
            st.code(str(rule["rule_set_string"]), language=None)
            st.caption(
                "Transaction-level inspection is not yet enabled. It requires the "
                "underlying transaction records and a validated rule evaluator."
            )
    else:
        st.error("No valid algorithm result files were loaded.")
else:
    st.info("Upload one or more algorithm result CSV files to begin.")
