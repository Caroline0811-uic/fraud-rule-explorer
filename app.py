
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="TransUnion Fraud Rule Explorer",
    layout="wide"
)

st.title("TransUnion Fraud Rule Explorer")
st.write("Explore optimized fraud detection rule sets across three evolutionary algorithms.")

# Upload all three algorithm result files
uploaded_files = st.file_uploader(
    "Upload NSGA-II, SPEA2, and MOEA/D results",
    type="csv",
    accept_multiple_files=True
)

def standardize_results(df):
    df = df.rename(columns={
        "val_precision": "precision",
        "val_recall": "recall",
        "val_f1": "f1",
        "rule_set": "rule_set_string"
    })

    if "solution" not in df.columns:
        df["solution"] = [
            f"R{i}" for i in range(1, len(df) + 1)
        ]

    return df

if uploaded_files:
    required_columns = {
         "solution", "precision", "recall", "f1",
         "n_rules", "total_conditions", "rule_set_string"
        }


    algorithm_data = {}
    errors = []

    for file in uploaded_files:
        filename = file.name.lower()

        # Identify the algorithm from the filename
        if "nsga" in filename:
            algorithm = "NSGA-II"
        elif "spea" in filename:
            algorithm = "SPEA2"
        elif "moead" in filename or "moea_d" in filename or "moea-d" in filename:
            algorithm = "MOEA/D"
        else:
            errors.append(
                f"Could not identify the algorithm in '{file.name}'. "
                "Include NSGA, SPEA, or MOEAD in the filename."
            )
            continue

        try:
            df = pd.read_csv(file)
            df = standardize_results(df)

            missing = required_columns - set(df.columns)

            if missing:
                errors.append(
                    f"{file.name} is missing columns: "
                    f"{', '.join(sorted(missing))}"
                )
                continue

            for col in ["precision", "recall", "f1", "total_conditions"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            algorithm_data[algorithm] = df

        except Exception as e:
            errors.append(f"Error reading {file.name}: {e}")

    for error in errors:
        st.warning(error)

    if algorithm_data:
        st.success(
            f"Successfully loaded {len(algorithm_data)} algorithm(s): "
            + ", ".join(algorithm_data.keys())
        )

        # Choose which algorithm to explore
        st.subheader("Select an Algorithm")

        selected_algorithm = st.selectbox(
            "Algorithm",
            list(algorithm_data.keys())
        )

        data = algorithm_data[selected_algorithm]

        st.subheader("Performance Preferences")

        min_precision = st.slider(
            "Minimum Precision (%)", 0, 100, 50
        )

        min_recall = st.slider(
            "Minimum Recall (%)", 0, 100, 40
        )

        max_conditions = st.slider(
            "Maximum Conditions", 1, 160, 50
        )

        # Filter the selected algorithm's results
        filtered = data[
            (data["precision"] * 100 >= min_precision) &
            (data["recall"] * 100 >= min_recall) &
            (data["total_conditions"] <= max_conditions)
        ]

        st.subheader(f"Matching Rule Sets — {selected_algorithm}")

        display = filtered[
            ["solution", "precision", "recall",
             "f1", "n_rules", "total_conditions"]
        ].copy()

        for col in ["precision", "recall", "f1"]:
            display[col] = (display[col] * 100).round(1)

        st.dataframe(display, use_container_width=True)

        st.write(f"Found {len(filtered)} matching rule sets.")

        if not filtered.empty:
            st.subheader("Inspect a Rule Set")

            # Select by row position so duplicate solution IDs
            # do not cause ambiguous matches.
            selected_row = st.selectbox(
                "Select a solution",
                options=list(range(len(filtered))),
                format_func=lambda i: str(
                    filtered.iloc[i]["solution"]
                )
            )

            rule = filtered.iloc[selected_row]

            st.text(str(rule["rule_set_string"]))

        else:
            st.info("No rule sets match these preferences.")

    else:
        st.error("No valid algorithm result files were loaded.")

else:
    st.info("Upload one or more algorithm result CSV files to begin.")
