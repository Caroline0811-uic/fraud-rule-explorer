
import streamlit as st
import pandas as pd

st.title("TransUnion Fraud Rule Explorer")
st.write("Explore optimized fraud detection rule sets.")

uploaded_file = st.file_uploader(
    "Upload algorithm results",
    type="csv"
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

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

    filtered = data[
        (data["precision"] * 100 >= min_precision) &
        (data["recall"] * 100 >= min_recall) &
        (data["total_conditions"] <= max_conditions)
    ]

    st.subheader("Matching Rule Sets")

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

        selected = st.selectbox(
            "Select a solution",
            filtered["solution"].tolist()
        )

        rule = filtered[
            filtered["solution"] == selected
        ].iloc[0]

        st.text(rule["rule_set_string"])
    else:
        st.info("No rule sets match these preferences.")
