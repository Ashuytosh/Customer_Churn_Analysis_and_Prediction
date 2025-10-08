# =========================
# Customer Churn Analysis Dashboard
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Page Configuration
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

# -------------------------
# Load Datasets
# -------------------------
df = pd.read_csv("Dataset/churn_modified.csv")
df_model = pd.read_csv("Dataset/df_model.csv")
feature_scores = pd.read_csv("Dataset/feature_scores.csv")

# -------------------------
# Page Title
# -------------------------
st.title("📊 Customer Churn Analysis and Prediction")
st.markdown("""
This dashboard presents **EDA insights**, **feature engineering results**, and **model performance comparison**
for the telecom customer churn prediction project.
""")

# -------------------------
# 1️⃣ Dataset Overview
# -------------------------
st.header("1️⃣ Dataset Overview")

col1, spacer, col2 = st.columns([1, 0.1, 1])


with col1:
    st.subheader("Original Dataset (Before Feature Engineering)")
    st.dataframe(df.head(5), use_container_width=True)
    st.caption(f"Shape: {df.shape}")

with col2:
    st.markdown("<h1 style='text-align:center;'>➡️</h1>", unsafe_allow_html=True)

with col2:
    pass  # empty space for arrow only

with col2:
    st.subheader("Transformed Dataset (After Feature Engineering)")
    st.dataframe(df_model.head(5), use_container_width=True)
    st.caption(f"Shape: {df_model.shape}")

st.markdown("---")

# -------------------------
# 2️⃣ Insights from Data
# -------------------------
st.header("2️⃣ Insights from Data")

# --- Row 1: Distribution & Boxplots ---
st.subheader("Distributions and Charges Overview")

colA, colB = st.columns(2)

with colA:
    st.markdown("**Distribution of Numeric Features**")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Create a grid of subplots manually
    n_cols = 3  # number of columns in the grid
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))

    axes = axes.flatten()  # flatten for easy iteration

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    # Turn off any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=2.0)
    st.pyplot(fig)


with colB:
    st.markdown("**Boxplots for Charges**")
    charge_cols = ['Day Charge', 'Eve Charge', 'Night Charge', "Int'l Charge"]
    available_cols = [c for c in charge_cols if c in df.columns]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    for i, col in enumerate(available_cols):
        sns.boxplot(y=df[col], ax=axs[i], color='skyblue')
        axs[i].set_title(col)
    for j in range(len(available_cols), 4):
        axs[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

# --- Row 2: Churn Insights ---
st.subheader("Customer Churn Insights")

colC, colD = st.columns(2)

with colC:
    st.markdown("**Area Code vs Churn (Violin Plot)**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(x='Area Code', y='Day Mins', hue='Churn', data=df, palette='coolwarm')
    ax.set_title("Area Code vs Churn (Day Mins)")
    st.pyplot(fig)

with colD:
    st.markdown("**Proportion of Churn Rate by State**")
    fig, ax = plt.subplots(figsize=(10, 4))
    crosstab = pd.crosstab(df['State'], df['Churn'], normalize='index')
    crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
    ax.set_ylabel("Proportion")
    ax.set_title("Churn Rate by State")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

# --- Row 3: Correlation & Feature Importance ---
st.subheader("Correlation & Feature Importance")

colE, colF = st.columns(2)

with colE:
    st.markdown("**Correlation Heatmap (Numerical Features)**")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="OrRd", linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

with colF:
    st.markdown("**Top 5 Features Contributing to Customer Churn**")
    top_features = feature_scores.head(5)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=top_features, x='Score', y='Feature', palette='viridis', ax=ax)
    ax.set_xlabel("Feature Importance Score (ANOVA F-value)")
    ax.set_ylabel("Feature")
    ax.set_title("Top 5 Important Features")
    st.pyplot(fig)

# --- Row 4: Service & Intl Plan ---
st.subheader("Service and International Plan Insights")

colG, colH = st.columns(2)

with colG:
    st.markdown("**Customer Service Calls vs Churn**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='CustServ Calls', hue='Churn', palette=['#32CD32', '#FFD700'], ax=ax)
    ax.set_title("Customer Service Calls vs Churn")
    ax.set_xlabel("Number of Calls")
    ax.set_ylabel("Number of Customers")
    ax.legend(title="Churn", labels=["No", "Yes"])
    st.pyplot(fig)

with colH:
    st.markdown("**Churned Customers with International Plan**")
    churned = df_model[df_model['Churn'] == 1]
    intl_churn_counts = churned["Int'l Plan"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        intl_churn_counts,
        labels=None,
        autopct='%1.1f%%',
        colors=['skyblue', 'yellow'],
        startangle=140,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )
    ax.legend(wedges, ["0 → No", "1 → Yes"], title="Int'l Plan", loc="best")
    ax.set_title("Churned Customers with International Plan")
    st.pyplot(fig)

st.markdown("---")

# -------------------------
# 3️⃣ Model Performance Summary
# -------------------------
st.header("3️⃣ Model Performance Summary")

results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting", "KNN", "SVM"],
    "Accuracy": [0.7166, 0.9070, 0.9190, 0.8291, 0.8591],
    "Precision": [0.2982, 0.6636, 0.7087, 0.4331, 0.5143],
    "Recall": [0.7010, 0.7320, 0.7526, 0.5670, 0.5567],
    "F1-Score": [0.4185, 0.6961, 0.7300, 0.4911, 0.5347]
})

st.dataframe(results_df, use_container_width=True)

st.markdown("---")
st.markdown("Developed by: **Ashutosh Sahoo**")
st.markdown("GitHub Repository: [Click Here](https://github.com/Ashuytosh/Customer_Churn_Analysis_and_Prediction) 🔗")
