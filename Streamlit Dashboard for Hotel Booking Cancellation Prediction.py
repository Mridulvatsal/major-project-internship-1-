import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page title
st.title("🏨 Hotel Booking Cancellation Prediction Dashboard")

# Load sample data (replace with your actual data)
# For demo purposes, creating a small DataFrame based on your outputs
data = {
    "is_canceled": [0] * 62947 + [1] * 23917,  # From your cancellation rate
    "lead_time": [50, 100, 150, 200, 250] * 17173 + [300] * 3,  # Sample lead time data
    "arrival_date_month": ["April"] * 8000 + ["August"] * 10000 + ["December"] * 9000 + ["February"] * 7000 +
                          ["January"] * 6000 + ["July"] * 8500 + ["June"] * 7500 + ["March"] * 8800 +
                          ["May"] * 7200 + ["November"] * 6800 + ["October"] * 6500 + ["September"] * 6700
}
df = pd.DataFrame(data)

# Cancellation Rate
st.header("Cancellation Rate Overview")
cancellation_rate = 27.6  # From your output
st.markdown(f"**Cancellation Rate**: {cancellation_rate}%")

# Bookings by Month
st.header("Bookings by Month")
monthly_bookings = df['arrival_date_month'].value_counts().sort_index()
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(x=monthly_bookings.index, y=monthly_bookings.values, ax=ax1)
plt.xticks(rotation=45)
plt.title("Bookings by Month")
st.pyplot(fig1)

# Lead Time vs Cancellation (Simulated Boxplot)
st.header("Lead Time vs Cancellation")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.boxplot(x="is_canceled", y="lead_time", data=df, ax=ax2)
plt.title("Lead Time vs Cancellation")
st.pyplot(fig2)

# Correlation Heatmap (Simulated with Sample Data)
st.header("Correlation Heatmap")
# Sample correlation matrix based on your heatmap
corr_data = {
    "is_canceled": [1.00, 0.18, -0.13],
    "lead_time": [0.18, 1.00, 0.02],
    "adr": [-0.13, 0.02, 1.00]
}
corr_df = pd.DataFrame(corr_data, index=["is_canceled", "lead_time", "adr"])
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
plt.title("Correlation Heatmap")
st.pyplot(fig3)

# ROC Curve (Simulated)
st.header("ROC Curve")
fpr = np.linspace(0, 1, 100)
tpr = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0])  # Simulated based on AUC = 0.72
fig4, ax4 = plt.subplots()
ax4.plot(fpr, tpr, label="AUC = 0.72")
ax4.plot([0, 1], [0, 1], "k--")
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.set_title("ROC Curve")
ax4.legend()
st.pyplot(fig4)

# Revenue Loss by Month
st.header("Revenue Loss by Month")
revenue_loss = {
    "Month": ["April", "August", "December", "February", "January", "July", "June", "March", "May", "November", "October", "September"],
    "Estimated Loss ($)": [82697.32, 248037.98, 254941.78, 109776.94, 75666.42, 177468.54, 110373.75, 281113.67, 104870.61, 100332.30, 43095.91, 64619.75]
}
revenue_df = pd.DataFrame(revenue_loss)
st.table(revenue_df)

# Bar Chart for Revenue Loss
fig5, ax5 = plt.subplots(figsize=(10, 5))
sns.barplot(x="Month", y="Estimated Loss ($)", data=revenue_df, ax=ax5)
plt.xticks(rotation=45)
plt.title("Revenue Loss by Month")
st.pyplot(fig5)

st.markdown("**Total Expected Revenue Loss**: $913,254.96")