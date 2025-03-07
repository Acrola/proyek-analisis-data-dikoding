import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
day_df = pd.read_csv('day_data.csv', parse_dates=['dteday'])
hour_df = pd.read_csv('hour_data.csv', parse_dates=['dteday'])

st.title("Bike Sharing Dashboard 🚲")

# Time filter
st.header("Filter Data by Date")
start_date = st.date_input("Start Date", day_df['dteday'].min().date())
end_date = st.date_input("End Date", day_df['dteday'].max().date())

# Filter dataframes based on selected dates
filtered_day_df = day_df[(day_df['dteday'].dt.date >= start_date) & (day_df['dteday'].dt.date <= end_date)]
filtered_hour_df = hour_df[(hour_df['dteday'].dt.date >= start_date) & (hour_df['dteday'].dt.date <= end_date)]

# Display the average rentals per hour chart
st.header("Average Bike Rentals per Hour")
pivot_table = filtered_hour_df.pivot_table(index='hr', values='cnt', aggfunc='mean')
st.line_chart(pivot_table)

# Display the proportion of rentals on workdays vs. non-workdays chart
st.header("Proportion of Bike Rentals on Workdays vs. Non-Workdays")
workday_rentals = filtered_day_df[filtered_day_df['workingday'] == 1]['cnt'].sum()
non_workday_rentals = filtered_day_df[filtered_day_df['workingday'] == 0]['cnt'].sum()
labels = ['Workday', 'Non-Workday']
sizes = [workday_rentals, non_workday_rentals]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Display the ambient temperature vs. bike rental amount chart
st.header("Ambient Temperature vs. Bike Rentals")
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='temp', y='cnt', data=filtered_day_df, ax=ax)
plt.title('Ambient Temperature vs. Bike Rentals (day)')
plt.xlabel('Ambient Temperature in Celsius')
plt.ylabel('Total Bike Rentals')

# Multiply x-axis values by 50 and update tick labels
x_ticks = plt.xticks()[0]
x_tick_labels = [str(int(tick * 50)) for tick in x_ticks]
plt.xticks(x_ticks, x_tick_labels)

# Calculate and display correlation
correlation = filtered_day_df['temp'].corr(filtered_day_df['cnt'])
correlation_text = f"Correlation: {correlation:.2f}"
plt.annotate(correlation_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)
st.pyplot(fig)