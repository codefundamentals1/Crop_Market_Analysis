# import matplotlib.pyplot as plt
# import seaborn as sns

# # Data
# previous_x = ["Mar 24", "Apr 24", "May 24", "Jun 24", "Jul 24", "Aug 24", "Sep 24", "Oct 24", "Nov 24", "Dec 24", "Jan 25", "Feb 25"]
# previous_y = [2645.82, 2686.78, 2689.05, 2677.68, 2650.38, 2654.92, 2666.3, 2693.6, 2707.25, 2693.6, 2482.02, 2516.15]

# forecast_x = ["Apr 25", "May 25", "Jun 25", "Jul 25", "Aug 25", "Sep 25", "Oct 25", "Nov 25", "Dec 25", "Jan 26", "Feb 26", "Mar 26"]
# forecast_y = [3942.58, 3960.78, 3908.45, 3949.4, 4108.65, 4135.95, 4215.58, 4377.1, 4509.05, 3744.65, 3778.78, 3942.58]

# # Line plot for price trends
# plt.figure(figsize=(12,6))
# sns.lineplot(x=previous_x, y=previous_y, marker='o', label="Previous Prices", color="blue")
# sns.lineplot(x=forecast_x, y=forecast_y, marker='s', label="Forecasted Prices", color="red")
# plt.xticks(rotation=45)
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.title("Wheat Price Trends (Past and Forecast)")
# plt.legend()
# plt.grid(True)
# plt.show()
