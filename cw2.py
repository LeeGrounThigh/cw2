import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Results_21MAR2022_nokcaladjust.csv")

water_columns = ["mean_watscar", "mean_watuse", "mean_eut"]
grouped = df.groupby("diet_group")[water_columns].mean().reset_index()

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(grouped[water_columns])
normalized_df = pd.DataFrame(normalized_data, columns=["Water Scarcity", "Water Use", "Eutrophication"])
normalized_df["DietGroup"] = grouped["diet_group"]

plot_data = normalized_df[["DietGroup", "Water Scarcity", "Water Use", "Eutrophication"]]

custom_palette = ["#007849", "#79C000", "#F7C100", "#EE3E38", "#763568"]

plt.figure(figsize=(10, 6))
parallel_coordinates(
    plot_data,
    class_column="DietGroup",
    cols=["Water Scarcity", "Water Use", "Eutrophication"],
    color=custom_palette  # 颜色清晰
)
plt.title(" Parallel Coordinates  Mean Water-related Impact by Diet")
plt.ylabel("Normalized Value")
plt.grid(True)
plt.tight_layout()
plt.show()
