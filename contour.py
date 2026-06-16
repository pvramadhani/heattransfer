import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Step 1: Load Excel file
df = pd.read_excel("Temperature at 30W.xlsx")   # make sure the file is in the same folder
df = df.dropna()                           # remove rows with missing values

# Step 2: Extract columns
x = df.iloc[:,0].values   # first column = x
y = df.iloc[:,1].values   # second column = y
T = df.iloc[:,2].values   # third column = temperature

# Step 3: Create grid for contour plotting
xi = np.linspace(x.min(), x.max(), 2000)
yi = np.linspace(y.min(), y.max(), 200)
Xi, Yi = np.meshgrid(xi, yi)

# Step 4: Interpolate scattered data onto grid
Ti = griddata((x, y), T, (Xi, Yi), method='linear')  # try 'linear' first

# Step 5: Plot contour
plt.figure(figsize=(370,6))
contour = plt.contourf(Xi, Yi, Ti, levels=20, cmap='jet')
plt.colorbar(contour, label="Temperature (T)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Temperature Contour Plot")
plt.gca().set_aspect('equal', adjustable='box')   # ensures 1 unit in x = 1 unit in y
plt.show()

# Optional: Save the plot as an image
# plt.savefig("contour_plot.png", dpi=300)
