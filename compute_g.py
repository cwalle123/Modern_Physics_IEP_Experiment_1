import numpy as np
import constants as const

g_actual = 9.812  # m/s^2

# Load the data
run, time_human, time_comp = np.loadtxt('Data/data.csv', delimiter=',', skiprows=1, unpack=True)

N = len(time_comp)
average_time_comp = np.mean(time_comp)
t23_uncertainty = np.std(time_comp, ddof=1) / np.sqrt(N)  # standard error of the mean

t_23 = average_time_comp

# --- g in cm/s^2, then converted to m/s^2 ---
g_cm_per_s2 = (np.sqrt(2 * const.d_13) - np.sqrt(2 * const.d_12))**2 / t_23**2
g = g_cm_per_s2 / 100

# --- Uncertainty propagation (linear addition, matching constants.py style) ---
g_cm_per_s2_uncertainty = g_cm_per_s2 * (
    (const.d_13_uncertainty / const.d_13 + const.d_12_uncertainty / const.d_12)
    + 2 * (t23_uncertainty / t_23))
g_uncertainty = g_cm_per_s2_uncertainty / 100

print(f"Measured g: {g:.3f} +/- {g_uncertainty:.3f} m/s^2")
print(f"Actual g: {g_actual} m/s^2")
print(f"Percentage Error: {abs((g - g_actual)/g_actual * 100):.2f}%")
