import numpy as np
import constants as const
import matplotlib.pyplot as plt

g_actual = 9.812  # m/s^2

# Load the data
run, time_human, time_comp = np.loadtxt('Data/data.csv', delimiter=',', skiprows=1, unpack=True)

N = len(time_comp)
average_time_comp = np.mean(time_comp)
t23_uncertainty = np.std(time_comp, ddof=1) / np.sqrt(N)  # standard error of the mean

t_23 = average_time_comp

# --- g in cm/s^2, then converted to m/s^2 ---
def get_g(t_23):
    g_cm_per_s2 = (np.sqrt(2 * const.d_13) - np.sqrt(2 * const.d_12))**2 / t_23**2
    return g_cm_per_s2 / 100

g = get_g(t_23)

# --- Uncertainty propagation ---
def get_g_uncertainty(t_23, t23_uncertainty):
    g_cm_per_s2 = (np.sqrt(2 * const.d_13) - np.sqrt(2 * const.d_12))**2 / t_23**2
    g_cm_per_s2_uncertainty = g_cm_per_s2 * (
        (const.d_13_uncertainty / const.d_13 + const.d_12_uncertainty / const.d_12)
        + 2 * (t23_uncertainty / t_23))
    return g_cm_per_s2_uncertainty / 100

g_uncertainty = get_g_uncertainty(t_23, t23_uncertainty)

print(f"Measured g: {g:.3f} +/- {g_uncertainty:.3f} m/s^2")
print(f"Actual g: {g_actual} m/s^2")
print(f"Percentage Error: {abs((g - g_actual)/g_actual * 100):.2f}%")



# --- g computed individually for every measured t23, not just the average ---
g_per_run = get_g(time_comp)
g_uncertainty_per_run = get_g_uncertainty(time_comp, t23_uncertainty)

fig, ax = plt.subplots()

ax.errorbar(time_comp, g_per_run, yerr=g_uncertainty_per_run,
            fmt='o', capsize=3, label='Measured g per run')

ax.axhline(g_actual, color='red', linestyle='--', label='Actual g')
ax.axhline(g, color='green', linestyle=':', label='Mean measured g')

ax.set_xlabel('t23 (s)')
ax.set_ylabel('g (m/s$^2$)')
ax.set_title('Measured g vs. t23 for each run')
ax.legend()

plt.show()
