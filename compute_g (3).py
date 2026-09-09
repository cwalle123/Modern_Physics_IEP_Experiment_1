import numpy as np
import constants as const
import matplotlib.pyplot as plt

g_actual = 9.812  # m/s^2

# --- Load timing + distance data (one row per trial, one file) ---
run, l_1, time_comp = np.loadtxt('Data/data.csv', delimiter=',', skiprows=1, unpack=True)

# --- Distances ---
# d_12 is fixed: gate 2 moves with gate 1, keeping the same separation.
# d_13 varies per run since gate 3 is fixed while gate 1 (and l_1) moves.
d_12 = const.d_12
d_13 = np.abs(l_1 - const.l_3)
d_12_uncertainty = const.d_12_uncertainty
d_13_uncertainty = const.d_13_uncertainty


# --- g in cm/s^2, then converted to m/s^2 ---
def get_g(t_23, d_12, d_13):
    g_cm_per_s2 = (np.sqrt(2 * d_13) - np.sqrt(2 * d_12))**2 / t_23**2
    return g_cm_per_s2 / 100


# --- Uncertainty propagation ---
def get_g_uncertainty(t_23, t23_uncertainty, d_12, d_13):
    g_cm_per_s2 = (np.sqrt(2 * d_13) - np.sqrt(2 * d_12))**2 / t_23**2
    g_cm_per_s2_uncertainty = g_cm_per_s2 * (
        (d_13_uncertainty / d_13 + d_12_uncertainty / d_12)
        + 2 * (t23_uncertainty / t_23))
    return g_cm_per_s2_uncertainty / 100


# --- g computed individually for every trial ---
g_per_run = get_g(time_comp, d_12, d_13)
g_uncertainty_per_run = get_g_uncertainty(time_comp, 0.0, d_12, d_13)

# --- Group trials by distance (same l_1) and average within each group ---
unique_l1 = np.unique(l_1)

group_d13, group_g, group_g_unc = [], [], []

print(f"{'l_1 (cm)':>10} {'d_13 (cm)':>10} {'N':>4} {'g (m/s^2)':>14} {'% error':>8}")
for val in unique_l1:
    mask = (l_1 == val)
    n = int(mask.sum())

    mean_t23 = np.mean(time_comp[mask])
    t23_uncertainty = np.std(time_comp[mask], ddof=1) / np.sqrt(n) if n > 1 else 0.0

    d12_val = d_12  # same fixed value for every run
    d13_val = d_13[mask][0]

    g_val = get_g(mean_t23, d12_val, d13_val)
    g_unc = get_g_uncertainty(mean_t23, t23_uncertainty, d12_val, d13_val)
    pct_err = abs((g_val - g_actual) / g_actual * 100)

    print(f"{val:10.2f} {d13_val:10.2f} {n:4d} {g_val:8.4f} +/- {g_unc:.4f} {pct_err:8.2f}")

    group_d13.append(d13_val)
    group_g.append(g_val)
    group_g_unc.append(g_unc)

group_d13 = np.array(group_d13)
group_g = np.array(group_g)
group_g_unc = np.array(group_g_unc)

overall_g = np.average(group_g, weights=1 / group_g_unc**2)
print(f"\nOverall weighted mean g: {overall_g:.4f} m/s^2")
print(f"Actual g: {g_actual} m/s^2")
print(f"Percentage Error: {abs((overall_g - g_actual)/g_actual * 100):.2f}%")


# --- Plot: g vs. t23, colored by distance (l_1) ---
fig, ax = plt.subplots()

colors = plt.cm.tab10(np.linspace(0, 1, len(unique_l1)))

for val, color in zip(unique_l1, colors):
    mask = (l_1 == val)
    d13_val = d_13[mask][0]
    ax.errorbar(time_comp[mask], g_per_run[mask], yerr=g_uncertainty_per_run[mask],
                fmt='o', capsize=3, color=color, label=f'l1={val:.1f} cm (d13={d13_val:.1f} cm)')

ax.axhline(g_actual, color='red', linestyle='--', label='Actual g')

ax.set_xlabel('t23 (s)')
ax.set_ylabel('g (m/s$^2$)')
ax.set_title('Measured g vs. t23, by distance')
ax.legend()

plt.show()
