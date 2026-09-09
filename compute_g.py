import csv
import math
import constants as const

g_actual = 9.812  # m/s^2

d_12 = const.d_12
d_12_uncertainty = const.d_12_uncertainty
d_13_uncertainty = const.d_13_uncertainty


def get_g(t_23, d_12, d_13):
    g_cm_per_s2 = (math.sqrt(2 * d_13) - math.sqrt(2 * d_12)) ** 2 / t_23 ** 2
    return g_cm_per_s2 / 100


def get_g_uncertainty(t_23, t23_uncertainty, d_12, d_13):
    g_cm_per_s2 = (math.sqrt(2 * d_13) - math.sqrt(2 * d_12)) ** 2 / t_23 ** 2
    g_cm_per_s2_uncertainty = g_cm_per_s2 * (
        (d_13_uncertainty / d_13 + d_12_uncertainty / d_12)
        + 2 * (t23_uncertainty / t_23))
    return g_cm_per_s2_uncertainty / 100


# --- Load data ---
runs = []
with open('Data/data.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if not row:  # skip blank lines
            continue
        run_id, l_1, time_comp = row
        runs.append((float(run_id), float(l_1), float(time_comp)))

# --- Group trials by l_1 ---
groups = {}  # l_1 -> list of time_comp values
for run_id, l_1, time_comp in runs:
    groups.setdefault(l_1, []).append(time_comp)

# --- Compute g for each group ---
group_g = []
group_g_unc = []

print(f"{'l_1 (cm)':>10} {'d_13 (cm)':>10} {'N':>4} {'g (m/s^2)':>14} {'% error':>8}")
for l_1 in sorted(groups):
    times = groups[l_1]
    n = len(times)
    d_13 = abs(l_1 - const.l_3)

    mean_t23 = sum(times) / n

    if n > 1:
        variance = sum((t - mean_t23) ** 2 for t in times) / (n - 1)
        t23_uncertainty = math.sqrt(variance) / math.sqrt(n)
    else:
        t23_uncertainty = 0.0

    g_val = get_g(mean_t23, d_12, d_13)
    g_unc = get_g_uncertainty(mean_t23, t23_uncertainty, d_12, d_13)
    pct_err = abs((g_val - g_actual) / g_actual * 100)

    print(f"{l_1:10.2f} {d_13:10.2f} {n:4d} {g_val:8.4f} +/- {g_unc:.4f} {pct_err:8.2f}")

    group_g.append(g_val)
    group_g_unc.append(g_unc)

# --- Overall weighted mean ---
weights = [1 / unc ** 2 for unc in group_g_unc]
overall_g = sum(g * w for g, w in zip(group_g, weights)) / sum(weights)

print(f"\nOverall weighted mean g: {overall_g:.4f} m/s^2")
print(f"Actual g: {g_actual} m/s^2")
print(f"Percentage Error: {abs((overall_g - g_actual) / g_actual * 100):.2f}%")

#####

import matplotlib.pyplot as plt

# --- Simple plot: g per run, with error bars and actual g line ---
all_t23 = []
all_g = []
all_g_unc = []

for run_id, l_1, time_comp in runs:
    d_13 = abs(l_1 - const.l_3)
    g_val = get_g(time_comp, d_12, d_13)
    g_unc = get_g_uncertainty(time_comp, 0.0, d_12, d_13)
    all_t23.append(time_comp)
    all_g.append(g_val)
    all_g_unc.append(g_unc)

plt.errorbar(all_t23, all_g, yerr=all_g_unc, fmt='o', capsize=3)
plt.axhline(g_actual, color='red', linestyle='--', label='Actual g')
plt.xlabel('t23 (s)')
plt.ylabel('g (m/s^2)')
plt.title('Measured g vs t23')
plt.legend()
plt.show()