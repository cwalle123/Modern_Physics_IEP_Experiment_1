"""Imports"""

# External imports
import csv
import math
import matplotlib.pyplot as plt

# Internal imports
import constants

################################################################################################################################################################
"""Constants"""

g_actual = 9.812  # m/s^2 (accepted/reference value for gravitational acceleration)
d_12 = constants.d_12  # distance between gate 1 and gate 2, fixed for every trial
d_12_uncertainty = constants.d_12_uncertainty  # uncertainty in d_12
d_13_uncertainty = constants.d_13_uncertainty  # uncertainty in d_13

################################################################################################################################################################
"""Functions"""

def get_g(t_23, d_12, d_13):
    """
    Compute g from the timing between gate 2 and gate 3, and the two
    known distances (d_12, d_13).

    t_23  : time between gate 2 and gate 3 (s)
    d_12  : distance between gate 1 and gate 2 (cm)
    d_13  : distance between gate 1 and gate 3 (cm)

    Returns g in m/s^2.
    """
    # Formula derived from constant acceleration kinematics, gives g in cm/s^2
    g_cm_per_s2 = (math.sqrt(2 * d_13) - math.sqrt(2 * d_12)) ** 2 / t_23 ** 2
    # Convert cm/s^2 -> m/s^2
    return g_cm_per_s2 / 100

def get_g_uncertainty(t_23, t23_uncertainty, d_12, d_13):
    """
    Propagate uncertainty in d_12, d_13, and t_23 into an uncertainty on g.

    t_23             : time between gate 2 and gate 3 (s)
    t23_uncertainty  : uncertainty in t_23 (s)
    d_12             : distance between gate 1 and gate 2 (cm)
    d_13             : distance between gate 1 and gate 3 (cm)

    Returns the uncertainty in g, in m/s^2.
    """
    # Recompute g in cm/s^2 (needed as a base value for relative-error propagation)
    g_cm_per_s2 = (math.sqrt(2 * d_13) - math.sqrt(2 * d_12)) ** 2 / t_23 ** 2

    # Combine relative uncertainties from each input (standard error propagation)
    g_cm_per_s2_uncertainty = g_cm_per_s2 * (
        (d_13_uncertainty / d_13 + d_12_uncertainty / d_12)
        + 2 * (t23_uncertainty / t_23))

    # Convert cm/s^2 -> m/s^2
    return g_cm_per_s2_uncertainty / 100

def load_data(filepath):
    """
    Read the raw trial data from a CSV file.

    Expected columns: run, l_1, t23 (with a header row to skip).

    Returns a list of tuples: (run_id, l_1, time_comp)
    """
    runs = []
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            if not row:  # skip blank lines (e.g. trailing newline at end of file)
                continue
            run_id, l_1, time_comp = row
            # Convert all values from strings to floats before storing
            runs.append((float(run_id), float(l_1), float(time_comp)))
    return runs

def group_by_l1(runs):
    """
    Group trial timing data by l_1 (i.e. by which distance setting was used).

    runs : list of (run_id, l_1, time_comp) tuples

    Returns a dict mapping l_1 -> list of time_comp values for that l_1.
    """
    groups = {}
    for run_id, l_1, time_comp in runs:
        # setdefault creates an empty list the first time this l_1 is seen
        groups.setdefault(l_1, []).append(time_comp)
    return groups

def compute_group_results(groups):
    """
    For each l_1 group: average the trial times, compute g and its
    uncertainty, and print a summary row.

    groups : dict mapping l_1 -> list of time_comp values

    Returns two lists (same order, sorted by l_1):
      group_g     : g value per group (m/s^2)
      group_g_unc : uncertainty on g per group (m/s^2)
    """
    group_g = []
    group_g_unc = []

    # Print table header
    print(f"{'l_1 (cm)':>10} {'d_13 (cm)':>10} {'N':>4} {'g (m/s^2)':>14} {'% error':>8}")

    for l_1 in sorted(groups):
        times = groups[l_1]
        n = len(times)
        d_13 = abs(l_1 - constants.l_3)  # distance between gate 1 and gate 3 for this l_1

        # Mean time across all trials at this l_1
        mean_t23 = sum(times) / n

        # Standard error of the mean time (only meaningful with >1 trial)
        if n > 1:
            variance = sum((t - mean_t23) ** 2 for t in times) / (n - 1)
            t23_uncertainty = math.sqrt(variance) / math.sqrt(n)
        else:
            t23_uncertainty = 0.0

        # Compute g and its uncertainty for this group
        g_val = get_g(mean_t23, d_12, d_13)
        g_unc = get_g_uncertainty(mean_t23, t23_uncertainty, d_12, d_13)
        pct_err = abs((g_val - g_actual) / g_actual * 100)

        # Print one row of the summary table
        print(f"{l_1:10.2f} {d_13:10.2f} {n:4d} {g_val:8.4f} +/- {g_unc:.4f} {pct_err:8.2f}")

        group_g.append(g_val)
        group_g_unc.append(g_unc)

    return group_g, group_g_unc

def compute_overall_g(group_g, group_g_unc):
    """
    Combine the per-group g values into a single overall estimate,
    using inverse-variance weighting (groups with smaller uncertainty
    count more).

    group_g     : list of g values per group (m/s^2)
    group_g_unc : list of uncertainties per group (m/s^2)

    Returns the overall weighted mean g (m/s^2), and prints a summary.
    """
    # Weight each group by 1 / uncertainty^2 (inverse-variance weighting)
    weights = [1 / unc ** 2 for unc in group_g_unc]
    overall_g = sum(g * w for g, w in zip(group_g, weights)) / sum(weights)

    print(f"\nOverall weighted mean g: {overall_g:.4f} m/s^2")
    print(f"Actual g: {g_actual} m/s^2")
    print(f"Percentage Error: {abs((overall_g - g_actual) / g_actual * 100):.2f}%")

    return overall_g

def plot_g_per_run(runs):
    """
    Plot g computed individually for every single trial (not grouped/averaged),
    with error bars, alongside a reference line for the actual g value.

    runs : list of (run_id, l_1, time_comp) tuples
    """
    all_t23 = []
    all_g = []
    all_g_unc = []

    # Compute g and its uncertainty for every individual trial
    for run_id, l_1, time_comp in runs:
        d_13 = abs(l_1 - constants.l_3)
        g_val = get_g(time_comp, d_12, d_13)
        g_unc = get_g_uncertainty(time_comp, 0.0, d_12, d_13)  # no averaging, so t23 uncertainty = 0
        all_t23.append(time_comp)
        all_g.append(g_val)
        all_g_unc.append(g_unc)

    # Scatter plot of g vs t23, with vertical error bars
    plt.errorbar(all_t23, all_g, yerr=all_g_unc, fmt='o', capsize=3)
    # Horizontal reference line at the accepted value of g
    plt.axhline(g_actual, color='red', linestyle='--', label='Actual g')

    plt.xlabel('t23 (s)')
    plt.ylabel('g (m/s^2)')
    plt.title('Measured g vs t23')
    plt.legend()
    plt.show()

################################################################################################################################################################
"""Main"""

def main():
    """
    Run the full analysis pipeline:
    load data -> group by distance -> compute g per group -> compute overall g -> plot results.
    """
    
    runs = load_data('Data/data.csv')
    groups = group_by_l1(runs)
    group_g, group_g_unc = compute_group_results(groups)
    compute_overall_g(group_g, group_g_unc)
    plot_g_per_run(runs)

if __name__ == '__main__':
    main()