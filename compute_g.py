"""Imports"""

# External imports
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

################################################################################################################################################################
"""Constants"""

g_actual = 9.812                                        # m/s^2 (accepted/reference value for gravitational acceleration)

# l_1 varies per run and is loaded from Data/data.csv instead of being hard-coded here.
d_12 = 7.5                                              # cm  distance between gate 1 and gate 2, fixed for every trial (original l_1 - l_2 = 101.5 - 94)  
l_3 = 1.3                                               # cm

l_1_uncertainty = 0.1                                   # cm  (measuring tape, mm accuracy)
l_2_uncertainty = 0.1                                   # cm  (measuring tape, mm accuracy)
l_3_uncertainty = 0.1                                   # cm  (measuring tape, mm accuracy)

d_12_uncertainty = l_1_uncertainty + l_2_uncertainty    # cm  (d12 = l1 - l2, so uncertainties add)
d_23_uncertainty = l_2_uncertainty + l_3_uncertainty    # cm  (d23 = l2 - l3, so uncertainties add)
d_13_uncertainty = d_12_uncertainty + d_23_uncertainty  # cm  (d13 = d12 - d23, so uncertainties add)

t_uncertainty = 1e-6                                    # s   (single photoelectric sensor reading, microsecond accuracy)
t23_sensor_uncertainty = 2 * t_uncertainty              # s   (t23 = t3 - t2, so the two sensor uncertainties add)

################################################################################################################################################################
"""Functions"""

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
    Propagate uncertainty in d_12, d_13, and t_23 into an uncertainty on g,
    using exact partial-derivative (first-order) error propagation:
 
        u_g^2 = (dg/dd13)^2 * u_d13^2 + (dg/dd12)^2 * u_d12^2 + (dg/dt23)^2 * u_t23^2
 
    Writing A = sqrt(2*d13), B = sqrt(2*d12), so g = (A-B)^2 / t23^2:
 
        dg/dd13 =  2*(A-B) / (A * t23^2)
        dg/dd12 = -2*(A-B) / (B * t23^2)
        dg/dt23 = -2*g / t23
 
    This treats the three input uncertainties as independent (adding their
    contributions in quadrature), unlike a simple relative-error sum, which
    implicitly assumes worst-case correlated errors and overestimates u_g.
 
    t_23             : time between gate 2 and gate 3 (s)
    t23_uncertainty  : uncertainty in t_23 (s)
    d_12             : distance between gate 1 and gate 2 (cm)
    d_13             : distance between gate 1 and gate 3 (cm)
 
    Returns the uncertainty in g, in m/s^2.
    """
    A = math.sqrt(2 * d_13)
    B = math.sqrt(2 * d_12)
    g_cm_per_s2 = (A - B) ** 2 / t_23 ** 2
 
    # Partial derivatives of g (in cm/s^2) with respect to each input
    dg_dd13 = 2 * (A - B) / (A * t_23 ** 2)
    dg_dd12 = -2 * (A - B) / (B * t_23 ** 2)
    dg_dt23 = -2 * g_cm_per_s2 / t_23
 
    # Combine contributions in quadrature (independent-error propagation)
    g_cm_per_s2_uncertainty = math.sqrt(
        (dg_dd13 * d_13_uncertainty) ** 2 +
        (dg_dd12 * d_12_uncertainty) ** 2 +
        (dg_dt23 * t23_uncertainty) ** 2
    )
 
    # Convert cm/s^2 -> m/s^2
    return g_cm_per_s2_uncertainty / 100

def compute_group_results(groups):
    """
    For each l_1 group: average the trial times, compute g and its
    uncertainty, and print a summary row.

    The uncertainty used for t_23 in this function is the STATISTICAL
    spread of the repeated timing measurements at this l_1 (standard error
    of the mean), combined in quadrature with the fixed sensor uncertainty
    (t23_sensor_uncertainty) -- so even a single trial (n=1) still carries
    the sensor's instrumental uncertainty rather than 0.

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
        d_13 = abs(l_1 - l_3)  # distance between gate 1 and gate 3 for this l_1

        # Mean time across all trials at this l_1
        mean_t23 = sum(times) / n

        # Standard error of the mean time (only meaningful with >1 trial)
        if n > 1:
            variance = sum((t - mean_t23) ** 2 for t in times) / (n - 1)
            t23_uncertainty_stat = math.sqrt(variance) / math.sqrt(n)
        else:
            t23_uncertainty_stat = 0.0  # no repeated trials to estimate spread from

        # Total t23 uncertainty: statistical spread and sensor (instrumental)
        # uncertainty combined in quadrature, since they're independent
        t23_uncertainty = math.sqrt(t23_uncertainty_stat ** 2 + t23_sensor_uncertainty ** 2)

        # Compute g and its uncertainty for this group
        g_val = get_g(mean_t23, d_12, d_13)
        g_unc = get_g_uncertainty(mean_t23, t23_uncertainty, d_12, d_13)
        pct_err = abs((g_val - g_actual) / g_actual * 100)

        # Print one row of the summary table
        print(f"{l_1:10.2f} {d_13:10.2f} {n:4d} {g_val:8.4f} +/- {g_unc:.4f} {pct_err:8.2f}")

        group_g.append(g_val)
        group_g_unc.append(g_unc)

    return group_g, group_g_unc

def check_agreement(a, u_a, b, u_b, label_a='a', label_b='b'):
    """
    Check whether two measured values are in good agreement, per the
    agreement criterion:

        |v| = |a - b| > 2*sqrt(u_a^2 + u_b^2) = 2*u_v  =>  NOT in good agreement

    a, u_a            : first value and its uncertainty
    b, u_b            : second value and its uncertainty
    label_a, label_b  : optional names used in the printed message

    Prints the result and returns True if a and b ARE in good agreement,
    False otherwise.
    """
    v = abs(a - b)
    u_v = 2 * np.sqrt(u_a ** 2 + u_b ** 2)
    in_agreement = v <= u_v

    verdict = "ARE in good agreement" if in_agreement else "are NOT in good agreement"
    comparison = "<=" if in_agreement else ">"
    print(f"{label_a} = {a:.4f} +/- {u_a:.4f}  and  {label_b} = {b:.4f} +/- {u_b:.4f}  "
          f"{verdict}  (|v| = {v:.4f} {comparison} 2u_v = {u_v:.4f})")

    return in_agreement


def d23_model(t_23, g_cm_per_s2):
    """
    Model function for curve_fit: predicts d_23 (cm) as a function of t_23,
    for the fixed d_12 spacing, using free-fall kinematics starting from
    rest at point 1 (v1 = 0).

    Coordinate convention: y increases UPWARD (matching how l_1, l_3 were
    actually measured -- l_1 is large/high, l_3 is small/low). Since the
    ball falls downward in this frame, both its velocity and the
    gravitational acceleration are NEGATIVE:

        a_y = -g_cm_per_s2

    v_y2 (signed velocity at point 2) comes from v_y2^2 = 2*g*d_12 (the
    square removes the sign either way), and is negative since the ball is
    moving down:

        v_y2 = -sqrt(2 * g_cm_per_s2 * d_12)

    Position update from point 2 to point 3:

        y3 = y2 + v_y2*t_23 + 0.5*a_y*t_23**2
        y2 - y3 = -(v_y2*t_23 + 0.5*a_y*t_23**2)

    d_23 = y2 - y3 is the positive distance fallen, so:

        d_23 = sqrt(2*g_cm_per_s2*d_12)*t_23 + 0.5*g_cm_per_s2*t_23**2

    t_23        : time between gate 2 and gate 3 (s)
    g_cm_per_s2 : gravitational acceleration magnitude (cm/s^2), the free
                  parameter curve_fit solves for

    Returns predicted d_23 (= y2 - y3, a positive distance) in cm.
    """
    a_y = -g_cm_per_s2                              # acceleration is downward; y increases upward
    v_y2 = -np.sqrt(2 * g_cm_per_s2 * d_12)         # velocity at point 2 (cm/s); negative, moving down
    delta_y = v_y2 * t_23 + 0.5 * a_y * t_23 ** 2   # = y3 - y2 (negative, since the ball fell)
    return -delta_y                                 # d_23 = y2 - y3 = -(y3 - y2), a positive distance

def fit_g_curve_fit(runs):
    """
    Estimate g by pooling every individual trial into a single least-squares
    fit (curve_fit, as introduced in Notebook 5), instead of solving the
    2-point algebraic formula per trial/group and propagating uncertainty
    by hand.

    d_23 = f(t_23; g) is fit directly against ALL trials at once, using the
    signed (y-up, v and g negative) derivation in d23_model(). This makes
    better use of the full dataset than the group-by-group algebraic
    method, and avoids the error amplification that comes from subtracting
    two similar-sized numbers (sqrt(2*d13) - sqrt(2*d12)) in get_g().

    runs : list of (run_id, l_1, time_comp) tuples

    Returns (g_fit, g_fit_uncertainty) in m/s^2.
    """
    all_t23 = np.array([time_comp for run_id, l_1, time_comp in runs])
    all_d23 = np.array([abs(l_1 - l_3) - d_12 for run_id, l_1, time_comp in runs])

    # Least-squares fit of g_cm_per_s2 via curve_fit. p0 is a rough initial
    # guess (in cm/s^2) -- the model is nonlinear in g (it appears under a
    # square root in v_2), so a sensible starting point matters.
    values, covariance = curve_fit(d23_model, all_t23, all_d23, p0=(981.0,))

    # Diagonal of the covariance matrix gives the squared standard error
    # of the fit parameter (see Notebook 5, "Uncertainty in the parameters")
    g_fit_cm_per_s2 = values[0]
    g_fit_uncertainty_cm_per_s2 = np.sqrt(covariance[0, 0])

    # Convert cm/s^2 -> m/s^2
    g_fit = g_fit_cm_per_s2 / 100
    g_fit_uncertainty = g_fit_uncertainty_cm_per_s2 / 100
    pct_err = abs((g_fit - g_actual) / g_actual * 100)

    print(f"g (curve_fit, all {len(runs)} trials pooled) = "
          f"{g_fit:.4f} +/- {g_fit_uncertainty:.4f} m/s^2  ({pct_err:.2f}% error)")

    # Plot the pooled data with the fitted curve on top. A test array is
    # used for a smooth fit line, same approach as Notebook 5.
    t_test = np.linspace(0, 1.1 * max(all_t23), 1000)
    d23_fit = d23_model(t_test, g_fit_cm_per_s2)

    plt.figure()
    plt.plot(all_t23, all_d23, 'k.', ms=4, label='measurements')
    plt.plot(t_test, d23_fit, 'r--', lw=2,
              label=f'fit (g = {g_fit:.3f} $\\pm$ {g_fit_uncertainty:.3f} m/s$^2$)')
    plt.xlabel('t23 (s)')
    plt.ylabel('d23 (cm)')
    plt.title('Global least-squares fit of g (all trials pooled)')
    plt.legend()
    plt.show()

    # Residuals, to sanity-check the fit the same way as Notebook 5
    residuals = all_d23 - d23_model(all_t23, g_fit_cm_per_s2)

    plt.figure()
    plt.plot(all_t23, residuals, 'k.', ms=4)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('t23 (s)')
    plt.ylabel('residual d23 (cm)')
    plt.title('Residuals of the global g fit')
    plt.show()

    return g_fit, g_fit_uncertainty


def plot_g_vs_t23(runs):
    """
    Plot g computed individually for every single trial (not grouped/averaged),
    against t23, with error bars, alongside a reference line for the actual g value.

    runs : list of (run_id, l_1, time_comp) tuples
    """
    all_t23 = []
    all_g = []
    all_g_unc = []

    # Compute g and its uncertainty for every individual trial
    for run_id, l_1, time_comp in runs:
        d_13 = abs(l_1 - l_3)
        g_val = get_g(time_comp, d_12, d_13)
        # Single trial, no averaging -- only the sensor's own timing
        # uncertainty applies (no statistical spread to combine with)
        g_unc = get_g_uncertainty(time_comp, t23_sensor_uncertainty, d_12, d_13)
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

def plot_g_vs_l1(groups, group_g, group_g_unc):
    """
    Plot the averaged g value per l_1 group against l_1 (drop height), to
    check whether apparent g varies with drop height -- in reality this is
    more likely a sign of drag effects than a real change in g.

    groups      : dict mapping l_1 -> list of time_comp values (used to get
                  the sorted l_1 values matching group_g/group_g_unc order)
    group_g     : g value per group (m/s^2), sorted by l_1
    group_g_unc : uncertainty on g per group (m/s^2), sorted by l_1
    """
    l1_values = sorted(groups)

    # Scatter plot of g vs l_1, with vertical error bars
    plt.errorbar(l1_values, group_g, yerr=group_g_unc, fmt='o', capsize=3)
    # Horizontal reference line at the accepted value of g
    plt.axhline(g_actual, color='red', linestyle='--', label='Actual g')

    plt.xlabel('l_1 (cm)')
    plt.ylabel('g (m/s^2)')
    plt.title('Measured g vs drop height (l_1)')
    plt.legend()
    plt.show()

def plot_g_surface(runs=None, d13_range=None, t23_range=None):
    """
    3D surface plot of g as a function of d_13 and t_23, with d_12 held
    fixed -- these are the only two quantities g actually depends on
    (see get_g()), so this shows the whole "response surface" g is drawn
    from rather than a single 1D slice of it.

    If `runs` is given, the actual per-trial (d_13, t23, g) points are
    scattered on top of the surface so you can see where your real data
    sits relative to the full surface.

    runs       : optional list of (run_id, l_1, time_comp) tuples
    d13_range  : optional (min, max) in cm for the d_13 axis; auto-derived
                 from runs if not given (falls back to a default range
                 above d_12 if runs is also not given)
    t23_range  : optional (min, max) in s for the t_23 axis; auto-derived
                 from runs if not given (falls back to a default range)
    """
    import numpy as np

    if runs is not None:
        d13_vals = [abs(l_1 - l_3) for run_id, l_1, time_comp in runs]
        t23_vals = [time_comp for run_id, l_1, time_comp in runs]
    else:
        d13_vals = []
        t23_vals = []

    # d_13 must exceed d_12 (point 3 is further from point 1 than point 2 is)
    if d13_range is None:
        d13_range = (min(d13_vals) * 0.9, max(d13_vals) * 1.1) if d13_vals \
            else (d_12 * 1.05, d_12 * 3)
    if t23_range is None:
        t23_range = (min(t23_vals) * 0.9, max(t23_vals) * 1.1) if t23_vals \
            else (0.01, 0.5)

    d13_grid = np.linspace(*d13_range, 100)
    t23_grid = np.linspace(*t23_range, 100)
    D13, T23 = np.meshgrid(d13_grid, t23_grid)

    # Vectorized version of get_g()'s formula, with d_12 fixed
    G = (np.sqrt(2 * D13) - np.sqrt(2 * d_12)) ** 2 / T23 ** 2 / 100  # m/s^2

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(D13, T23, G, cmap='viridis', alpha=0.75, edgecolor='none')
    fig.colorbar(surf, shrink=0.6, label='g (m/s^2)')

    if runs is not None:
        g_vals = [get_g(t23, d_12, d13) for t23, d13 in zip(t23_vals, d13_vals)]
        ax.scatter(d13_vals, t23_vals, g_vals, color='red', s=25, label='measured trials')
        ax.legend()

    ax.set_xlabel('d_13 (cm)')
    ax.set_ylabel('t_23 (s)')
    ax.set_zlabel('g (m/s^2)')
    ax.set_title('g as a function of d_13 and t_23 (d_12 fixed)')

    plt.tight_layout()
    plt.show()

################################################################################################################################################################
"""Main"""

def main():
    """
    Run the full analysis pipeline:
    load data -> group by distance -> compute g per group (algebraic method)
    -> compute g via a pooled least-squares fit -> plot results.
    """
    
    runs = load_data('Data/data.csv')
    groups = group_by_l1(runs)
    group_g, group_g_unc = compute_group_results(groups)
    g_fit, g_fit_unc = fit_g_curve_fit(runs)
    check_agreement(g_fit, g_fit_unc, g_actual, 0.0, 'g_fit', 'g_actual')
    # plot_g_vs_t23(runs)
    # plot_g_vs_l1(groups, group_g, group_g_unc)
    # plot_g_surface(runs)

if __name__ == '__main__':
    main()
