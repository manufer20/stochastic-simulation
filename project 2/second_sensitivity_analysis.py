import random
import heapq
from scipy.stats import t
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from math import log, sqrt


########################################################################
#  Global RNG seed – set *once* so that replications differ but remain
#  reproducible across programme runs.  Delete or vary the number below
#  to obtain a fresh experiment.
########################################################################
np.random.seed(42)

# 1. BASIC SIMULATION (EVENT–DRIVEN)

# Basic simulation 
#Determine if F should be included or not. Write FALSE or TRUE
INCLUDE_F = False  # Set to True to include Ward F
WARDS = ['A', 'B', 'C', 'D', 'E'] + (['F'] if INCLUDE_F else [])

# Given Parameters

arrival_rates = {'A': 14.5, 'B': 11.0, 'C': 8.0, 'D': 6.5, 'E': 5.0, 'F': 13.0}
stay_means = {'A': 2.9, 'B': 4.0, 'C': 4.5, 'D': 1.4, 'E': 3.9, 'F': 2.2}

# === Sensitivity toggles =================================================
CURRENT_DIST = "exp"    # "exp"  or "lognormal"
VAR_MULT     = 1.0      # if lognormal: 2/µ², 3/µ², 4/µ² → set 2,3,4 here
# ========================================================================

# ------------------------------------------------------------------
# Helper: draw a single length‑of‑stay for *one* patient
# ------------------------------------------------------------------
def sample_los(ptype, distribution="exp", var_mult=1.0):
    """
    Parameters
    ----------
    ptype        : ward / patient type, e.g. 'A'
    distribution : "exp" (baseline) or "lognormal"
    var_mult     : for log‑normal, factor k so that  Var = k / µ_i²  (k ≥ 1)

    Returns
    -------
    los : float  (length‑of‑stay in days)
    """
    μ = stay_means[ptype]  # target mean

    if distribution == "exp":
        return np.random.exponential(μ)

    # --- log‑normal with same mean, inflated variance ---------------
    # We want   E[X] = μ,   Var[X] = k / μ²         (k = var_mult)
    σ2 = var_mult / μ**2            # target variance
    # For a log‑normal  X ~ LN( m, s² ):
    #   mean = exp(m + s²/2)
    #   var  = (exp(s²) - 1) * exp(2m + s²)
    # Solve for m, s²:
    s2 = np.log(σ2 / μ**2 + 1)
    m  = np.log(μ) - 0.5 * s2
    return np.random.lognormal(mean=m, sigma=np.sqrt(s2))


urgency_points = {'A': 7, 'B': 5, 'C': 2, 'D': 10, 'E': 5, 'F': 0}
initial_beds = {'A': 55, 'B': 40, 'C': 30, 'D': 20, 'E': 20, 'F': 0}

relocation_probs = {
    'A': {'B': 0.05, 'C': 0.10, 'D': 0.05, 'E': 0.80},
    'B': {'A': 0.20, 'C': 0.50, 'D': 0.15, 'E': 0.15},
    'C': {'A': 0.30, 'B': 0.20, 'D': 0.20, 'E': 0.30},
    'D': {'A': 0.35, 'B': 0.30, 'C': 0.05, 'E': 0.30},
    'E': {'A': 0.20, 'B': 0.10, 'C': 0.60, 'D': 0.10},
    'F': {'A': 0.20, 'B': 0.20, 'C': 0.20, 'D': 0.20, 'E': 0.20}
}

# -------------------------------------------------------------
#  Baseline bed‑share across wards  (Sec. 5.2 capacity study)
# -------------------------------------------------------------
BED_FRACTIONS = {'A': 0.34, 'B': 0.25, 'C': 0.18, 'D': 0.12, 'E': 0.12}

# Initialization Function
def initialize_events(days=395):
    """Generate an *untruncated* Poisson arrival stream.

    We keep drawing inter‑arrival times until the cumulative
    sum exceeds the horizon *days*; any arrival that would fall
    outside the simulation window is discarded.  **No RNG reseed
    happens inside this function** so that subsequent calls in the
    same replication continue the global stream.
    """
    event_list = []
    # IMPORTANT – no reseed here (global seed set once at program start)
    for ptype in WARDS:
        t = np.random.exponential(1 / arrival_rates[ptype])
        while t < days:
            heapq.heappush(event_list, (t, 'Arr', ptype))
            t += np.random.exponential(1 / arrival_rates[ptype])
    return event_list

# Event Handling
def handle_arrival(event, bed, stay_means, penalties, blocked, event_list,
                   patients_total, patients_admitted, patients_relocated, patients_lost,
                   bed_full_events, burnin):
    time, _, ptype = event
    if time >= burnin:
        patients_total[ptype] += 1

    # --- relocation guard ---
    if ptype not in relocation_probs or sum(relocation_probs[ptype].values()) == 0:
        if time >= burnin:
            patients_lost[ptype] += 1
            blocked += 1
        return bed, penalties, blocked

    if bed[ptype] > 0:
        bed[ptype] -= 1
        los = sample_los(ptype, CURRENT_DIST, VAR_MULT) # length of stay
        heapq.heappush(event_list, (time + los, 'Dep', ptype))
        if time >= burnin:
            patients_admitted[ptype] += 1
    else: # If no bed is available
        dests = list(relocation_probs[ptype].keys())
        probs = list(relocation_probs[ptype].values())
        dest = random.choices(dests, probs)[0] #We randomly choose a destination ward based on these prob
        heapq.heappush(event_list, (time, 'Tra', (ptype, dest)))  # same time for transfer
        if time >= burnin:
            bed_full_events[ptype] += 1
            penalties[ptype] += urgency_points[ptype]

    return bed, penalties, blocked

def handle_departure(event, bed):
    _, _, ptype = event
    bed[ptype] += 1
    return bed


def handle_transfer(event, bed, stay_means, penalties, blocked, event_list,
                    patients_relocated, patients_lost, patients_admitted, burnin):
    time, _, (from_type, dest) = event
    if bed[dest] > 0:
        bed[dest] -= 1
        if time >= burnin:
            patients_relocated[from_type] += 1
        los = sample_los(from_type, CURRENT_DIST, VAR_MULT)
        heapq.heappush(event_list, (time + los, 'Dep', dest))
    else:
        if time >= burnin:
            patients_lost[from_type] += 1
            blocked += 1

    return bed, penalties, blocked

# Main Simulation Function
def simulate_hospital_flow(days=395, burnin=30, beds_override=None):
    bed = (beds_override or initial_beds).copy()
    penalties = {w: 0 for w in WARDS}
    blocked = 0
    event_list = initialize_events(days)

    patients_total = {w: 0 for w in WARDS}
    patients_admitted = {w: 0 for w in WARDS}
    patients_relocated = {w: 0 for w in WARDS}
    patients_lost = {w: 0 for w in WARDS}
    bed_full_events = {w: 0 for w in WARDS}


    while event_list:
        event = heapq.heappop(event_list)
        event_type = event[1]

        if event_type == 'Arr':
            bed, penalties, blocked = handle_arrival(
            event, bed, stay_means, penalties, blocked, event_list,
            patients_total, patients_admitted, patients_relocated, patients_lost, bed_full_events, burnin)


        elif event_type == 'Dep':
            bed = handle_departure(event, bed)

        elif event_type == 'Tra':
            bed, penalties, blocked = handle_transfer(
            event, bed, stay_means, penalties, blocked, event_list,
            patients_relocated, patients_lost, patients_admitted, burnin)


    # Compute metrics
    fraction_direct = {
        w: patients_admitted[w] / patients_total[w] if patients_total[w] > 0 else 0
        for w in WARDS
    }
    prob_full = {
        w: bed_full_events[w] / patients_total[w] if patients_total[w] > 0 else 0
        for w in WARDS
    }

    # Output results
    #print("\n--- Simulation Results ---")
    #for w in WARDS:
    #    print(f"Ward {w}")
    #    print(f"  N: patients   = {patients_total[w]}")
    #    print(f"  N: directly admitted   = {patients_admitted[w]}")
    #    print(f"  N: relocated  = {patients_relocated[w]}")
    #    print(f"  N: lost       = {patients_lost[w]}")
    #    print(f"  Penalty       = {penalties[w]}")
    #    print(f"  Fraction Directly Admitted        = {fraction_direct[w]:.4f}")
    #    print(f"  Probability Beds Full on Arrival  = {prob_full[w]:.4f}")
    #print(f"\nTotal blocked patients: {blocked}")

    return (patients_total, patients_admitted, patients_relocated, patients_lost, penalties, blocked)

# ▶ Run it
simulate_hospital_flow()



#2

def run_multiple_simulations(n=100):
    totals = {w: 0 for w in WARDS}
    admitted = {w: 0 for w in WARDS}
    relocated = {w: 0 for w in WARDS}
    lost = {w: 0 for w in WARDS}
    penalties = {w: 0 for w in WARDS}
    total_blocked_all = 0

    for _ in range(n):
        pts_total, pts_admitted, pts_relocated, pts_lost, pen, blocked = simulate_hospital_flow()
        for w in WARDS:
            totals[w] += pts_total[w]
            admitted[w] += pts_admitted[w]
            relocated[w] += pts_relocated[w]
            lost[w] += pts_lost[w]
            penalties[w] += pen[w]
        total_blocked_all += blocked

    # ✅ Compute average dictionaries — OUTSIDE the loop
    avg_totals = {w: totals[w] / n for w in WARDS}
    avg_admitted = {w: admitted[w] / n for w in WARDS}
    avg_relocated = {w: relocated[w] / n for w in WARDS}
    avg_lost = {w: lost[w] / n for w in WARDS}
    avg_penalties = {w: penalties[w] / n for w in WARDS}

    print(f"\n=== Averaged Results over {n} Simulations ===")
    for w in WARDS:
        avg_total = avg_totals[w]
        avg_adm = avg_admitted[w]
        avg_rel = avg_relocated[w]
        avg_lost_w = avg_lost[w]
        avg_pen = avg_penalties[w]
        frac = avg_adm / avg_total if avg_total > 0 else 0
        prob = (avg_rel + avg_lost_w) / avg_total if avg_total > 0 else 0

        print(f"\nWard {w}")
        print(f"  Avg patients            = {avg_total:.1f}")
        print(f"  Avg admitted            = {avg_adm:.1f}")
        print(f"  Avg relocated           = {avg_rel:.1f}")
        print(f"  Avg lost                = {avg_lost_w:.1f}")
        print(f"  Avg penalty             = {avg_pen:.1f}")
        print(f"  Avg directly admitted % = {frac:.4f}")
        print(f"  Avg full on arrival %   = {prob:.4f}")

    print(f"\nAvg total blocked patients per run: {total_blocked_all / n:.1f}")

    # Return all the full average dictionaries
    return avg_totals, avg_admitted, avg_relocated, avg_lost, avg_penalties

run_multiple_simulations(50)

# ------------------------------------------------------------------
# Sensitivity runner: vary LOS distribution & variance
# ------------------------------------------------------------------
def run_sensitivity(var_factors=(2.0, 3.0, 4.0), reps=30):
    global CURRENT_DIST, VAR_MULT

    scenario_penalties = {}      # raw totals per scenario label
    mean_results       = []      # (label, mean‑of‑totals) for bar plot

    # ---------- lognormal scenarios ----------
    for k in var_factors:
        CURRENT_DIST, VAR_MULT = "lognormal", k
        raw_totals = []                         # store 1 total per replication

        for _ in range(reps):
            _, _, _, _, pen, _ = simulate_hospital_flow()
            raw_totals.append(sum(pen.values()))

        scenario_penalties[str(k)] = raw_totals
        mean_results.append((str(k), np.mean(raw_totals)))

    # ---------- baseline exponential ----------
    CURRENT_DIST, VAR_MULT = "exp", 1.0
    raw_totals = []
    for _ in range(reps):
        _, _, _, _, pen, _ = simulate_hospital_flow()
        raw_totals.append(sum(pen.values()))

    scenario_penalties["exp"] = raw_totals
    mean_results.insert(0, ("exp", np.mean(raw_totals)))

    # ---------- numerical table ----------
    print(f"\n--- Sensitivity results over {reps} replications ---")
    print("Scenario   mean      var        95 % CI")
    for label, samples in scenario_penalties.items():
        arr  = np.asarray(samples, dtype=float)
        mean = arr.mean()
        var  = arr.var(ddof=1)
        se   = arr.std(ddof=1) / np.sqrt(reps)
        ci_lo, ci_hi = mean - 1.96*se, mean + 1.96*se
        print(f"{label:>7}  {mean:9.1f}  {var:9.1f}   [{ci_lo:,.1f}, {ci_hi:,.1f}]")

    # ---------- bar plot of means ----------
    labels, means = zip(*mean_results)
    plt.figure(figsize=(8,4))
    plt.bar(labels, means, color="steelblue")
    plt.ylabel("Average total penalty")
    plt.xlabel("LOS distribution (variance factor)")
    plt.title(f"Sensitivity analysis  –  {reps} replications per scenario")
    plt.tight_layout(); plt.show()

    # ---------- histograms ----------
    ncol = 2
    nrow = int(np.ceil(len(scenario_penalties)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3*nrow))
    axes = axes.ravel()

    for ax, (label, samples) in zip(axes, scenario_penalties.items()):
        ax.hist(samples, bins=10, color="gray", edgecolor="black")
        ax.set_title(f"LOS {label}")
        ax.set_xlabel("Total penalty"); ax.set_ylabel("freq")

    # blank any unused subplot panels
    for ax in axes[len(scenario_penalties):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()

    # reset globals for interactive runs
    CURRENT_DIST, VAR_MULT = "exp", 1.0


# ------------------------------------------------------------------
#  Capacity–sensitivity: vary TOTAL beds, keep same fractional split
# ------------------------------------------------------------------
def run_capacity_sensitivity(totals=(140, 150, 170, 180), reps=30):
    """Assess average penalty when total bed stock changes."""
    scenario_penalties, mean_results = {}, []

    for tot in totals:
        new_beds = {w: round(tot * BED_FRACTIONS[w]) for w in BED_FRACTIONS}
        raw_totals = []
        for _ in range(reps):
            _, _, _, _, pen, _ = simulate_hospital_flow(beds_override=new_beds)
            raw_totals.append(sum(pen.values()))

        scenario_penalties[str(tot)] = raw_totals
        mean_results.append((tot, np.mean(raw_totals)))

    # --- numeric table ---
    print(f"\n--- Capacity sensitivity ({reps} reps) ---")
    print("Beds   mean      var        95 % CI")
    for tot, mean in mean_results:
        arr = np.asarray(scenario_penalties[str(tot)])
        var = arr.var(ddof=1)
        se  = arr.std(ddof=1) / np.sqrt(reps)
        lo, hi = mean - 1.96*se, mean + 1.96*se
        print(f"{tot:>4}  {mean:9.1f}  {var:9.1f}   [{lo:,.1f}, {hi:,.1f}]")

    # --- bar chart of means ---
    beds, means = zip(*mean_results)
    plt.figure(figsize=(7,4))
    plt.bar([str(b) for b in beds], means, color="teal")
    plt.ylabel("Average total penalty")
    plt.xlabel("Total bed stock")
    plt.title(f"Penalty vs capacity – {reps} replications")
    plt.tight_layout(); plt.show()

    # --- histograms ---
    ncol = 2
    nrow = int(np.ceil(len(beds)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3*nrow))
    axes = axes.ravel()
    for ax, tot in zip(axes, beds):
        ax.hist(scenario_penalties[str(tot)], bins=10, color="gray",
                edgecolor="black")
        ax.set_title(f"{tot} beds")
        ax.set_xlabel("Total penalty"); ax.set_ylabel("freq")
    for ax in axes[len(beds):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    # ① LOS‑variance sensitivity
    run_sensitivity()
    # ② Capacity‑change sensitivity
    run_capacity_sensitivity()


#3

import matplotlib.pyplot as plt
import numpy as np

def plot_penalties(penalties):
    wards = list(penalties.keys())
    scores = [penalties[w] for w in wards]

    plt.figure(figsize=(10, 5))
    plt.bar(wards, scores, color='salmon')
    plt.ylabel('Penalty Score')
    plt.title('Penalty Score per Ward')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


# --- Pie Chart: Share of Total Patients per Ward ---
def plot_pie_total_patients(patients_total):
    labels = patients_total.keys()
    sizes = patients_total.values()

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Share of Total Patients per Ward')
    plt.axis('equal')
    plt.show()

# --- Bar Plot: Percentages of Outcomes per Ward ---
def plot_outcome_percentages(patients_total, patients_admitted, patients_relocated, patients_lost):
    wards = list(patients_total.keys())
    admitted_pct = [patients_admitted[w] / patients_total[w] * 100 if patients_total[w] > 0 else 0 for w in wards]
    relocated_pct = [patients_relocated[w] / patients_total[w] * 100 if patients_total[w] > 0 else 0 for w in wards]
    lost_pct = [patients_lost[w] / patients_total[w] * 100 if patients_total[w] > 0 else 0 for w in wards]

    x = np.arange(len(wards))
    width = 0.3

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, admitted_pct, width, label='Admitted %')
    plt.bar(x, relocated_pct, width, label='Relocated %')
    plt.bar(x + width, lost_pct, width, label='Lost %')

    plt.xticks(x, wards)
    plt.ylabel('Percentage (%)')
    plt.title('Patient Outcome Percentages per Ward')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# --- Bar Plot: Raw Patient Flow Counts ---
def plot_patient_flow(patients_total, patients_admitted, patients_relocated, patients_lost):
    wards = list(patients_total.keys())
    totals = [patients_total[w] for w in wards]
    admitted = [patients_admitted[w] for w in wards]
    relocated = [patients_relocated[w] for w in wards]
    lost = [patients_lost[w] for w in wards]

    x = np.arange(len(wards))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width, totals, width, label='Total Patients')
    plt.bar(x - 0.5*width, admitted, width, label='Admitted')
    plt.bar(x + 0.5*width, relocated, width, label='Relocated')
    plt.bar(x + 1.5*width, lost, width, label='Lost')

    plt.xticks(x, wards)
    plt.ylabel('Number of Patients')
    plt.title('Patient Flow per Ward')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# --- Bar Plot: Penalty Scores per Ward ---
def plot_penalties(penalties):
    wards = list(penalties.keys())
    scores = [penalties[w] for w in wards]

    plt.figure(figsize=(10, 5))
    plt.bar(wards, scores, color='salmon')
    plt.ylabel('Penalty Score')
    plt.title('Penalty Score per Ward')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Run the simulation (returns 6 values)
pt, pa, pr, pl, pen = run_multiple_simulations(50)

#Plot the results
plot_pie_total_patients(pt)
plot_patient_flow(pt, pa, pr, pl)
plot_penalties(pen)
plot_outcome_percentages(pt, pa, pr, pl)
