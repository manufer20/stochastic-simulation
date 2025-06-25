# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:51:57 2025

@author: User
"""
###----PACKAGES----####
import numpy as np
import random
import heapq
from scipy.stats import t
from scipy import stats
import matplotlib.pyplot as plt
import time

# Create a Generator using MT19937
bitgen = np.random.MT19937(seed=42)
rng = np.random.Generator(bitgen)

####----- GLOBAL VARIABLES -----#####
arrival_rates = {'A': 14.5, 'B': 11.0, 'C': 8.0, 'D': 6.5, 'E': 5.0, 'F': 13.0}
stay_means = {'A': 2.9, 'B': 4.0, 'C': 4.5, 'D': 1.4, 'E': 3.9, 'F': 2.2}
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

def ward_sim(n, burn_in, capacity, lambdas, mus, P):
    m = len(capacity)  # Number of wards / patient types

    # Initialize result storage
    patients = np.zeros(m, dtype=int)
    admitted = np.zeros(m, dtype=int)
    relocated = np.zeros(m, dtype=int)
    lost = np.zeros(m, dtype=int)

    # Initialize beds (time left until bed is free)
    max_capacity = max(capacity)
    beds = np.full((m, max_capacity), np.nan)
    for i in range(m):
        beds[i, :capacity[i]] = 0  # Beds are initially free (0 time)

    # Main loop: simulate burn-in + n patients
    for i in range(burn_in + n):
        # Sample inter-arrival time and reduce all bed times
        arrival_time = rng.exponential(scale=1 / sum(lambdas))
        beds = np.where(np.isnan(beds), np.nan, beds - arrival_time)

        # Sample patient type
        p_type = rng.choice(m, p=lambdas / sum(lambdas))

        # Only count patients after burn-in
        if i >= burn_in:
            patients[p_type] += 1

        # Intended ward is same as patient type
        assigned_ward = p_type
        los = rng.exponential(scale=1 / mus[p_type])

        # Try to find free bed in intended ward
        ward_beds = beds[assigned_ward]
        free_index = np.nanargmin(ward_beds)
        if ward_beds[free_index] <= 0:
            beds[assigned_ward, free_index] = los
            if i >= burn_in:
                admitted[p_type] += 1
        else:
            # Try to relocate
            dest_ward = rng.choice(m, p=P[p_type])
            dest_beds = beds[dest_ward]
            free_index = np.nanargmin(dest_beds)
            if dest_beds[free_index] <= 0:
                beds[dest_ward, free_index] = los
                if i >= burn_in:
                    relocated[p_type] += 1
            else:
                if i >= burn_in:
                    lost[p_type] += 1

    # Results
    return {
        "patients": patients,
        "admitted": admitted,
        "relocated": relocated,
        "lost": lost
    }
def simulate_burnin_occupancy(n=3000, burn_in=1000, capacity=None, lambda_rates=None, mu_rates=None, P=None, track_time=50):
    """
    Simulates hospital ward occupancy over time to determine burn-in period.
    Returns occupancy time series for each ward.
    """
    if capacity is None:
        capacity = [55, 40, 30, 20, 20]
    if lambda_rates is None:
        lambda_rates = [14.5, 11.0, 8.0, 6.5, 5.0]
    if mu_rates is None:
        mu_rates = [1/2.9, 1/4.0, 1/4.5, 1/1.4, 1/3.9]
    if P is None:
        P = np.array([
            [0.0, 0.05, 0.10, 0.05, 0.80],
            [0.20, 0.0, 0.50, 0.15, 0.15],
            [0.30, 0.20, 0.0, 0.20, 0.30],
            [0.35, 0.30, 0.05, 0.0, 0.30],
            [0.20, 0.10, 0.60, 0.10, 0.0]
        ])
        

    m = len(capacity)
    max_capacity = max(capacity)
    beds = np.full((m, max_capacity), np.nan)
    for i in range(m):
        beds[i, :capacity[i]] = 0  # initialize all usable beds to free

    occupancy_time_series = [[] for _ in range(m)]
    current_time = 0

    for i in range(burn_in):
        arrival_time = rng.exponential(1 / sum(lambda_rates))
        current_time += arrival_time
        beds = np.where(np.isnan(beds), np.nan, beds - arrival_time)

        patient_type = rng.choice(m, p=np.array(lambda_rates) / sum(lambda_rates))
        assigned_ward = patient_type
        los = rng.exponential(1 / mu_rates[patient_type])

        bed_index = np.nanargmin(beds[assigned_ward])
        bed_time = beds[assigned_ward, bed_index]

        if bed_time <= 0:
            beds[assigned_ward, bed_index] = los
        else:
            assigned_ward = rng.choice(m, p=P[patient_type])
            bed_index = np.nanargmin(beds[assigned_ward])
            bed_time = beds[assigned_ward, bed_index]
            if bed_time <= 0:
                beds[assigned_ward, bed_index] = los
            # else lost (not tracked here)

        if i % (burn_in // track_time) == 0:
            for j in range(m):
                occ = np.sum(beds[j, :] > 0)
                occupancy_time_series[j].append(occ)

    return occupancy_time_series, capacity

# Run the simulation
#occupancy_time_series, capacity = simulate_burnin_occupancy()
def plot_ward_occupancy(occupancy_time_series):
    wards = ['A', 'B', 'C', 'D', 'E', 'F']
    plt.figure(figsize=(12, 6))
    for i, series in enumerate(occupancy_time_series):
        plt.plot(series, label=f"Ward {wards[i]} (Cap={capacity[i]})")
    plt.xlabel("Simulation Steps (relative to burn-in)")
    plt.ylabel("Beds Occupied")
    plt.title("Ward Occupancy Over Time During Burn-in")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#plot_ward_occupancy(occupancy_time_series)

INCLUDE_F = True  # Set to True to include Ward F
WARDS = ['A', 'B', 'C', 'D', 'E'] + (['F'] if INCLUDE_F else [])



# Initialization Function
def initialize_events(days=395):
    event_list = []
    for ptype in WARDS:
        # no. of arrivals to simulate
        num_arrivals = int( arrival_rates[ptype] * days) 
        # mean time between arrivals
        mean_interarrival_time = 1 / arrival_rates[ptype]
        # Generate interarrival time
        interarrival_times = rng.exponential(mean_interarrival_time, num_arrivals)
        arrivals = np.cumsum(interarrival_times)
        for t in arrivals:
            if t < days:
                heapq.heappush(event_list, (t, 'Arr', ptype))
    return event_list

# Event Handling
def handle_arrival(event, bed, stay_means, penalties, blocked, event_list,
                   patients_total, patients_admitted, patients_relocated,
                   patients_lost, bed_full_events, burnin):
    time, _, ptype = event
    #print(event)
    if time >= burnin:
        patients_total[ptype] += 1

    if bed[ptype] > 0:
        bed[ptype] -= 1
        los = rng.exponential(stay_means[ptype]) # length of stay
        heapq.heappush(event_list, (time + los, 'Dep', ptype))
        if time >= burnin:
            patients_admitted[ptype] += 1
    else: # If no bed is available
        dests = list(relocation_probs[ptype].keys())
        probs = list(relocation_probs[ptype].values())
        dest = rng.choice(dests, p=probs)  #We randomly choose a destination ward based on these prob
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
        los = rng.exponential(stay_means[from_type])
        heapq.heappush(event_list, (time + los, 'Dep', dest))
    else:
        if time >= burnin:
            patients_lost[from_type] += 1
            blocked += 1

    return bed, penalties, blocked

# Main Simulation Function
def simulate_hospital_flow(days=395, burnin=30, bed_config=None):
    bed = bed_config.copy() if bed_config else initial_beds.copy()
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
   
    total_penalty = sum(penalties.values())

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
    ##print(f"\nTotal penalty points: {total_penalty}")
    return (
    patients_total,
    patients_admitted,
    patients_relocated,
    patients_lost,
    penalties,
    blocked,
    bed_full_events,  # per-ward dict
    total_penalty     # scalar
)

# ▶ Run it
#simulate_hospital_flow()

def run_multiple_simulations(n=1, bed_config=None):
    totals = {w: 0 for w in WARDS}
    admitted = {w: 0 for w in WARDS}
    relocated = {w: 0 for w in WARDS}
    lost = {w: 0 for w in WARDS}
    penalties = {w: 0 for w in WARDS}
    bed_full_events_total = {w: 0 for w in WARDS}
    total_blocked_all = 0
    total_penalties = 0

    for _ in range(n):
        pts_total, pts_admitted, pts_relocated, pts_lost, pen, blocked, bed_full, total_penalty = simulate_hospital_flow(bed_config=bed_config)
        for w in WARDS:
            totals[w] += pts_total[w]
            admitted[w] += pts_admitted[w]
            relocated[w] += pts_relocated[w]
            lost[w] += pts_lost[w]
            penalties[w] += pen[w]
            bed_full_events_total[w] += bed_full[w]
        total_blocked_all += blocked
        total_penalties += total_penalty

    # Averages
    avg_totals = {w: totals[w] / n for w in WARDS}
    avg_admitted = {w: admitted[w] / n for w in WARDS}
    avg_relocated = {w: relocated[w] / n for w in WARDS}
    avg_lost = {w: lost[w] / n for w in WARDS}
    avg_penalties = {w: penalties[w] / n for w in WARDS}
    avg_bed_full = {w: bed_full_events_total[w] / n for w in WARDS}
    avg_total_penalties = total_penalties / n
    avg_total_blocked = total_blocked_all / n

    print(f"\n=== Averaged Results over {n} Simulations ===")
    for w in WARDS:
        avg_total = avg_totals[w]
        avg_adm = avg_admitted[w]
        avg_rel = avg_relocated[w]
        avg_lost_w = avg_lost[w]
        avg_pen = avg_penalties[w]
        avg_full = avg_bed_full[w]
        frac = avg_adm / avg_total if avg_total > 0 else 0
        prob = avg_full / avg_total if avg_total > 0 else 0

        print(f"\nWard {w}")
        print(f"  Avg patients            = {avg_total:.1f}")
        print(f"  Avg admitted            = {avg_adm:.1f}")
        print(f"  Avg relocated           = {avg_rel:.1f}")
        print(f"  Avg lost                = {avg_lost_w:.1f}")
        print(f"  Avg penalty             = {avg_pen:.1f}")
        print(f"  Avg full on arrival     = {avg_full:.1f}")
        print(f"  Avg directly admitted % = {frac:.4f}")
        print(f"  Probability full on arrival = {prob:.4f}")

    print(f"\nAvg total blocked patients per run: {avg_total_blocked:.1f}")
    print(f"Avg total penalty points per run: {avg_total_penalties:.1f}")

    return (
        avg_totals,
        avg_admitted,
        avg_relocated,
        avg_lost,
        avg_penalties,
        avg_bed_full,
        avg_total_blocked,
        avg_total_penalties
    )
#run_multiple_simulations(1)
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

# Run the simulation 
pt, pa, pr, pl, pen, *_ = run_multiple_simulations(1)


#Plot the results
def plot_multiple_sim(pt, pa, pr, pl, pen, *_):
    plot_pie_total_patients(pt)
    plot_patient_flow(pt, pa, pr, pl)
    plot_penalties(pen)
    plot_outcome_percentages(pt, pa, pr, pl)
#plot_multiple_sim(pt, pa, pr, pl, pen, *_)

class HospitalModel:
    """Represents the hospital's rules, state, and statistics for a single simulation run."""
    def __init__(self, bed_config):
        # --- Static Parameters (The "Rules") ---
        self.wards = list(bed_config.keys())
        self.capacities = bed_config.copy()
        self.arrival_rates = {'A': 14.5, 'B': 11.0, 'C': 8.0, 'D': 6.5, 'E': 5.0, 'F': 13.0}
        self.mean_stays = {'A': 2.9, 'B': 4.0, 'C': 4.5, 'D': 1.4, 'E': 3.9, 'F': 2.2}
        self.urgency_points = {'A': 7, 'B': 5, 'C': 2, 'D': 10, 'E': 5, 'F': 0}
        self.reloc_probs = {
            'A': {'B': 0.05, 'C': 0.10, 'D': 0.05, 'E': 0.80},
            'B': {'A': 0.20, 'C': 0.50, 'D': 0.15, 'E': 0.15},
            'C': {'A': 0.30, 'B': 0.20, 'D': 0.20, 'E': 0.30},
            'D': {'A': 0.35, 'B': 0.30, 'C': 0.05, 'E': 0.30},
            'E': {'A': 0.20, 'B': 0.10, 'C': 0.60, 'D': 0.10},
            'F': {'A': 0.20, 'B': 0.20, 'C': 0.20, 'D': 0.20, 'E': 0.20}
        }
        
        # --- Dynamic State Variables (What changes during simulation) ---
        self.occupied_beds = {w: 0 for w in self.wards}
        self.stats = {
            'total_arrivals': {w: 0 for w in self.wards},
            'primary_admissions': {w: 0 for w in self.wards},
            'relocated_from': {w: 0 for w in self.wards},
            'lost_patients': {w: 0 for w in self.wards}
        }

    def handle_arrival(self, patient_type):
        """Handles a patient arrival and returns the ward they were admitted to (or None)."""
        self.stats['total_arrivals'][patient_type] += 1
        
        # --- Primary Admission Attempt ---
        if self.occupied_beds[patient_type] < self.capacities[patient_type]:
            self.occupied_beds[patient_type] += 1
            self.stats['primary_admissions'][patient_type] += 1
            return patient_type
        
        # --- Relocation Attempt ---
        else:
            self.stats['relocated_from'][patient_type] += 1
            
            # Choose an alternative ward based on probabilities
            reloc_options = self.reloc_probs.get(patient_type, {})
            if not reloc_options: # Handle case where a type has no relocation options
                self.stats['lost_patients'][patient_type] += 1
                return None
                
            wards = list(reloc_options.keys())
            probs = list(reloc_options.values())
            alt_ward = rng.choice(wards, p=probs)
            
            if self.occupied_beds[alt_ward] < self.capacities[alt_ward]:
                self.occupied_beds[alt_ward] += 1
                return alt_ward
            else:
                self.stats['lost_patients'][patient_type] += 1
                return None

    def handle_departure(self, ward_of_admission):
        """Frees up a bed in the specified ward."""
        if self.occupied_beds[ward_of_admission] > 0:
            self.occupied_beds[ward_of_admission] -= 1

def simulate_hospital_flow(bed_config, duration=365, burn_in=50):
    """
    This is the main simulation function. It takes a bed layout and returns
    the key performance metrics needed by the optimizer.
    """
    model = HospitalModel(bed_config)
    event_list = []
    current_time = 0.0

    def schedule_event(delay, event_type, details):
        heapq.heappush(event_list, (current_time + delay, event_type, details))

    # Kickstart by scheduling the first arrival for each patient type
    for p_type in model.wards:
        if p_type in model.arrival_rates:
            delay = rng.exponential(1.0 / model.arrival_rates[p_type])
            schedule_event(delay, 'ARRIVAL', {'patient_type': p_type})

    # Main simulation loop
    while event_list and current_time < duration:
        time, event_type, details = heapq.heappop(event_list)
        current_time = time

        if event_type == 'ARRIVAL':
            p_type = details['patient_type']
            # Always schedule the next arrival for this type
            delay = rng.exponential(1.0 / model.arrival_rates[p_type])
            schedule_event(delay, 'ARRIVAL', {'patient_type': p_type})
            
            # Only process/collect stats after the burn-in period
            if current_time > burn_in:
                ward_admitted = model.handle_arrival(p_type)
                if ward_admitted:
                    stay = rng.exponential(model.mean_stays[p_type])
                    schedule_event(stay, 'DEPARTURE', {'ward': ward_admitted})
        
        elif event_type == 'DEPARTURE':
            if current_time > burn_in:
                model.handle_departure(details['ward'])

    # --- Calculate final metrics at the end of the simulation ---
    # 1. Total penalty score from relocated patients
    total_penalty = sum(model.urgency_points[w] * model.stats['relocated_from'][w] for w in model.wards if w != 'F')
    
    # 2. Admission rate for Ward F
    f_arrivals = model.stats['total_arrivals']['F']
    f_admissions = model.stats['primary_admissions']['F']
    f_admission_rate = (f_admissions / f_arrivals) if f_arrivals > 0 else 1.0

    return f_admission_rate, total_penalty


# ==============================================================================
# PART 2: The Simulated Annealing Optimizer
# ==============================================================================
# This section uses the simulation model to find the best bed layout.

def objective_function(bed_config):
    """
    This is the function the optimizer tries to minimize.
    It returns a single "cost" value for a given bed layout.
    """
    f_admission_rate, penalty_score = simulate_hospital_flow(bed_config=bed_config)
    
    # Add a massive penalty if the Ward F admission constraint is violated.
    # This forces the optimizer to find solutions that are valid.
    if f_admission_rate < 0.95:
        penalty_score += 1000

    return penalty_score

def generate_neighbor(config, src_weights):
    """
    Creates a new "neighbor" configuration by intelligently moving one bed.
    """
    config = config.copy()
    
    # Choose a source ward to take a bed FROM.
    # It must have beds to give and we prefer to take from low-urgency wards.
    valid_sources = {w: weight for w, weight in src_weights.items() if config[w] > 0}
    if not valid_sources: return config
    
    items = list(valid_sources.items())
    wards, weights = zip(*items)
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    src = rng.choice(wards, p=probs)

    # Choose a destination ward to give the bed TO.
    dst_candidates = [w for w in config.keys() if w != src]
    dst = rng.choice(dst_candidates)

    config[src] -= 1
    config[dst] += 1
    return config

def simulated_annealing_optimizer(initial_config, src_weights, max_iter=2000):
    """
    The main simulated annealing algorithm.
    """
    current_config = initial_config.copy()
    current_cost = objective_function(current_config)
    
    best_config = current_config.copy()
    best_cost = current_cost
    
    cost_history = [current_cost]
    patience = 200
    stall_counter = 0

    T0 = 20.0  # Initial temperature
    T_end = 0.1 # Final temperature

    for k in range(1, max_iter + 1):
        T = T0 * (T_end / T0) ** (k / max_iter)
        
        neighbor_config = generate_neighbor(current_config, src_weights)
        cost = objective_function(neighbor_config)
        
        delta = cost - current_cost
        
        # Metropolis acceptance criterion
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_config = neighbor_config
            current_cost = cost
        
        # Track the best valid solution found so far
        if current_cost < best_cost:
            best_config = current_config.copy()
            best_cost = current_cost
            stall_counter = 0
        else:
            stall_counter += 1
        
        cost_history.append(current_cost)
        
        if stall_counter >= patience:
            print(f"Stopping early at iteration {k} due to no improvement.")
            break
            
    # Plot the optimization progress
    plt.figure(figsize=(10,4))
    plt.plot(cost_history)
    plt.title("Simulated Annealing Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Total Penalty Score")
    plt.grid(True)
    plt.show()

    return best_config, best_cost

# ==============================================================================
# PART 3: Main Execution Block
# ==============================================================================

# --- Define Global Parameters for the run ---
WARDS = ['A', 'B', 'C', 'D', 'E', 'F']
urgency_points = {'A': 7, 'B': 5, 'C': 2, 'D': 10, 'E': 5, 'F': 0}
src_weights = {w: 1 / urgency_points[w] if urgency_points.get(w, 0) > 0 else 0 for w in WARDS if w != 'F'}

# --- Set Initial Configuration ---
# Start with a simple, balanced configuration.
initial_beds = {'A': 48, 'B': 33, 'C': 23, 'D': 14, 'E': 13, 'F': 34} 
print(f"Initial bed configuration: {initial_beds}")
print(f"Total beds: {sum(initial_beds.values())}")

# --- Run the Optimizer ---
print("\nStarting optimization process...")
start_time = time.time()
best_config, best_cost = simulated_annealing_optimizer(initial_beds, src_weights)
end_time = time.time()

# --- Report Final Results ---
print("\n--- Best Bed Allocation Found by Optimizer ---")
for ward in WARDS:
    print(f"  Ward {ward}: {best_config[ward]} beds")

print(f"\nMinimum Total Penalty Found: {best_cost:.2f}")
print(f"Time Taken for optimization: {end_time - start_time:.2f} seconds")

# --- Final verification run with the best configuration ---
print("\n--- Verifying performance of best configuration ---")
f_admission_rate, penalty_score = simulate_hospital_flow( duration=365, burn_in=50, bed_config=best_config,)
print(f"  Verification Run - Final Penalty Score: {penalty_score:.2f}")
print(f"  Verification Run - Ward F Admission Rate: {f_admission_rate:.2%}")