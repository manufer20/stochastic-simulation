import numpy as np

def roll_relocation_die(origin, reloc_probs):
    """
    Draw a relocation ward for a blocked arrival from `origin`.

    Parameters
    ----------
    origin : str
        The ward that is currently full (e.g. 'A', 'B', …).
    reloc_probs : dict
        Relocation-probability matrix exactly as you defined it.

    Returns
    -------
    dest : str
        Ward to which the patient should be redirected.
    """
    row = reloc_probs[origin]                 # e.g. {'A':0.0, 'B':0.05, …}
    dests   = [w for w, p in row.items() if p > 0]   # remove zero-prob self entry
    weights = [row[w] for w in dests]
    probs   = np.array(weights) / sum(weights)        # normalise (just in case)
    return np.random.choice(dests, p=probs)









import numpy as np

np.random.seed(0)
capacities = {'A': 55, 'B': 40, 'C': 30, 'D': 20, 'E': 20}
arrival_rates = {'A': 14.5, 'B': 11.0, 'C': 8.0, 'D': 6.5, 'E': 5.0}
mean_stays = {'A': 2.9, 'B': 4.0, 'C': 4.5, 'D': 1.4, 'E': 3.9}
reloc_probs = {
    'A': {'A': 0.0, 'B': 0.05, 'C': 0.10, 'D': 0.05, 'E': 0.80},
    'B': {'A': 0.2, 'B': 0.0, 'C': 0.50, 'D': 0.15, 'E': 0.15},
    'C': {'A': 0.3, 'B': 0.2, 'C': 0.0, 'D': 0.20, 'E': 0.30},
    'D': {'A': 0.35,'B': 0.3, 'C': 0.05, 'D': 0.0, 'E': 0.30},
    'E': {'A': 0.2, 'B': 0.1, 'C': 0.60, 'D': 0.1, 'E': 0.0}
}

actual_time = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
actual_capacities= {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
patients_log= {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}

departure_times= {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}

occupancy_log = []  # (event‑time, ward, +1 for admission / –1 for discharge)

#initial kick
blocked_log=[]
for capacity in actual_capacities.keys():
            actual_time[capacity]+=np.random.exponential(scale=1/arrival_rates[capacity])
            
next_patient_to_arrive = min(actual_time, key=actual_time.get)            
departure_times[next_patient_to_arrive].append(actual_time[next_patient_to_arrive]+np.random.exponential(scale=1/mean_stays[next_patient_to_arrive]))            



while any(days < 365 for days in actual_time.values()):
         

        next_patient_to_arrive = min(actual_time, key=actual_time.get)
        all_departures_generator = (
        (time, ward) 
        for ward, times_list in departure_times.items() 
        for time in times_list
        )
        # Use min() to find the pair with the smallest time.
        # The `default` handles the case where there are no departures scheduled.
        
        earliest_departure_time, ward_to_depart = min(all_departures_generator, default=(float('inf'), None))
    

        if actual_time[next_patient_to_arrive] < earliest_departure_time:
            if actual_capacities[next_patient_to_arrive] < capacities[next_patient_to_arrive]:
                actual_capacities[next_patient_to_arrive]+=1
                occupancy_log.append((actual_time[next_patient_to_arrive], next_patient_to_arrive, +1))
                # patient_stay=actual_time[next_patient_to_arrive]+np.random.exponential(scale=1/mean_stays[next_patient_to_arrive])
                departure_times[next_patient_to_arrive].append(actual_time[next_patient_to_arrive]+np.random.exponential(scale=mean_stays[next_patient_to_arrive]))            
                # patients_log[next_patient_to_arrive].append([actual_time[next_patient_to_arrive],actual_capacities[next_patient_to_arrive],patient_stay])
                actual_time[next_patient_to_arrive]+=np.random.exponential(scale=1/arrival_rates[next_patient_to_arrive])
            else:
            
                new_ward = roll_relocation_die(next_patient_to_arrive, reloc_probs)

                if actual_capacities[new_ward] < capacities[new_ward]:
                       actual_capacities[new_ward] += 1
                       occupancy_log.append((actual_time[next_patient_to_arrive], new_ward, +1))
                       dep_time = actual_time[next_patient_to_arrive] + np.random.exponential(
                       scale=mean_stays[new_ward])
                       departure_times[new_ward].append(dep_time)
                       # always schedule the next arrival for the origin ward
                       actual_time[next_patient_to_arrive] += np.random.exponential(
                           scale=1/arrival_rates[next_patient_to_arrive])
                else:
                    blocked_log.append(actual_time[next_patient_to_arrive])        
                    # even when blocked, schedule the ward’s next arrival
                    actual_time[next_patient_to_arrive] += np.random.exponential(
                        scale=1/arrival_rates[next_patient_to_arrive])
        else: 
                    actual_capacities[ward_to_depart] = actual_capacities[ward_to_depart]-1
                    occupancy_log.append((earliest_departure_time, ward_to_depart, -1))
                    departure_times[ward_to_depart].remove(earliest_departure_time)