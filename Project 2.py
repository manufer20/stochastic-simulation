# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:30:17 2025

@author: User
"""

#import simpy
import numpy as np


# Create a Generator using MT19937
bitgen = np.random.MT19937(seed=0)
rng = np.random.Generator(bitgen)

# Parameters from Table 1
Data=np.array([['A',55, 14.5,2.9,7],['B',40,11,4,5],['C',30,8,4.5,2],['D',20,6.5,1.4,10],['E',20,5,3.9,5],['F',0,13,2.2,0]])
#print(Data[:,[0,2]])

# Relocation probabilities (pij) if ward i is full
relocation_probs = {
    'A': {'B': 0.05, 'C': 0.10, 'D': 0.05, 'E': 0.80},
    'B': {'A': 0.20, 'C': 0.50, 'D': 0.15, 'E': 0.15},
    'C': {'A': 0.30, 'B': 0.20, 'D': 0.20, 'E': 0.30},
    'D': {'A': 0.35, 'B': 0.30, 'C': 0.05, 'E': 0.30},
    'E': {'A': 0.20, 'B': 0.10, 'C': 0.60, 'D': 0.10}
}

def initialization(data_array, length=1):
    """Simulates a day's worth of arrivals from ward data"""
    all_events = []
    #count=[]
    for row in data_array:
        patient_type = row[0]
        arrival_rate = float(row[2])  # Ensure it's a float

        time = 0.0
        while time < length:
            interarrival = rng.exponential(scale=1 / arrival_rate)
            time += interarrival
            if time < length:
                all_events.append((round(time, 2), patient_type, "arrival"))
            
    
    # Sort all events by time
    all_events.sort(key=lambda x: x[0])
    all_events=np.array(all_events)
    ward_arrivals=all_events[:,1]       
    wards, counts = np.unique(ward_arrivals, return_counts=True)
    #print(counts/day_length)
    return all_events

# Generate and display
def arrivals(events):
    capacity=Data[:, 1].astype(int)
    print(capacity)
    ward_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 0}
    for time, ward, event_type in events:
        if event_type == "arrival":
            i = ward_to_index[ward]
            #if beds[i] > 0:
            capacity[i] -= 1
        
    print(capacity)
events = initialization(Data)
arrivals(events)
