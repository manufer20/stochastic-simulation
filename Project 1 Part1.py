import dtumathtools, pandas, scipy, statsmodels, uncertainties

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, t

#############################################
# Project 1
#############################################

############ TASK 1 #######################################
# Load libraries
# (Not needed in Python as imports are above)

# Define the probability matrix
P = np.array([
    [0.9915, 0.005, 0.0025, 0, 0.001],
    [0, 0.986, 0.005, 0.004, 0.005],
    [0, 0, 0.992, 0.003, 0.005],
    [0, 0, 0, 0.991, 0.009],
    [0, 0, 0, 0, 1]
])

# Define parameters
women = 1000
states = P.shape[0]

# Simulate the cancer evolution of women until death
def SimulateOneWoman(P):
    # Initialize
    evolution = []
    current_state = 0  # Python is 0-indexed

    # Evolution, when still alive
    while current_state != 4:
        evolution.append(current_state)
        current_state = np.random.choice(states, p=P[current_state])

    # When a woman dies we add the "death state" to the evolution
    evolution.append(4)

    # Return the number of months a woman is alive and state evolution.
    return {'months_alive': len(evolution), 'evolution': evolution}

# Simulate the cancer evolution of all women
def SimulateAllWomen(n, P):
    months_alive = np.zeros(n, dtype=int)
    evolution_list = []
    longest_lifespan = 0

    for i in range(n):
        result = SimulateOneWoman(P)
        months_alive[i] = result['months_alive']
        evolution_list.append(result['evolution'])
        longest_lifespan = max(longest_lifespan, result['months_alive'])

    evolutions = np.full((n, longest_lifespan), 4)
    for i in range(n):
        evolutions[i, :len(evolution_list[i])] = evolution_list[i]

    return {'months_alive': months_alive, 'evolutions': evolutions}

# Proportions of each state
def StateProportions(evolutions, states):
    months = evolutions.shape[1]
    state_proportions = np.zeros((months, states))
    for t in range(months):
        for s in range(states):
            state_proportions[t, s] = np.mean(evolutions[:, t] == s)
    return state_proportions

# Perform simulation
np.random.seed(123)
results = SimulateAllWomen(women, P)

# Compute the proportions in each state
proportions = StateProportions(results['evolutions'], states)
plt.figure()
plt.plot(proportions)
plt.xlabel("Months")
plt.ylabel("Proportion")
plt.title("State Proportions Over Time")
plt.legend(["State 1", "State 2", "State 3", "State 4", "State 5"], loc="right")
plt.show()

# Proportion of women where the cancer reappears locally
LocalReappearance = np.any(results['evolutions'] == 1, axis=1)
LocalReappearance_Proportion = np.mean(LocalReappearance)

print("Proportion of women with local recurrence:", LocalReappearance_Proportion)

######## TASK 2 ######################################################

# Set evaluation time
month = 120

# Everyone starts in state 1
initial_distribution = np.array([1, 0, 0, 0, 0])

# Theoretical distribution at month = 120
theoretical_distribution = initial_distribution @ np.linalg.matrix_power(P, month)

# Empirical distribution at month=120
empirical_distribution = proportions[month - 1, :]

# Compare the two distributions
comparison = np.vstack([empirical_distribution, theoretical_distribution])

# Barplot visualizing the comparison
bar_width = 0.35
index = np.arange(states)
plt.figure()
plt.bar(index, comparison[0], bar_width, label='Empirical', color='skyblue')
plt.bar(index + bar_width, comparison[1], bar_width, label='Theoretical', color='red')
plt.xlabel('States')
plt.ylabel('Proportion')
plt.title('Empirical vs. Theoretical Distribution at Month 120')
plt.xticks(index + bar_width / 2, [f'State {i+1}' for i in range(states)])
plt.legend()
plt.show()

# Chi-squared test for verification
observed_counts = empirical_distribution * women
chisq_result = chisquare(f_obs=observed_counts, f_exp=theoretical_distribution * women)
print(chisq_result)




######### TASK 3 ######################################################

# Define the empirical lifetime distribution
Pi = np.array([1, 0, 0, 0])
P_s = P[:4, :4]
p_s = P[:4, 4]

def EmpiricalLifetimeDistribution(t):
    return (Pi @ np.linalg.matrix_power(P_s, t) @ p_s).item()

max_t = max(results['months_alive'])
xvalues = np.arange(1, max_t + 1)
theoretical_probs = np.array([EmpiricalLifetimeDistribution(t) for t in xvalues])

# Plot empirical vs theoretical PDF
plt.figure()
plt.hist(results['months_alive'], bins=30, density=True, color='skyblue', alpha=0.6, label='Empirical')
plt.plot(xvalues, theoretical_probs, color='blue', label='Theoretical', linewidth=2)
plt.title("Empirical vs Theoretical PDF of Lifetime")
plt.xlabel("Months until death")
plt.legend()
plt.show()

# Plot empirical vs theoretical CDF
empirical_cdf = np.array([np.mean(results['months_alive'] <= x) for x in xvalues])
theoretical_cdf = np.cumsum(theoretical_probs)
plt.figure()
plt.plot(xvalues, theoretical_cdf, color='blue', label='Theoretical', linewidth=2)
plt.plot(xvalues, empirical_cdf, color='red', label='Empirical', linewidth=2)
plt.title("Empirical vs Theoretical CDF")
plt.xlabel("Months")
plt.ylabel("CDF")
plt.legend()
plt.show()

# Kolmogorov-Smirnov Test
ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
adjusted_ks = (np.sqrt(max_t) + 0.12 + 0.11 / np.sqrt(max_t)) * ks_stat
print("Adjusted KS Statistic:", adjusted_ks)

#Note: critical value for 5% significance levels is 1.358, so we do not reject

##### TASK 4 ###################################################################

women_accepted = 1000
lifespan_accepted = []
accepted = 0
np.random.seed(123)

while accepted < women_accepted:
    woman = SimulateOneWoman(P)
    evo = woman['evolution']

    if len(evo) > 12:
        first_12 = evo[:12]
        if 1 in first_12 or 2 in first_12:
            accepted += 1
            lifespan_accepted.append(woman['months_alive'])

mean_lifetime = np.mean(lifespan_accepted)
sd_lifetime = np.std(lifespan_accepted, ddof=1)

plt.figure()
plt.hist(lifespan_accepted, bins=40, color='skyblue')
plt.title("Lifetime Distribution (Given Recurrence in First 12 Months)")
plt.xlabel("Months alive")
plt.show()

print("Estimated expected lifetime (given recurrence within 12 months):", mean_lifetime, "months")
print("Standard deviation:", sd_lifetime, "months")

######### TASK 5 ##################################################################################

np.random.seed(123)
n_simulations = 100
n_women = 200
death_month_threshold = 350
death_proportions = np.zeros(n_simulations)
mean_months_alive = np.zeros(n_simulations)

for i in range(n_simulations):
    sim = SimulateAllWomen(n_women, P)
    months_alive = sim['months_alive']
    death_proportions[i] = np.mean(months_alive <= death_month_threshold)
    mean_months_alive[i] = np.mean(months_alive)

# Crude Monte Carlo Estimator
crudeMC_mean = np.mean(death_proportions)
crudeMC_var = np.var(death_proportions, ddof=1)
print("Crude MC estimate of probability:", crudeMC_mean)
print("Crude MC variance:", crudeMC_var)

# Confidence interval
alpha = 0.05
q = t.ppf(1 - alpha / 2, df=n_simulations - 1)
crude_CI = [
    crudeMC_mean - q * np.sqrt(crudeMC_var) / np.sqrt(n_simulations),
    crudeMC_mean + q * np.sqrt(crudeMC_var) / np.sqrt(n_simulations)
]
print("Crude CI:", crude_CI)

# Control variate estimator
Pi = np.array([1, 0, 0, 0])
P_s = P[:4, :4]
p_s = P[:4, 4]
I = np.eye(4)
theor_mean_months_alive = (Pi @ np.linalg.inv(I - P_s) @ np.ones(4)).item()
c = -np.cov(death_proportions, mean_months_alive)[0,1] / np.var(mean_months_alive, ddof=1)
Z = death_proportions + c * (mean_months_alive - theor_mean_months_alive)
cv_mean = np.mean(Z)
cv_var = np.var(Z, ddof=1)

print("Control variate estimate of probability:", cv_mean)
print("Control variate variance:", cv_var)

cv_CI = [
    cv_mean - q * np.sqrt(cv_var) / np.sqrt(n_simulations),
    cv_mean + q * np.sqrt(cv_var) / np.sqrt(n_simulations)
]
print("Control Variate CI:", cv_CI)



## RESULTS

#Means
print("Crude MC estimate of probability:", crudeMC_mean)
print("Control variate estimate of probability:", cv_mean)

#Variance 
print("Crude MC variance:", crudeMC_var)
print("Control variate variance:", cv_var)

#Confidence intervals
print("Crude CI:", crude_CI)
print("Control Variate CI:", cv_CI)

# Reduction percent
reduction_percent = (crudeMC_var - cv_var) / crudeMC_var * 100
print("Variance reduction percent:", reduction_percent)