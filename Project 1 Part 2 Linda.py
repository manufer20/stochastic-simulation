import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, t, chi2
from scipy.linalg import expm

###################################################
# Project 1 Part 2
###################################################

# Define Q matrix
Q = np.array([
    [-0.0085, 0.005, 0.0025, 0, 0.001],
    [0, -0.014, 0.005, 0.004, 0.005],
    [0, 0, -0.008, 0.003, 0.005],
    [0, 0, 0, -0.009, 0.009],
    [0, 0, 0, 0, 0]
])

# Function to simulate a single woman
def SimulateContTimeWoman(Q):

    # Initialize
    current_state = 0  # Starts in state 1, python is 0 indexed.
    time = 0  # time starts at month 0

    # Prepare storage
    state_evolution = [current_state + 1]  # Use +1 to match R's 1-based indexing
    times = [time]

    # When still alive
    while current_state != 4:

        # Time spent in current state
        rate = -Q[current_state, current_state]  # Exit rate from current state
        holding_time = expon.rvs(scale=1 / rate)  # Draws a random time using current state rate
        time += holding_time

        # Sample the next transition
        probs = Q[current_state, :].copy()  # transition probability from current state to others
        probs[current_state] = 0  # No self transitions
        probs = probs / np.sum(probs)  # Normalize the rates
        current_state = np.random.choice(5, p=probs)  # Randomly select state based on probability

        # Record the new state and time
        state_evolution.append(current_state + 1)
        times.append(time)

    # Return total time, state evolution and state transition times
    return {'total_time': time, 'state_evolution': state_evolution, 'times': times}

# Simulate 1000 women
np.random.seed(123)
women = 1000
results = [SimulateContTimeWoman(Q) for _ in range(women)]

months_alive = np.array([res['total_time'] for res in results])  # Initialize empty vector to store lifetimes

# Histogram of months alive
plt.hist(months_alive, bins=40, color='skyblue')
plt.title("Lifetime Distribution (CTMC)")
plt.xlabel("Months until death")
plt.ylabel("Frequency")
plt.show()

# Summary statistics
mean_months_alive = np.mean(months_alive)
sd_months_alive = np.std(months_alive, ddof=1)

# 95% confidence interval for mean
alpha = 0.05
t_val = t.ppf(1 - alpha / 2, df=women - 1)
mean_CI = mean_months_alive + np.array([-1, 1]) * t_val * sd_months_alive / np.sqrt(women)

# Confidence interval for sd
df = women - 1
chi_low = chi2.ppf(1 - alpha / 2, df)
chi_high = chi2.ppf(alpha / 2, df)
var_lower = (df * sd_months_alive**2) / chi_low
var_upper = (df * sd_months_alive**2) / chi_high
sd_CI = np.sqrt([var_lower, var_upper])

#Results
mean_months_alive
sd_months_alive
mean_CI
sd_CI

### Proportion of women with distant cancer reappearance after 30.5 months ###
counter = 0  # Initialize counter

# Loop through each simulated woman
for i in range(women):
    state_evolution = results[i]['state_evolution']
    times = results[i]['times']

    # Check if state 3 appears within 30.5 months, if so increase count
    for j in range(1, len(state_evolution)):  # Note we skip state 1, since it will always be 1
        if state_evolution[j] == 3 and times[j] <= 30.5:
            counter += 1
            break  # Leave loop after woman enters state 3

# Compute Proportion of distant reappearance
counter / women

######## TASK 8 ############################################################################

# The continuous phase time distribution is given by
# F_T(t) = 1 - p0 * exp(Qs * t) * 1

Qs = Q[:4, :4]  # Submatrix without state 5
p0 = np.array([1, 0, 0, 0])  # All start in state 1
ones = np.ones(4)

# Define the theoretical distribution function
def FT(t):
    return 1 - np.dot(p0, np.dot(expm(Qs * t), ones))

# Find the empirical CDF
xvalues = np.sort(months_alive)  # Empirical x-axis
theoretical_cdf = np.array([FT(t) for t in xvalues])
empirical_cdf = np.searchsorted(np.sort(months_alive), xvalues, side='right') / women

plt.plot(xvalues, theoretical_cdf, label='Theoretical', color='blue', linewidth=2)
plt.plot(xvalues, empirical_cdf, label='Empirical', color='red', linewidth=2)
plt.xlabel("Months")
plt.ylabel("CDF")
plt.title("Empirical vs Theoretical CDF")
plt.legend(loc='lower right')
plt.show()

### Perform Kolmogorov Smirnov test ###
# Test statistic calculated using formula on page 15 slide2bm1
ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))

# The adjusted statistic is calculated using formula on page 16 slide2bm1
adjusted_ks = (np.sqrt(women) + 0.12 + 0.11 / np.sqrt(women)) * ks_stat

# Remember critical value for a 5% significance level is 1.358 (page 16 slide2bm1)

######### TASK 9 #####################################################################

# Introduce a preventative treatment

# Define new Q matrix
Q_treated = np.array([
    [-0.00475, 0.0025, 0.00125, 0, 0.001],
    [0, -0.007, 0, 0.002, 0.005],
    [0, 0, -0.008, 0.003, 0.005],
    [0, 0, 0, -0.009, 0.009],
    [0, 0, 0, 0, 0]
])

# Simulate women with new treatment
np.random.seed(456)
results_treated = [SimulateContTimeWoman(Q_treated) for _ in range(women)]
months_alive_treated = np.array([res['total_time'] for res in results_treated])

# Define the Kaplan-Meier estimator as in the project description
def Shat(month, months_alive):
    N = len(months_alive)  # Corresponds to number of women
    d = np.sum(months_alive <= month)  # Number of women died before threshold
    # The survival fraction
    return (N - d) / N

# Compare survival curves
tvalues = np.arange(0, int(max(months_alive.max(), months_alive_treated.max())) + 1)
SurvFrac_untreated = np.array([Shat(t, months_alive) for t in tvalues])
SurvFrac_treated = np.array([Shat(t, months_alive_treated) for t in tvalues])

# Plot Kaplan-Meier curves
plt.plot(tvalues, SurvFrac_untreated, label='Untreated', color='blue', linewidth=2)
plt.plot(tvalues, SurvFrac_treated, label='Treated', color='green', linewidth=2)
plt.xlabel("Months")
plt.ylabel("Survival Probability")
plt.title("Kaplan-Meier Survival Curves")
plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.show()
