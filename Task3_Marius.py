import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, ttest_1samp

# Set seed
np.random.seed(42)

# Number of women
n = 1000
# Initial state
x0 = 1

# Matrix of probabilities
Pmat = np.array([
    [0.9915, 0.005, 0.0025, 0, 0.001],
    [0, 0.986, 0.005, 0.004, 0.005],
    [0, 0, 0.992, 0.003, 0.005],
    [0, 0, 0, 0.991, 0.009],
    [0, 0, 0, 0, 1]
])

# Function to calculate the jumps between state
def new_State(initial):
    return np.random.choice(np.arange(1, 6), p=Pmat[initial - 1])

# Matrix to hold output
result = np.zeros((1, n), dtype=int)

# Set the first state for all women to x0
result[0, :] = x0

# Variable to hold how many are still alive
remaining = n

# Loop while there still are living women
while remaining > 0:
    tempRes = np.zeros(n, dtype=int)
    for i in range(n):
        if result[-1, i] != 5:
            tempRes[i] = new_State(result[-1, i])
            if tempRes[i] == 5:
                remaining -= 1
        else:
            tempRes[i] = 5
    result = np.vstack([result, tempRes])
    print(f"Month {result.shape[0]} done! Women remaining: {remaining}")

# Get the lifetime of each woman
simLifetimes = np.argmax(result == 5, axis=0) - 1

# Truncate the outliers at the 95th percentile
cutoff = int(np.percentile(simLifetimes, 95))
simLifetimesT = np.where(simLifetimes > cutoff, cutoff, simLifetimes)

# Get the empirical distribution of the truncated lifetimes
simLifeDistT = np.bincount(simLifetimesT, minlength=cutoff + 1)[1:]

# Get the empirical mean lifetime
simLifeMean = np.mean(simLifetimes)

# Analytical distribution
π = np.array([1, 0, 0, 0])
Ps = Pmat[:4, :4]
ps = Pmat[:4, 4]

def anaLifetime(t):
    return float(π @ np.linalg.matrix_power(Ps, t) @ ps)

anaLifeMean = float(π @ np.linalg.inv(np.eye(4) - Ps) @ np.ones(Ps.shape[0]))
anaLifeDist = np.array([anaLifetime(i) for i in range(1, max(simLifetimes) + 1)])

# Truncate the analytical distribution
anaLifeDistT = anaLifeDist.copy()
anaLifeDistT[cutoff:] = 0
anaLifeCountT = anaLifeDistT * n

#Normalize expected frequencies to match observed total
expected = anaLifeCountT[:cutoff - 1]
expected *= simLifeDistT[:cutoff - 1].sum() / expected.sum()


# Chi-square test (you can fix normalization here)
 chisq_result = chisquare(simLifeDistT[:cutoff - 1], f_exp=anaLifeCountT[:cutoff - 1])
 print("Chi-square test p-value:", chisq_result.pvalue)

# T-test
ttest_result = ttest_1samp(simLifetimes, popmean=anaLifeMean)
print("T-test p-value:", ttest_result.pvalue)

# KS test
anaLifeDistN = anaLifeDist / np.sum(anaLifeDist)
anaLifeCDF = np.cumsum(anaLifeDistN)
simLifeCDF = np.array([np.mean(simLifetimes <= i) for i in range(1, max(simLifetimes) + 1)])

plt.plot(np.arange(1, max(simLifetimes) + 1), anaLifeCDF, label="Theoretical", color="blue", linewidth=2)
plt.plot(np.arange(1, max(simLifetimes) + 1), simLifeCDF, label="Empirical", color="red", linewidth=2)
plt.xlabel("Months")
plt.ylabel("CDF")
plt.title("Empirical vs Theoretical CDF")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

ksStat = np.max(np.abs(simLifeCDF - anaLifeCDF))
ksAdju = (np.sqrt(max(simLifetimes)) + 0.12 + 0.11 / np.sqrt(max(simLifetimes))) * ksStat
print("Adjusted KS statistic:", ksAdju)
