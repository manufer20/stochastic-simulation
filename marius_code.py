import numpy as np

# Set seed for reproducibility
np.random.seed(1234)

# Number of women to simulate
n = 1000

# Transition rate matrix Q
Q = np.array([
    [-0.0085, 0.0050, 0.0025, 0.0000, 0.0010],
    [0.0000, -0.0140, 0.0050, 0.0040, 0.0050],
    [0.0000, 0.0000, -0.0080, 0.0030, 0.0050],
    [0.0000, 0.0000, 0.0000, -0.0090, 0.0090],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
])

# Time between observations (in months)
obsDel = 48

# Convergence criterion
convCrit = 0.001

# ------------------------------------------------------------
# Function to simulate a single woman
# ------------------------------------------------------------
def sim1Woman():
    # list of [time, state] – start in state 1 at t = 0
    output = [[0.0, 1]]
    currstate = 1
    while currstate != 5:
        # waiting time in current state
        wait = np.random.exponential(scale=-1 / Q[currstate - 1, currstate - 1])
        # time of next transition
        time_next = output[-1][0] + wait
        # choose next state
        stateProps = np.delete(Q[currstate - 1], currstate - 1)
        statePropsN = stateProps / stateProps.sum()
        nextstate = np.random.choice(
            [s for s in range(1, 6) if s != currstate],
            p=statePropsN
        )
        output.append([time_next, nextstate])
        currstate = nextstate
    return np.array(output)


# ------------------------------------------------------------
# Function to simulate n women, observed only every obsDel
# ------------------------------------------------------------
def simObsWomen():
    output = []
    for _ in range(n):
        woman = sim1Woman()
        ind = 1
        currstate = 1
        obs = [currstate]
        while currstate != 5:
            t = ind * obsDel
            idx = np.where(woman[:, 0] < t)[0][-1]
            currstate = int(woman[idx, 1])
            obs.append(currstate)
            ind += 1
        output.append(obs)
    return output


# Simulate all women
simWomen1000 = simObsWomen()

# ------------------------------------------------------------
# Generate initial Q0 from observed transitions
# ------------------------------------------------------------
transCounts = np.zeros((5, 5))
for seq in simWomen1000:
    for j in range(len(seq) - 1):
        if seq[j] != seq[j + 1]:
            transCounts[seq[j] - 1, seq[j + 1] - 1] += 1

Q0 = np.zeros((5, 5))
for i in range(5):
    row_sum = transCounts[i].sum()
    if row_sum > 0:
        Q0[i] = transCounts[i] / (row_sum * obsDel)
        Q0[i, i] = -Q0[i, np.arange(5) != i].sum()

# ------------------------------------------------------------
# Simulate path between two observations
# ------------------------------------------------------------
def simPath(stateInit, stateEnd, Q):
    while True:
        # rows 0-4: transition counts; row 5: sojourn times
        output = np.zeros((6, 5))
        currstate = stateInit
        while output[5].sum() < obsDel:
            if currstate != 5:
                wait = np.random.exponential(scale=-1 / Q[currstate - 1, currstate - 1])
            else:
                wait = obsDel
            if output[5].sum() + wait >= obsDel:
                output[5, currstate - 1] += obsDel - output[5].sum()
                break
            output[5, currstate - 1] += wait
            stateProps = np.delete(Q[currstate - 1], currstate - 1)
            statePropsN = stateProps / stateProps.sum()
            newstate = np.random.choice(
                [s for s in range(1, 6) if s != currstate],
                p=statePropsN
            )
            output[currstate - 1, newstate - 1] += 1
            currstate = newstate
        if currstate == stateEnd:
            return output


# ------------------------------------------------------------
# EM-like iteration to refine Q
# ------------------------------------------------------------
Qold = Q0.copy()
Qnew = np.zeros((5, 5))

while True:
    NSij = np.zeros((6, 5))
    # E-step: expected counts/sojourn times
    for seq in simWomen1000:
        for j in range(len(seq) - 1):
            NSij += simPath(seq[j], seq[j + 1], Qold)
    # M-step: update Q
    for i in range(5):
        sojourn = NSij[5, i]
        if sojourn > 0:
            for j in range(5):
                if i != j:
                    Qnew[i, j] = NSij[i, j] / sojourn
            Qnew[i, i] = -Qnew[i, np.arange(5) != i].sum()
    # check convergence
    Qconv = np.max(np.abs(Qnew - Qold).sum(axis=1))
    print(f"New Q found! Conv is {Qconv:.6f}")
    if Qconv < convCrit:
        break
    Qold = Qnew.copy()

Qfinal = Qnew

# Relative error against the true Q
Qdiff = np.abs(Qfinal - Q)
Qe = np.divide(Qdiff, Q, out=np.zeros_like(Qdiff), where=Q != 0)

print("Final estimated Q:\n", Qfinal)
print("Relative error Qe:\n", Qe)