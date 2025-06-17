
#############################################
# Project 1
#############################################

############ TASK 1 #######################################
#Load libraries
library(matrixcalc)

# Define the probability matrix
P <- matrix(c(0.9915, 0.005, 0.0025, 0, 0.001,
              0, 0.986, 0.005, 0.004, 0.005,
              0, 0, 0.992, 0.003, 0.005,
              0, 0, 0, 0.991, 0.009,
              0, 0, 0, 0, 1), nrow = 5, byrow = TRUE)

#Define parameters
women <- 1000
states <- nrow(P)

# Simulate the cancer evolution of women until death
SimulateOneWoman <- function(P){
  
  # Initialize
  evolution <- c()
  current_state <- 1 # All women start in first state
  
  #Evolution, when still alive
  while (current_state !=5){
    evolution <- c(evolution, current_state) #Tracks the state of the woman 
    
    #Sample next state from the probability distribution defined by current row of P
    current_state <- sample(1:states, size = 1, prob = P[current_state,])
  }
  
  #When a woman dies we add the "death state" to the evolution
  evolution <- c(evolution, 5) #Here we include death state
  
  #Return the number of months a woman is alive and state evolution.
  return(list(months_alive = length(evolution), evolution = evolution))
} 



#Simulate the cancer evolution of all women
SimulateAllWomen <- function(n, P){
  
  #Initialize
  months_alive <- numeric(n)
  evolution_list <- vector("list",n)
  longest_lifespan <- 0
  
  #Loop through one woman at a time
  for (i in 1:n){
    result <- SimulateOneWoman(P)
    months_alive[i] <- result$months_alive
    evolution_list[[i]] <- result$evolution
    longest_lifespan <- max(longest_lifespan, result$months_alive)
  }
  
  #Store cancer evolution for each woman in a square matrix
  evolutions <- matrix(5, nrow=n, ncol=longest_lifespan)
  for (i in 1:n){
    length_i <- length(evolution_list[[i]])
    evolutions[i, 1:length_i] <- evolution_list[[i]]
  }
  
  return(list(months_alive = months_alive, evolutions = evolutions))
  
}


# Proportions of each state
StateProportions <- function(evolutions, states){
  
  #Initialize
  months <- ncol(evolutions)
  state_proportions <- matrix(0, nrow = months, ncol= states)
  
  #Convert state count into proportion
  for (t in 1:months){
    for (s in 1:states){
      state_proportions[t,s] <- sum(evolutions[,t] == s)/nrow(evolutions)
    }
  }
  return(state_proportions)
}



#Perform simulation
set.seed(123)
results <- SimulateAllWomen(women, P)

#Compute the proportions in each state
proportions <- StateProportions(results$evolutions, states)
matplot(proportions, type = "l", lty = 1, col = 1:5,
        xlab = "Months", ylab = "Proportion", 
        main = "State Proportions Over Time")
legend("right", legend = paste("State", 1:5), col = 1:5, lty = 1)

#Plot histogram of the month's women are alive
hist(results$months_alive, breaks = 50, main = "Lifetime Distribution After Surgery", 
     xlab = "Months until death", col = "lightblue")

# Proportion of women where the cancer reappears locally
LocalReappearance <- apply(results$evolutions, 1, function(row) any(row == 2))
LocalReappearance_Proportion <- mean(LocalReappearance)

cat("Proportion of women with local recurrence:", LocalReappearance_Proportion, "\n")


######## TASK 2 ######################################################

# Set evaluation time
month <- 120

# Everyone starts in state 1
initial_distribution <- c(1, 0, 0, 0, 0)

# Theoretical distribution at month = 120
theoretical_distribution <- initial_distribution %*% matrix.power(P, month)

# Empirical distribution at month=120
empirical_distribution <- proportions[month, ]

# Compare the two distributions
rbind(Empirical = empirical_distribution,
                    Theoretical = theoretical_distribution)


#Comparesion of the two distributions
comparison <- rbind(Empirical = empirical_distribution,
                     Theoretical = as.numeric(theoretical_distribution))

#Barplot visualizing the comparison
barplot(comparison,
        beside = TRUE,
        col = c("skyblue", "red"),
        names.arg = paste("State", 1:5),
        main = "Empirical vs. Theoretical Distribution at Month 120",
        ylab = "Proportion",
        ylim = c(0, max(comparison) * 1.1),
        legend.text = TRUE,
        args.legend = list(x = "topright", bty = "n"))



# Chi-squared test for verification
observed_counts <- empirical_distribution * women
chisq_result <- chisq.test(x = observed_counts, p = as.numeric(theoretical_distribution))
print(chisq_result)


######### TASK 3 ######################################################


# Setup
library(matrixcalc)

# The distribution is given by P(T=t)=Pi(P_s)^t*p_s
# We define the relevant parameters
Pi <- c(1, 0, 0, 0) # All women start in state 1
P_s <- P[1:4, 1:4] #Submatrix of transitions between all states except for 5
p_s <- P[1:4, 5] # Probability of transitioning to the death state 5.


#Define the empirical lifetime distribution
EmpiricalLifetimeDistribution <- function(t) {
  as.numeric(Pi %*% matrix.power(P_s, t) %*% p_s)
}

# Generate theoretical probabilities
max_t <- max(results$months_alive)
xvalues <- 1:max_t
theoretical_probs <- sapply(xvalues, EmpiricalLifetimeDistribution)


# Plot empirical vs theoretical PDF
hist(results$months_alive, breaks = 30, probability = TRUE,
     main = "Empirical vs Theoretical PDF of Lifetime",
     xlab = "Months until death", col = "skyblue")
lines(xvalues, theoretical_probs, col = "blue", lwd = 2)
legend("topright", legend = c("Empirical", "Theoretical"),
       col = c("skyblue", "blue"), lwd = 2, lty = 1)


# Plot empirical vs theoretical CDF
empirical_cdf <- ecdf(results$months_alive)(xvalues)
theoretical_cdf <- cumsum(theoretical_probs)
plot(xvalues, theoretical_cdf, type = "l", col = "blue", lwd = 2,
     xlab = "Months", ylab = "CDF", main = "Empirical vs Theoretical CDF")
lines(xvalues, empirical_cdf, col = "red", lwd = 2)
legend("bottomright", legend = c("Empirical", "Theoretical"),
       col = c("red", "blue"), lwd = 2, lty = 1)


### Perform Kolmogorov Smirnov test ###
#Test statistic calculated using formula on page 15 slide2bm1
ks_stat <- max(abs(empirical_cdf - theoretical_cdf))

#The adjusted statistic is calculated using formula on page 16 slide2bm1
adjusted_ks <- (sqrt(max_t) + 0.12 + 0.11 / sqrt(max_t)) * ks_stat
adjusted_ks

#Remember critical value for a 5% significanse level is 1.358 (page 16 slide2bm1)


##### TASK 4 ###################################################################

# Number of accepted women we want
women_accepted <- 1000

# Initialize
lifespan_accepted <- numeric(women_accepted)
accepted <- 0
set.seed(123)  # for reproducibility

#Run until we have 1000 accepted women
while (accepted < women_accepted) {
  woman <- SimulateOneWoman(P)
  evo <- woman$evolution #Store the woman's state_evolution
  
  # Check if she survives past 12 months
  if (length(evo) > 12) {
    
    # Extract only the first 12 months
    first_12 <- evo[1:12]
    
    # Check for recurrence local (state 2) or distant (state 3)
    if (any(first_12 == 2) || any(first_12 == 3)) {
      accepted <- accepted + 1
      lifespan_accepted[accepted] <- woman$months_alive 
    }
  }
}

# Estimate expected lifetime of women meeting the condition
mean_lifetime <- mean(lifespan_accepted)
sd_lifetime <- sd(lifespan_accepted)

# Plot lifespan of women surviving past 12 months
hist(lifespan_accepted, breaks = 40, col = "skyblue",
     main = "Lifetime Distribution (Given Recurrence in First 12 Months)",
     xlab = "Months alive")

cat("Estimated expected lifetime (given recurrence within 12 months):", mean_lifetime, "months\n")
cat("Standard deviation:", sd_lifetime, "months\n")




######### TASK 5 ##################################################################################
set.seed(123)

# Parameters
n_simulations <- 100        # Number of replications
n_women <- 200              # Number of women per replication
death_month_threshold <- 350 # Death within the chosen number of months

# Store results
death_proportions <- numeric(n_simulations)
mean_months_alive <- numeric(n_simulations)

# Perform 100 simulations 
for (i in 1:n_simulations) {
  
  #Simulate 200 women
  sim <- SimulateAllWomen(n_women, P)
  months_alive <- sim$months_alive  
  
  # Proportion who die within 350 months
  death_proportions[i] <- mean(months_alive <= death_month_threshold)
  
  # Mean lifespan for the replication
  mean_months_alive[i] <- mean(months_alive)
}

### CRUDE MONTE CARLO ESTIMATOR ###-------------------------------------------

crudeMC_mean <- mean(death_proportions)
crudeMC_var <- var(death_proportions)

crudeMC_mean
crudeMC_var


### Confidence interval
alpha <- 0.05
q <- qt(1 - alpha/2, df = n_simulations - 1)

crude_CI <- c(
  crudeMC_mean - q * sqrt(crudeMC_var) / sqrt(n_simulations),
  crudeMC_mean + q * sqrt(crudeMC_var) / sqrt(n_simulations)
)

crude_CI

### CONTROL VARIATE ESTIMATOR ###----------------------------------------------

# Theoretical expected lifetime: E[T] = π (I - P_s)^(-1) 1
Pi <- c(1, 0, 0, 0)                     # Initial state distribution
P_s <- P[1:4, 1:4]                      # Submatrix without death state
p_s <- P[1:4, 5]                        # Probability of dying from each state
I <- diag(4)                            # Identity matrix
theor_mean_months_alive <- as.numeric(Pi %*% solve(diag(4)-P_s) %*% rep(1,4))

# Estimate c using formula on page 15 slide7m1
c <- -cov(death_proportions, mean_months_alive) / var(mean_months_alive)

Z <- death_proportions + c * (mean_months_alive - theor_mean_months_alive)

# Estimate mean and variance with control variate
cv_mean <- mean(Z)
cv_var <- var(Z)

cv_mean
cv_var

# Compare variance reduction
reduction_ratio <- crudeMC_var / cv_var
cat("Variance reduction factor:", reduction_ratio, "\n")

### Confidence interval with covariates
alpha <- 0.05
q <- qt(1 - alpha/2, df = n_simulations - 1)


# Control variate confidence interval
cv_CI <- c(
  cv_mean - q * sqrt(cv_var) / sqrt(n_simulations),
  cv_mean + q * sqrt(cv_var) / sqrt(n_simulations)
)

cv_CI



#### Results ####
#Means
cat("Crude MC estimate of probability:", crudeMC_mean, "\n")
cat("Control variate estimate of probability:", cv_mean, "\n")

#Variance 
cat("Crude MC variance:", crudeMC_var, "\n")
cat("Control variate variance:", cv_var, "\n")

#CI
cat("Crude CI:", crude_CI, "\n")
cat("Control Variate CI:", cv_CI, "\n")

#CI Width
cat("Crude CI width:", diff(crude_CI), "\n")
cat("Control Variate CI width:", diff(cv_CI), "\n")


#Variance reduction factor
reduction_ratio <- crudeMC_var / cv_var
cat("Variance reduction factor:", reduction_ratio, "\n")


