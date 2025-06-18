###################################################
# Project 1 Part 2
###################################################

library(expm)

# Define Q matrix
Q <- matrix(c(
  -0.0085, 0.005, 0.0025, 0, 0.001,
  0, -0.014, 0.005, 0.004, 0.005,
  0, 0, -0.008, 0.003, 0.005,
  0, 0, 0, -0.009, 0.009,
  0, 0, 0, 0, 0
), nrow = 5, byrow = TRUE)

# Function to simulate a single woman
SimulateContTimeWoman <- function(Q) {
  
  #Initialize
  current_state <- 1  # Starts in state 1
  time <- 0 # time starts at month 0
  
  #Prepare storage
  state_evolution <- c(current_state)
  times <- c(time)
  
  #When still alive
  while (current_state != 5) {
    
    #Time spent in current state
    rate <- -Q[current_state, current_state] #Exit rate from current state
    holding_time <- rexp(1, rate) #Draws a random time using current state rate.
    time <- time + holding_time
    
    #Sample the next transition
    probs <- Q[current_state, ] #transition probability from current state to others
    probs[current_state] <- 0   #No self transitions
    probs <- probs / sum(probs) #Normalize the rates
    current_state <- sample(1:5, 1, prob = probs) #Randomly select state based on probability.
    
    #Record the new state and time
    state_evolution <- c(state_evolution, current_state)
    times <- c(times, time)
  }
  
  #Return total time, state evolution and state transition times.
  return(list(total_time = time, state_evolution = state_evolution, times = times))
}



# Simulate 1000 women
set.seed(123)
women <- 1000
results <- replicate(women, SimulateContTimeWoman(Q), simplify = FALSE)

months_alive <- numeric(women) # Initialize empty vector to store lifetimes

# Extract total months women are alive
for (i in 1:women) {
  months_alive[i] <- results[[i]]$total_time
}


# Histogram of months alive
hist(months_alive, breaks = 40, col = "skyblue",
     main = "Lifetime Distribution (CTMC)",
     xlab = "Months until death")

# Summary statistics
mean_months_alive <- mean(months_alive)
sd_months_alive <- sd(months_alive)

# 95% confidence interval for mean
alpha <- 0.05
t <- qt(1 - alpha/2, df = women - 1)  
mean_CI <- mean_months_alive + c(-1, 1) * t * sd_months_alive / sqrt(women)

#Confidence interval for sd
df <- women-1
chi_low <- qchisq(1 - alpha/2, df)
chi_high <- qchisq(alpha/2, df)
var_lower <- (df * sd_months_alive^2) / chi_low
var_upper <- (df * sd_months_alive^2) / chi_high
sd_CI <- sqrt(c(var_lower, var_upper))


### Proportion of women with distant cancer reappearance after 30.5 months ###
counter <- 0 # Initialize counter

# Loop through each simulated woman
for (i in 1:women) {
  state_evolution <- results[[i]]$state_evolution
  times <- results[[i]]$times
  
  #Check if state 3 appears within 30.5 months, if so increase count
  for (j in 2:length(state_evolution)) { #Note we skip state 1, since it will always be 1.
    if (state_evolution[j] == 3 && times[j] <= 30.5) { 
      counter <- counter + 1
      break  # Leave loop after woman enters state 3
    }
  }
}

# Compute Proportion of distant reappearance
counter/women



######## TASK 8 ############################################################################

#The continuous phase time distribution is given by
# F_T(t)=1- p0*exp(Qs*t)*1

Qs <- Q[1:4, 1:4]      #Submatrix without state 5
p0 <- c(1, 0, 0, 0)    # All start in state 1
ones <- rep(1, 4)

#Define the theoretical distribution function
FT <- function(t) {1 - as.numeric(p0 %*% expm(Qs * t) %*% ones)}


# Find the empirical CDF
xvalues <- sort(months_alive)  # Empirical x-axis
theoretical_cdf <- sapply(xvalues, FT)
empirical_cdf <- ecdf(months_alive)(xvalues)


plot(xvalues, theoretical_cdf, type = "l", col = "blue", lwd = 2,
     xlab = "Months", ylab = "CDF", main = "Empirical vs Theoretical CDF")
lines(xvalues, empirical_cdf, col = "red", lwd = 2)
legend("bottomright", legend = c("Empirical", "Theoretical"),
       col = c("red", "blue"), lwd = 2, lty = 1)


### Perform Kolmogorov Smirnov test ###
#Test statistic calculated using formula on page 15 slide2bm1
ks_stat <- max(abs(empirical_cdf - theoretical_cdf))

#The adjusted statistic is calculated using formula on page 16 slide2bm1
adjusted_ks <- (sqrt(women) + 0.12 + 0.11 / sqrt(women)) * ks_stat
adjusted_ks

#Remember critical value for a 5% significanse level is 1.358 (page 16 slide2bm1)





######### TASK 9 #####################################################################

#Introduce a preventative treatment

# Define new Q matrix
Q_treated <- matrix(c(
  -0.00475, 0.0025, 0.00125, 0, 0.001,
  0, -0.007, 0, 0.002, 0.005,
  0, 0, -0.008, 0.003, 0.005,
  0, 0, 0, -0.009, 0.009,
  0, 0, 0, 0, 0
), nrow = 5, byrow = TRUE)

# Simulate women with new treatment
set.seed(456)  
results_treated <- replicate(women, SimulateContTimeWoman(Q_treated), simplify = FALSE)

months_alive_treated <- numeric(women) # Initialize empty vector to store lifetimes

# Extract total months women are alive
for (i in 1:women) {
  months_alive_treated[i] <- results_treated[[i]]$total_time
}

months_alive_treated <- sapply(results_treated, function(res) res$total_time)

#Define the Kaplan-Meier estimator as in the project description
Shat <- function(month, months_alive){
  N <- length(months_alive) #Corresponds to number of women
  d <- sum(months_alive <= month) #Number of women died before threshold
  
  #The survival fraction
  fraction <- (N - d) / N 
  return(fraction)
}


# Compare survival curves
tvalues <- seq(0, max(months_alive, months_alive_treated), by = 1)
SurvFrac_untreated <- KaplanMeier(months_alive, tvalues)
SurvFrac_treated <- KaplanMeier(months_alive_treated, tvalues)

# Plot Kaplan-Meier curves
plot(tvalues, SurvFrac_untreated, type = "l", lwd = 2, col = "blue",
     ylim = c(0, 1), xlab = "Months", ylab = "Survival Probability",
     main = "Kaplan-Meier Survival Curves")
lines(tvalues, SurvFrac_treated, col = "green", lwd = 2)
legend("topright", legend = c("Untreated", "Treated"),
       col = c("blue", "green"), lty = 1, lwd = 2)









