set.seed(42)
library(expm)


#Number of women
n = 1000
#Initial state
x0 <- 1

#Matrix of probabilities
Pmat <- matrix(c(0.9915,0.005,0.0025,0    ,0.001,
                 0     ,0.986,0.005 ,0.004,0.005,
                 0     ,0    ,0.992 ,0.003,0.005,
                 0     ,0    ,0     ,0.991,0.009,
                 0     ,0    ,0     ,0    ,1),
               nrow = 5, ncol = 5, byrow = T)

#Function to calculate the jumps between state
# Input is initial state, output is new state at i+1
new_State <- function(initial){
  switch (initial,
          {
            state <- sample(c(1:5),1,prob = Pmat[1,])
          },
          {
            state <- sample(c(1:5),1,prob = Pmat[2,])
          },
          {
            state <- sample(c(1:5),1,prob = Pmat[3,])
          },
          {
            state <- sample(c(1:5),1,prob = Pmat[4,])
          },
          {
            state <- sample(c(1:5),1,prob = Pmat[5,])
          }
  )
  return(state)
}

#Matrix to hold output
result <- matrix(ncol = n)

#Set the first state for all women to x0
for (i in 1:n) {
  result[1,i] <- 1
}

#Variable to hold how many are still alive
remaining <- n

#Loop while there still are living women
while (remaining > 0) {
  tempRes <- integer(length = n)
  for (i in 1:n) {
    #For each woman, check if she were still alive last month
    if(result[nrow(result),i] != 5){
      #If she was, generate a new state
      tempRes[i] <- new_State(result[nrow(result),i])
      #If the new state is dead, subtract 1 from remaining
      if(tempRes[i] == 5) remaining <- remaining - 1
    }
    #If she were dead last month, she's still dead now
    else tempRes[i] <- 5
  }
  #Add the tempRes to the results
  result <- rbind(result,tempRes)
  #Debug message
  print(paste0("Month ",nrow(result), " done! Women remaining: ", remaining))
}
########################################
#Task 3 starts here
########################################

#Get the lifetime of each woman
#This is measured by the index of the last measurement before entering stage 5
simLifetimes <- apply(result==5,2,which.max) - 1
#Truncate the outliers at the 95th percentile
cutoff <- quantile(simLifetimes,0.95)
simLifetimesT <- ifelse(simLifetimes > cutoff, cutoff, simLifetimes)
#Get the empirical distribution of the truncated lifetimes
simLifeDistT <- table(factor(simLifetimesT, levels = 1:max(simLifetimesT)))
#Get the empirical mean lifetime, measured in months
simLifeMean <- mean(simLifetimes)

#Calculate the analytical distribution
π <- c(1,0,0,0)
Ps <- Pmat[1:4,1:4]
ps <- Pmat[1:4,5]
#Function that calculates probability of dying at time t
anaLifetime <- function(t){
  return(as.numeric(π %*% (Ps%^%t)%*%ps))
}
#Get the analytical mean lifetime
anaLifeMean <- π %*% solve(diag(4) - Ps) %*% rep(1,nrow(Ps))
#Calculate the probability of dying at all relevant t's
anaLifeDist <- numeric(length = max(simLifetimes))
for(i in 1:max(simLifetimes)) {
  anaLifeDist[i] <- anaLifetime(i)
}
#Truncate the analytical distribution
anaLifeDistT <- anaLifeDist
anaLifeDistT[(cutoff+1):length(anaLifeDistT)] <- 0
#Convert to count
anaLifeCountT <- anaLifeDistT * n

#Run a chisquare test
chisq.test(simLifeDistT[1:(cutoff-1)],p = anaLifeCountT[1:(cutoff-1)] / sum(anaLifeCountT[1:(cutoff-1)]))
#p-value = 0.6838

#Run a t test of the means
t.test(simLifetimes, mu = as.numeric(anaLifeMean))
#p-value = 0.4676

#K-S Test
#Normalize the distribution
anaLifeDistN <- anaLifeDist / sum(anaLifeDist)

#Calculate the analytical CDF
anaLifeCDF <- cumsum(anaLifeDistN)

#Get the empirical CDF
simLifeCDF <- ecdf(sort(simLifetimes))(1:max(simLifetimes))

#Plot the empirical vs theoretical CDFs
plot(1:max(simLifetimes), anaLifeCDF, type = "l", col = "blue", lwd = 2,
     xlab = "Months", ylab = "CDF", main = "Empirical vs Theoretical CDF")
lines(1:max(simLifetimes), simLifeCDF, col = "red", lwd = 2)
legend("bottomright", legend = c("Empirical", "Theoretical"),
       col = c("red", "blue"), lwd = 2, lty = 1)

#Calculate the KS test statistic
ksStat <- max(abs(simLifeCDF - anaLifeCDF))

#Calculate the adjusted test statistic, as shown on page 16 slide2bm1
ksAdju <- (sqrt(max(simLifetimes)) + 0.12 + 0.11 / sqrt(max(simLifetimes))) * ksStat
#Adjusted test statistic is  1.1476, which is below the critical value for a 5% significance level

#The first two tests yield p-values indicating that the simulation does in fact follow this distribution
#The KS-test agrees