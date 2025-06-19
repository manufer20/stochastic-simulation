set.seed(1234)

n <- 1000
Q <- matrix(c(-0.0085, 0.0050, 0.0025, 0.0000, 0.0010,
               0.0000,-0.0140, 0.0050, 0.0040, 0.0050,
               0.0000, 0.0000,-0.0080, 0.0030, 0.0050,
               0.0000, 0.0000, 0.0000,-0.0090, 0.0090,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000),
            ncol = 5, nrow = 5, byrow = T)
#Time between observations
obsDel <- 48

#Convergence criterion
convCrit <- 0.001

#Function to simulate a single woman
sim1Woman <- function(){
  #Create matrix to hold results. First column is time, second column is state
  #Time unit is months
  output <- matrix(ncol = 2)
  #Initial state is 1, initial time is 0
  output[1,] <- c(0,1)
  currstate <- 1
  #Index variable
  i = 1

  while (currstate != 5) {
    #Sample the waiting time
    wait <- rexp(1, rate = -Q[currstate,currstate])
    #Set the time of the next transition to the time of last transition + the wait time
    output <- rbind(output, c(output[i,1] + wait, NA)) 
    #Get the next state
    stateProps <- Q[currstate,-currstate]
    statePropsN <- stateProps / sum(stateProps)
    currstate <- sample(c(1,2,3,4,5)[-currstate], 1, prob = statePropsN)
    output[i+1,2] <- currstate
    #Increment index
    i <- nrow(output)
  }
  return(output)
}

#Function to simulate n women, but only looking at observed states
simObsWomen <- function(){
  #Variable to hold output
  output <- vector(mode = "list", length = n)
  for (i in 1:n) {
    #Simulate a woman
    woman <- sim1Woman()
    #Set an index
    ind <- 1
    #Set the current state (starts at 1)
    currstate <- 1
    #Add the current state to the output
    output[[i]][1] <- currstate
    #Repeatedly check the state every obsDel month, until death
    while (currstate != 5) {
      #Check the state the woman is in at time ind * obsDel
      currstate <- woman[tail(which(woman[,1] < (ind * obsDel)),1), 2]
      #Update the index
      ind <- ind + 1
      #Record the output
      output[[i]][ind] <- currstate
    }
  }
  return(output)
}

simWomen1000 <- simObsWomen()

#Generate a Q0 based on the observed data
transCounts <- matrix(0,nrow = 5, ncol = 5)
#Loop over all observations
for (i in simWomen1000) {
  for (j in 1:(length(i)-1)) {
    #If state changes, increment transCounts
    if(i[j] != i[j+1]) transCounts[i[j], i[j+1]] <- transCounts[i[j], i[j+1]] + 1
  }
}

#Convert counts to rates
Q0 <- matrix(0,nrow = 5, ncol = 5)
for(i in 1:5){
  if(sum(transCounts[i,]) > 0) {
    #Crude estimation of rate
    Q0[i,] <- transCounts[i,] / (sum(transCounts[i,]) * obsDel)
    #Calculate the diagonal separately, to make the row sum correct
    Q0[i,i] <- -sum(Q0[i,-i])
  }
}

#Function to simulate path between 2 observations
simPath <- function(stateInit, stateEnd, Q) {
  repeat{
    #Create a matrix to hold results. First 5 rows are transition counts, row 6 is sojourn times
    output <- matrix(rep(0,30), ncol = 5, nrow = 6)
    #Set the current state
    currstate <- stateInit
    while(sum(output[6,] < obsDel)){
      #Sample the wait time
      if(currstate != 5) {
        wait <- rexp(1, rate = -Q[currstate,currstate])
      } else {
        wait <- obsDel
      }
      
      #If next jump overshoots the interval, adjust and break
      if((sum(output[6,])+wait) >= obsDel) {
        output[6,currstate] <- output[6,currstate] + (obsDel - sum(output[6,]))
        break
      }
      
      #Add wait time to sojourn time
      output[6,currstate] <- output[6,currstate] + wait
      #Get new state, and increment the transition
      stateProps <- Q[currstate,-currstate]
      statePropsN <- stateProps / sum(stateProps)
      newstate <- sample(c(1,2,3,4,5)[-currstate], 1, prob = statePropsN)
      output[currstate, newstate] <- output[currstate,newstate] + 1
      #Update currstate
      currstate <- newstate
      #Repeat until the time between observations have passed
    }
    #Path has been found, test to see if path is accepted. If it has, return output
    if(currstate == stateEnd) return(output)
    #If rejected, repeat from the top
  }
}

#Variable to hold the old Q
Qold <- Q0
#Variable to hold the new Q
Qnew <- matrix(0,nrow = 5, ncol = 5)

#The following is looped until the simulation converges
repeat{
  #Variable to hold the outputs, with the same format as above
  NSij <- matrix(0, nrow = 6, ncol = 5)
  
  #print("Loop started")
  
  #Run the simulation over all women
  for(i in simWomen1000) {
    #For each pair of observations
    for (j in 1:(length(i)-1)) {
      NSij <- NSij + simPath(i[j], i[j+1], Qold)
      #print("Simulated pair done")
    }
    #print("Simulated woman done")
  }
  
  #Calculate the new Q
  for (i in 1:5) {
    for (j in 1:5) {
      if(i != j) Qnew[i,j] <- NSij[i,j] / NSij[6,i]
    }
    Qnew[i,i] <- -sum(Qnew[i,-i])
  }
  
  #Calculate the convergence criterion
  Qdiff <- abs(Qnew - Qold)
  Qconv <- max(rowSums(Qdiff))
  
  print(paste0("New Q found! Conv is ", Qconv))
  
  #If convergence criterion is reached, break out of the loop
  if(Qconv < convCrit) break
  
  #Set old Q to new Q
  Qold <- Qnew
}

#The final estimated Q
Qnew

#Calculate the difference between Q and Qnew
Qdiff <- abs(Qnew-Q)

#Calculate the relative error
Qe <- Qdiff / Q
