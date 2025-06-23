#Import the reticulate library, allowing to set seed, and be reproducible with Python
library(reticulate)
np <- import("numpy")

# Create a Generator with MT19937
bitgen <- np$random$MT19937(seed = 42L)
rng <- np$random$Generator(bitgen)

#Create eventList, which holds all of the unhandled events
eventList <- data.frame(e_time = numeric(),
                        e_type = factor(levels = c("Dep", "Tra", "Arr")),
                        p_type = character(),
                        stringsAsFactors = F)
#Create compevent List, which holds all of the handled events
compeventList <- data.frame(e_time = numeric(),
                            e_type = factor(levels = c("Dep", "Tra", "Arr")),
                            p_type = character(),
                            stringsAsFactors = F)

#Create bed, a list of how many free beds are in each ward
bed <- numeric(length = 6)

#Create Penalties, which holds the sim output of how much penalty each ward has accrued
Penalties <- numeric(length = 6)

#Create Blocked, which holds the sim output of how many patients were blocked
Blocked <- numeric(length = 6)

#Similarly we hodl the following sim output
TotalArrivals <- numeric(length = 6)
DirectAdmissions <- numeric(length = 6)
Relocations <- numeric(length = 6)
BedsFull <- numeric(length = 6)


#Create endTime, which sets the period of time to be simulated, in days
endTime <- 395
burnin <- 30
#Given values
bed <- c(55,40,30,20,20,0)
ArrRate <- c(14.5,11,8,6.5,5,13)
μLoS <- c(2.9,4,4.5,1.4,3.9,2.2)
urgency <- c(7,5,2,10,5,0)
relProbs <- matrix(c(0.00,0.05,0.10,0.05,0.80,0.00,
                     0.20,0.00,0.50,0.15,0.15,0.00,
                     0.30,0.20,0.00,0.20,0.30,0.00,
                     0.35,0.30,0.05,0.00,0.30,0.00,
                     0.20,0.10,0.60,0.10,0.00,0.00,
                     0.20,0.20,0.20,0.20,0.20,0.00),
                   nrow = 6, ncol = 6, byrow = T,)

#Main function. withF is either 0 or 1, depending on if sim should include ward F
simMain <- function(withF){
  #Run init function
  simInit(withF)
  print("Init complete")
  #Repeat as long as there are more events, and the time haven't exceeded the end time
  while(nrow(eventList) > 0 && replace(eventList[1,1], is.na(eventList[1,1]), 0)<endTime){
    #Check the first event in list, run appropriate function
    
    #Get the values from list
    tempTim <- eventList[1,1]
    tempEve <- eventList[1,2]
    tempTyp <- eventList[1,3]
    #Add event to complist
    compeventList <<- rbind(compeventList,eventList[1,])
    #Remove event from list
    eventList <<- eventList[-1,]
    print(paste0("t = ",tempTim," out of ",endTime))
    switch (as.character(tempEve),
            "Arr" = {
              #Run Arrival function
              simArr(tempTim,tempTyp)
            },
            "Dep" = {
              #Run Departure function
              simDep(tempTim,tempTyp)
            },
            "Tra" = {
              #Run Departure function
              simTra(tempTim,tempTyp)
            }
    )
    
  }
  
  
  
  cat("-------- Performance Summary --------\n")
  # use sum(TotalArrivals) instead of sum(compeventList$e_type=="Arr")
  print(paste0(
    "Total patients blocked: ", sum(Blocked),
    " out of ", sum(TotalArrivals), " total patients. ",
    "Total Penalty: ", sum(Penalties)
  ))
  
  for (i in 1:6) {
    fracDirect   <- if (TotalArrivals[i] > 0) DirectAdmissions[i] / TotalArrivals[i] else NA
    pAllBedsFull <- if (TotalArrivals[i] > 0) BedsFull[i]            / TotalArrivals[i] else NA
    
    cat(sprintf("Patient type %s:\n", LETTERS[i]))
    cat(sprintf("  Total Arrivals           : %d\n", TotalArrivals[i]))
    cat(sprintf("  Direct Admissions        : %d\n", DirectAdmissions[i]))
    cat(sprintf("  Fraction Direct Admission: %.4f\n", fracDirect))
    cat(sprintf("  Relocations              : %d\n", Relocations[i]))
    cat(sprintf("  Blocked                  : %d\n", Blocked[i]))
    cat(sprintf("  Penalty Points           : %d\n", Penalties[i]))
    cat(sprintf("  P(All Beds Full)         : %.4f\n\n", pAllBedsFull))
  }
 
}

#Function to add event to eventList
simAddEvent <- function(e_time, e_type, p_type){
  #Create new event with input
  newEvent <- data.frame(e_time = e_time, 
                         e_type = factor(e_type, levels = levels(eventList$e_type)),
                         p_type = p_type,
                         stringsAsFactors = F)
  #Add it to the list
  eventList <<- rbind(eventList,newEvent)
  
  #Sort event list
  eventList <<- eventList[order(eventList$e_time, eventList$e_type), ]
}

#Function to initialize simulation
simInit <- function(withF){
  #Create list to hold inter arrival times
  IAT <- vector(mode = "list", length = 6)
  #Create list to hold actual arrival times
  ArrT <- vector(mode = "list", length = 6)
  #Repeat for each patient type
  for (i in 1:(5+withF)){
    while(sum(IAT[[i]]) < endTime){
      #Generate inter arrival times for a years worth of patients
      #IAT[[i]][length(IAT[[i]])+1] <- rexp(1,rate = ArrRate[i])
      IAT[[i]][length(IAT[[i]])+1] <- rng$exponential(scale = 1/ArrRate[i], size = 1L)
    }
    #Convert to actual arrival times
    ArrT[[i]] <- cumsum(IAT[[i]])
    #Create new event in event list for each patient. AddEvent function also sorts list
    for (j in 1:length(ArrT[[i]])) {
      simAddEvent(ArrT[[i]][j],"Arr",LETTERS[i])
    }
  }
}



# simInit <- function(withF){
#   max_time <- endTime
#   rate_total <- sum(ArrRate[1:(5 + withF)])
#   type_probs <- ArrRate[1:(5 + withF)] / rate_total
#   
#   time <- 0
#   
#   while (time < max_time) {
#     # Sample interarrival time
#     IAT <- rng$exponential(scale = 1 / rate_total, size = 1L)
#     time <- time + IAT
#     
#     if (time >= max_time) break
#     
#     # Sample patient type
#     type_index <- rng$choice(a = 1:(5 + withF), size = 1L, p = type_probs)
#     type_letter <- LETTERS[type_index]
#     
#     # Add event to queue
#     simAddEvent(time, "Arr", type_letter)
#   }
# }




#Function to handle arrivals
simArr <- function(time, type){

  #Check if the ward the patient arrived at has any available beds
  if(bed[which(LETTERS==type)] > 0) {
    #If bed is available, generate the length of stay for that patient
    #LoS <- rexp(1,rate = 1/μLoS[which(LETTERS==type)])
    LoS <- rng$exponential(scale = μLoS[which(LETTERS==type)], size = 1L)
    #Add a departure event to event list
    simAddEvent(time+LoS,"Dep",type)
    #Patient is now using a bed, so decrement the amount of beds in that ward
    bed[which(LETTERS==type)] <<- bed[which(LETTERS==type)] - 1
    
    #Collect stats only after burn-in period
    if (time >= burnin) {
      TotalArrivals[which(LETTERS==type)]     <<- TotalArrivals[which(LETTERS==type)] + 1
      DirectAdmissions[which(LETTERS==type)]  <<- DirectAdmissions[which(LETTERS==type)] + 1
    }
    
  } else {
    #If no bed is available, add a transfer event at the same time
    simAddEvent(time, "Tra", type)
    
    #Collect stats after burn-in period
    if (time >= burnin) {
      TotalArrivals[which(LETTERS==type)] <<- TotalArrivals[which(LETTERS==type)] + 1
      BedsFull[which(LETTERS==type)]      <<- BedsFull[which(LETTERS==type)] + 1
    }
  }
}



#Function to handle departures
simDep <- function(time, type){
  #Patient has departed, so the bed is now free. 
  #Increment the amount of beds in that ward
  bed[which(LETTERS==type)] <<- bed[which(LETTERS==type)] + 1
}

#Function to handle transfers
simTra <- function(time, type){
  #Sample length of stay of patient
  #LoS <- rexp(1,rate = 1/μLoS[which(LETTERS==type)])
  LoS <- rng$exponential(scale = μLoS[which(LETTERS==type)], size = 1L)
  #Sample transfer destination
  #Dest <- sample(LETTERS[1:6],size = 1,prob = relProbs[which(LETTERS==type),])
  Dest <- as.character(rng$choice(a = LETTERS[1:6], size = 1L, p = relProbs[which(LETTERS==type),]))
  #Check availability of dest
  if(bed[which(LETTERS==Dest)] > 0) {
    simAddEvent(time + LoS, "Dep", Dest)
    bed[which(LETTERS==Dest)] <<- bed[which(LETTERS==Dest)] - 1
    
    #Collect stats after burn-in period
    if (time >= burnin) {
      #Add penalty
      Penalties[which(LETTERS==type)] <<- Penalties[which(LETTERS==type)] + urgency[which(LETTERS==type)]
      #Track relocations
      Relocations[which(LETTERS==type)] <<- Relocations[which(LETTERS==type)] + 1
    }
  } else {
    if (time >= burnin) {
    #Add penalty
    Penalties[which(LETTERS==type)] <<- Penalties[which(LETTERS==type)] + urgency[which(LETTERS==type)]
    #Track blocked
    Blocked[which(LETTERS==type)] <<- Blocked[which(LETTERS==type)]     + 1
    }
  }
}

# Things done differently than Marius:
#I have added more stats
# I have included a burn-in period.

#Run the simulation, argument is if Ward F is included (1) or not (0)
simMain(0)

##Results: Ward A and D appear very healthy. Especially D.



#### RUN MULTIPLE SIMULATIONS ####################################

# Run 10 simulations and collect results
runMultipleSimulations <- function(n_runs = 10, withF = 0) {
  # Set up storage for cumulative results
  allDirectAdmissions <- matrix(0, nrow = n_runs, ncol = 6)
  allRelocations <- matrix(0, nrow = n_runs, ncol = 6)
  allPenalties <- matrix(0, nrow = n_runs, ncol = 6)
  allBlocked <- matrix(0, nrow = n_runs, ncol = 6)
  allTotalArrivals <- matrix(0, nrow = n_runs, ncol = 6)
  
  for (run in 1:n_runs) {
    cat(sprintf("\n======== Running Simulation %d ========\n", run))
    
    # Reset simulation state
    eventList <<- data.frame(e_time = numeric(),
                             e_type = factor(levels = c("Dep", "Tra", "Arr")),
                             p_type = character(),
                             stringsAsFactors = F)
    compeventList <<- data.frame(e_time = numeric(),
                                 e_type = factor(levels = c("Dep", "Tra", "Arr")),
                                 p_type = character(),
                                 stringsAsFactors = F)
    bed <<- c(55,40,30,20,20,0)
    Penalties <<- numeric(length = 6)
    Blocked <<- numeric(length = 6)
    TotalArrivals <<- numeric(length = 6)
    DirectAdmissions <<- numeric(length = 6)
    Relocations <<- numeric(length = 6)
    BedsFull <<- numeric(length = 6)
    
    # Run the simulation
    simMain(withF)
    
    # Store results
    allDirectAdmissions[run, ] <- DirectAdmissions
    allRelocations[run, ] <- Relocations
    allPenalties[run, ] <- Penalties
    allBlocked[run, ] <- Blocked
    allTotalArrivals[run, ] <- TotalArrivals
  }
  
  # Compute and print averages
  cat("\n\n======== AVERAGED RESULTS OVER", n_runs, "SIMULATIONS ========\n")
  for (i in 1:6) {
    avgDirect <- mean(allDirectAdmissions[, i])
    avgReloc <- mean(allRelocations[, i])
    avgPenalty <- mean(allPenalties[, i])
    avgBlocked <- mean(allBlocked[, i])
    avgArrivals <- mean(allTotalArrivals[, i])
    fracDirect <- if (avgArrivals > 0) avgDirect / avgArrivals else NA
    pAllBedsFull <- if (avgArrivals > 0) avgBlocked / avgArrivals else NA
    
    cat(sprintf("Patient type %s:\n", LETTERS[i]))
    cat(sprintf("  Avg Total Arrivals        : %.2f\n", avgArrivals))
    cat(sprintf("  Avg Direct Admissions     : %.2f\n", avgDirect))
    cat(sprintf("  Avg Relocations           : %.2f\n", avgReloc))
    cat(sprintf("  Avg Blocked               : %.2f\n", avgBlocked))
    cat(sprintf("  Avg Penalty Points        : %.2f\n", avgPenalty))
    cat(sprintf("  Fraction Direct Admission : %.4f\n", fracDirect))
    cat(sprintf("  P(All Beds Full)          : %.4f\n\n", pAllBedsFull))
  }
}


runMultipleSimulations(1, 0)











########## WORK IN PROGRESS #######################################







bed <- c(55,40,30,20,20,0)

# ─── 1) Re-use your simInit to seed the model ─────────────────────────────────
simInit(withF = 0)    # 0 = no ward F, or 1 if you want F included

# ─── 2) Prepare storage for times & bed-levels ────────────────────────────────
times    <- numeric()
beds_mat <- matrix(nrow = 0, ncol = 6)
colnames(beds_mat) <- LETTERS[1:6]

# ─── 3) Run your event loop “by hand”, recording before each step ────────────
while (nrow(eventList) > 0 && eventList$e_time[1] < endTime) {
  # a) snapshot
  t0       <- eventList$e_time[1]
  times    <- c(times, t0)
  beds_mat <- rbind(beds_mat, bed)
  
  # b) pop and dispatch
  ev <- eventList[1, ]
  eventList <<- eventList[-1, ]
  compeventList <<- rbind(compeventList, ev)
  
  switch(as.character(ev$e_type),
         "Arr" = simArr(t0, ev$p_type),
         "Dep" = simDep(t0, ev$p_type),
         "Tra" = simTra(t0, ev$p_type)
  )
}

# ─── 4) Finally, plot all six wards at once ───────────────────────────────────
matplot(times, beds_mat, type = "l", lty = 1,
        xlab = "Time (days)", ylab = "Free beds",
        main = "Beds available over time by ward")
legend("topright", legend = colnames(beds_mat),
       col = 1:6, lty = 1, cex = 0.8)
























#Try with F
bed <- c(49,40,30,16,20,10)
simMain(1)









 ## Run 10 simulations

# Run 10 simulations and collect results
runMultipleSimulations <- function(n_runs = 10, withF = 0) {
  # Set up storage for cumulative results
  allDirectAdmissions <- matrix(0, nrow = n_runs, ncol = 6)
  allRelocations <- matrix(0, nrow = n_runs, ncol = 6)
  allPenalties <- matrix(0, nrow = n_runs, ncol = 6)
  allBlocked <- matrix(0, nrow = n_runs, ncol = 6)
  allTotalArrivals <- matrix(0, nrow = n_runs, ncol = 6)
  
  for (run in 1:n_runs) {
    cat(sprintf("\n======== Running Simulation %d ========\n", run))
    
    # Reset simulation state
    eventList <<- data.frame(e_time = numeric(),
                             e_type = factor(levels = c("Dep", "Tra", "Arr")),
                             p_type = character(),
                             stringsAsFactors = F)
    compeventList <<- data.frame(e_time = numeric(),
                                 e_type = factor(levels = c("Dep", "Tra", "Arr")),
                                 p_type = character(),
                                 stringsAsFactors = F)
    bed <<- c(55,40,30,20,20,0)
    Penalties <<- numeric(length = 6)
    Blocked <<- numeric(length = 6)
    TotalArrivals <<- numeric(length = 6)
    DirectAdmissions <<- numeric(length = 6)
    Relocations <<- numeric(length = 6)
    BedsFull <<- numeric(length = 6)
    
    # Run the simulation
    simMain(withF)
    
    # Store results
    allDirectAdmissions[run, ] <- DirectAdmissions
    allRelocations[run, ] <- Relocations
    allPenalties[run, ] <- Penalties
    allBlocked[run, ] <- Blocked
    allTotalArrivals[run, ] <- TotalArrivals
  }
  
  # Compute and print averages
  cat("\n\n======== AVERAGED RESULTS OVER", n_runs, "SIMULATIONS ========\n")
  for (i in 1:6) {
    avgDirect <- mean(allDirectAdmissions[, i])
    avgReloc <- mean(allRelocations[, i])
    avgPenalty <- mean(allPenalties[, i])
    avgBlocked <- mean(allBlocked[, i])
    avgArrivals <- mean(allTotalArrivals[, i])
    fracDirect <- if (avgArrivals > 0) avgDirect / avgArrivals else NA
    pAllBedsFull <- if (avgArrivals > 0) avgBlocked / avgArrivals else NA
    
    cat(sprintf("Patient type %s:\n", LETTERS[i]))
    cat(sprintf("  Avg Total Arrivals        : %.2f\n", avgArrivals))
    cat(sprintf("  Avg Direct Admissions     : %.2f\n", avgDirect))
    cat(sprintf("  Avg Relocations           : %.2f\n", avgReloc))
    cat(sprintf("  Avg Blocked               : %.2f\n", avgBlocked))
    cat(sprintf("  Avg Penalty Points        : %.2f\n", avgPenalty))
    cat(sprintf("  Fraction Direct Admission : %.4f\n", fracDirect))
    cat(sprintf("  P(All Beds Full)          : %.4f\n\n", pAllBedsFull))
  }
}


runMultipleSimulations(1, 0)
