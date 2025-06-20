set.seed(0)

#Create eventList, which holds all of the unhandled events
eventList <- data.frame(e_time = numeric(),
                        e_type = factor(levels = c("Dep", "Tra", "Arr")),
                        p_type = character(),
                        stringsAsFactors = F)
oldeventList <- data.frame(e_time = numeric(),
                           e_type = factor(levels = c("Dep", "Tra", "Arr")),
                           p_type = character(),
                           stringsAsFactors = F)

#Create bed, a list of how many free beds are in each ward
bed <- numeric(length = 6)

#Create Penalties, which holds the sim output of how much penalty each ward has accrued
Penalties <- numeric(length = 6)

#Create Blocked, which holds the sim output of how many patients were blocked
Blocked <- 0

#Create endTime, which sets the period of time to be simulated, in days
endTime <- 10
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
  #Repeat as long as there are more events, and the time haven't exceeded the end time
  while(nrow(eventList) > 0 && replace(eventList[1,1], is.na(eventList[1,1]), 0)<endTime){
    #Check the first event in list, run appropriate function
    
    #Get the values from list
    tempTim <- eventList[1,1]
    tempEve <- eventList[1,2]
    tempTyp <- eventList[1,3]
    #Add event to oldlist
    oldeventList <<- rbind(oldeventList,eventList[1,])
    #Remove event from list
    eventList <<- eventList[-1,]
    
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
  print(paste0("Penalty ward ",LETTERS[1:6],": ", Penalties))
  print(paste0("Total patients blocked: ", Blocked, " Total Penalty: ", sum(Penalties)))
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

simInit <- function(withF){
  #Create list to hold inter arrival times
  IAT <- vector(mode = "list", length = 6)
  #Create list to hold actual arrival times
  ArrT <- vector(mode = "list", length = 6)
  #Repeat for each patient type
  for (i in 1:(5+withF)){
    while(sum(IAT[[i]]) < endTime){
      #Generate inter arrival times for a years worth of patients
      IAT[[i]][length(IAT[[i]])+1] <- rexp(1,rate = ArrRate[i])
    }
    #Convert to actual arrival times
    ArrT[[i]] <- cumsum(IAT[[i]])
    #Create new event in event list for each patient. AddEvent function also sorts list
    for (j in 1:length(ArrT[[i]])) {
      simAddEvent(ArrT[[i]][j],"Arr",LETTERS[i])
    }
  }
}

#Function to handle arrivals
simArr <- function(time, type){
  if(bed[which(LETTERS==type)] > 0) {
    LoS <- rexp(1,rate = 1/μLoS[which(LETTERS==type)])
    simAddEvent(time+LoS,"Dep",type)
    bed[which(LETTERS==type)] <<- bed[which(LETTERS==type)] - 1
  } else {
    simAddEvent(time, "Tra", type)
  }
}

#Function to handle departures
simDep <- function(time, type){
  bed[which(LETTERS==type)] <- bed[which(LETTERS==type)] + 1
}

#Function to handle transfers
simTra <- function(time, type){
  #Add penalty
  Penalties[which(LETTERS==type)] <<- Penalties[which(LETTERS==type)] + urgency[which(LETTERS==type)]
  #Sample length of stay of patient
  LoS <- rexp(1,rate = 1/μLoS[which(LETTERS==type)])
  #Sample transfer destination
  Dest <- sample(LETTERS[1:6],size = 1,prob = relProbs[which(LETTERS==type),])
  #Check availability of dest
  if(bed[which(LETTERS==Dest)] > 0) {
    simAddEvent(time + LoS, "Dep", Dest)
    bed[which(LETTERS==Dest)] <<- bed[which(LETTERS==Dest)] - 1
  } else {
    Blocked <<- Blocked + 1
  }
}

simMain(0)
