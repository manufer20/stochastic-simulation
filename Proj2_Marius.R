#Create eventList, which holds all of the unhandled events
eventList <- data.frame(e_time = numeric(),
                        e_type = factor(levels = c("Dep", "Tra", "Arr")),
                        p_type = character(),
                        stringsAsFactors = F)

#Create bed, a list of how many free beds are in each ward
bed <- numeric(length = 0)

#Create Penalties, which holds the sim output of how much penalty each ward has accrued
Penalties <- numeric(length = 6)

#Create Blocked, which holds the sim output of how many patients were blocked
Blocked <- 0

#Create endTime, which sets the period of time to be simulated, in days
endTime <- 10
#Rate of arrivals
ArrRate <- numeric(length = 6)
#Rate of Length of stay for each patient
μLoS <- numeric(length = 6)
#Urgency, stores the amount of penalty a blocked patient generates
urgency <- numeric(length = 6)

#Main function. withF is either 0 or 1, depending on if sim should include ward F
simMain <- function(withF){
  #Run init function
  simInit(withF)
  #Repeat as long as there are more events, and the time haven't exceeded the end time
  while(nrow(eventList) > 0 && replace(eventList[1,1], is.na(eventList[1,1]), 0)<endTime){
    #Check the first event in list, run appropriate function
    switch (as.character(eventList[1,2]),
      "Arr" = simArr(eventList[1,1],eventList[1,3]),
      "Dep" = simDep(eventList[1,1],eventList[1,3]),
      "Tra" = simTra(eventList[1,1],eventList[1,3])
    )
    #Remove that event from list
    eventList <<- eventList[-1,]
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
simArr <- function(time, type){print("Arr")}

#Function to handle departures
simDep <- function(time, type){print("Dep")}

#Function to handle transfers
simTra <- function(time, type){print("Tra")}

#Given values
bed <- c(55,40,30,20,20,0)
ArrRate <- c(14.5,11,8,6.5,5,13)
μLoS <- c(2.9,4,4.5,1.4,3.9,2.2)
urgency <- c(7,5,2,10,5,0)


