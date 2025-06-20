eventList <- data.frame(e_time = numeric(),
                        e_type = factor(levels = c("Dep", "Tra", "Arr")),
                        p_type = numeric())

bed <- numeric(length = 6)

Penalties <- numeric(length = 6)

Blocked <- 0



simInit <- function(){}