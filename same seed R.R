install.packages("reticulate")
library(reticulate)
np <- import("numpy")

# Create a Generator with MT19937
bitgen <- np$random$MT19937(seed = 123L)
rng <- np$random$Generator(bitgen)

# Sample uniform float using NumPy's MT19937
print(rng$uniform(0, 1, 1L))  # This now matches Python!

