
import numpy as np
import pandas as pd

T = 50
t = np.arange(T)
# Dummy data
dC = -0.5 * (0.8 ** t)
dB = 1.0 * (0.8 ** t) # Dummy bond response

df = pd.DataFrame({'t': t, 'dC': dC, 'dB': dB})

# Write standard outputs
df.to_csv("gpu_jacobian_R.csv", index=False)
df.to_csv("gpu_jacobian_Z.csv", index=False) # Reuse for Z just for existence
print("Created dummy gpu_jacobian_R.csv and gpu_jacobian_Z.csv")
