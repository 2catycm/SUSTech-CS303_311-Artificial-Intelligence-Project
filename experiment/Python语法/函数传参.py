import numpy as np
def account(a):
    a[0]+=1

a = np.zeros(1)
account(a)
account(a)
print(a)