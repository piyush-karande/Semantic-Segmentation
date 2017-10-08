import numpy as np
import glob

import matplotlib as mpl
if "MACOSX" in mpl.get_backend().upper():
  mpl.use("TkAgg")

import matplotlib.pyplot as plt

f = glob.glob('./runs/1507401207.945786/*.csv')
loss = np.loadtxt(f[0])

epochs = 30
x = np.arange(0,epochs,epochs/len(loss))

plt.semilogy(x,loss)

plt.xticks(np.arange(0,epochs+1,5))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('on')

plt.show()