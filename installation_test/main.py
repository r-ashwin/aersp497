import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 2
mpl.rc('xtick', labelsize=18)
mpl.rc('ytick', labelsize=18)
mpl.rc('axes', labelsize=18)
mpl.rc('font', size=18)

# 2D example
x1 = np.linspace(-5,5,100)
x2 = np.linspace(-5,5,100)
x1v,x2v = np.meshgrid(x1, x2)
x = np.column_stack((x1v.reshape(-1,1), x2v.reshape(-1,1)))
y = (x[...,0] - 2)**2 + (x[...,1] - 1)**2

plt.figure(figsize=[12,8])
c=plt.contour(x1v, x2v, y.reshape(100,100), levels=25, colors='k')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
# plt.colorbar(c)
plt.clabel(c, c.levels, inline=True, fontsize=14)
plt.tight_layout()
plt.show()