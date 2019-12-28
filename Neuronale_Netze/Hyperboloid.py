from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm

# here is a function to calculate the z for a (x,y) pair
def get_z(x, y, width, height):
    # opens downward
    c = float(height) # -
    # circular so same width in both directions
    x2 = (x ** 2) / float(width) 
    y2 = (y ** 2) / float(width) 
    z = (x2 + y2) / c

    return (y2 + x * y + y2 - 3.0)#(z)  

# and here a function that does the plotting based on the xs and the ys
def plot_par(axes, xs, ys, zero_x=0, zero_y=0, width=1, height=100):
    zs = np.array([])
    for x in xs:
        for y in ys:
            # need to subtract the zero to center the surface
            zs= np.append(zs, get_z(x - zero_x, y - zero_y, width, height))
    
    Z    = zs.reshape(len(xs),len(ys))
    
    X, Y = np.meshgrid(xs, ys)
    print(X)
    print(Y)
    print(Z)

    axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
#plot_par(ax,np.arange(0,100,1),np.arange(0,500,5),zero_x=50, zero_y=250)
plot_par(ax,np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), zero_x=0.5, zero_y=0.5)
plt.show()

'''
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

x, y = np.meshgrid(np.linspace(np.min(doex), np.max(doex),10), np.linspace(np.min(doey),np.max(doey), 10))
ax.plot_wireframe(x, y, paraBolEqn((x,y), *popt))
ax.scatter(doex, doey, doez, color='b')
'''


'''
# Hyperboloid

plt.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

# Prepare arrays x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-1, 1, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()
'''