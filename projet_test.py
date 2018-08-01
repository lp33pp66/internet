import numpy as np
import matplotlib.pyplot as plt

t1 = np.arange(-np.pi, np.pi, 0.1)
t2 = np.arange(-np.pi, np.pi, 0.02)

C = np.cos(t1)
C1= np.sin(t2)
C2 = np.tan(t2)

plt.figure(1)
plt.subplot(611)
plt.plot(t1, C, 'b.')

plt.subplot(612)
plt.plot(t2, C1, 'r--', label = "xxxx")
plt.legend()

plt.subplot(613)
plt.plot(t2, C2, 'r.')
plt.show()


