# import matplotlib.pyplot as plt
# import numpy as np
#
# a = np.zeros((8,8))
# plt.imshow(a)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
mat = np.arange(0, 100).reshape(10, 10)
plt.imshow(mat, cmap=plt.cm.Blues)
plt.show()