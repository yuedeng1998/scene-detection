import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


#n_groups = 5


means_women = (25, 32, 34, 20, 25)


#fig, ax = plt.plot()

index = np.arange(5)
#bar_width = 0.35

#opacity = 0.4
#error_config = {'ecolor': '0.3'}


# rects2 = ax.bar(index + bar_width, means_women, bar_width,
#                 alpha=opacity, color='r',
#                 yerr=std_women, error_kw=error_config,
#                 label='Women')
rects2 = plt.bar(index, means_women)

# ax.set_xlabel('Group')
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
#ax.legend()

#fig.tight_layout()
plt.show()