import matplotlib.pyplot as plt
import numpy as np

models = np.arange(1, 8)
params = [250.037, 22.980, 1.153, 0.354, 1.152, 0.353, 0.403]
blops = [35.249, 4.017, 0.225, 0.088, 0.217, 0.081, 0.087]

width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(models - width/2, params, width, label='Params (MB)')
rects2 = ax.bar(models + width/2, blops, width, label='BLOPS')

# Add values above columns
def autolabel(rects):
    """Attach a text label above each bar to display its height."""
    for rect in rects:
        height = rect.get_height()
        # Handle height = 0 to prevent plotting errors on a log scale
        if height > 0:
            ax.annotate('{:.3f}'.format(height), # Format decimal places
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8) # Reduce font size if needed
        else:
            ax.annotate('0',
                        xy=(rect.get_x() + rect.get_width() / 2, 0.01), # Place at the minimum log scale value
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

ax.set_ylabel('Value (Log Scale)')
ax.set_xlabel('Model')
ax.set_title('Params & BLOPS Comparison') # Fixed spelling
ax.set_xticks(models)
ax.legend()

plt.yscale('log')
plt.ylim(0.01, 500)

fig.tight_layout()
plt.show()
