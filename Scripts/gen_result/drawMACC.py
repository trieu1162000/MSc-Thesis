import matplotlib.pyplot as plt
import numpy as np

models = np.arange(1, 6)
macc = [119.15, 48.23, 115.053, 44.13, 47.56]
flash = [434.77, 436.26, 438.79, 221.32, 243.41]
ram = [1230.0, 1230.0, 670.44, 457.04, 459.86]

width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))

rects1 = ax.bar(x - width, macc, width, label='MACC (M)')
rects2 = ax.bar(x, flash, width, label='Flash (KB)')
rects3 = ax.bar(x + width, ram, width, label='RAM (KB)')

# Vẽ đường ngang giới hạn Flash
plt.axhline(y=512, color='r', linestyle='--', label='RAM D1 STM32H743 Limit (512KB)')

ax.set_ylabel('KBytes (KBs)') # Sửa nhãn trục y cho phù hợp (KB)
ax.set_xlabel('Model')
ax.set_title('MACC, Flash & RAM comparision')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.show()
