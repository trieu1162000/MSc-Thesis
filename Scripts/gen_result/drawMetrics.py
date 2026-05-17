import matplotlib.pyplot as plt

models = [1, 2, 3, 4, 5, 6, 7]
mAP = [0.9543, 0.8918, 0.8738, 0.8171, 0.7865, 0.7841, 0.8022]
precision = [0.9, 0.86, 0.72, 0.71, 0.76, 0.71, 0.71]
recall = [0.92, 0.86, 0.86, 0.80, 0.75, 0.76, 0.77]
f1_score = [0.91, 0.86, 0.78, 0.75, 0.76, 0.74, 0.73]

# Find the maximum value across all metrics
max_value = max(mAP + precision + recall + f1_score)

# Set y-axis limits from 0 to slightly above the maximum value (for better visualization)
plt.ylim(0, max_value * 1.05)  # Adjust the multiplier (1.05) if needed

plt.plot(models, mAP, label='mAP')
plt.plot(models, precision, label='Precision')
plt.plot(models, recall, label='Recall')
plt.plot(models, f1_score, label='F1-Score')

plt.xlabel('Model')
plt.ylabel('Value')
plt.title('Performance Metrics')
plt.legend()
plt.show()
