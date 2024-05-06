import matplotlib.pyplot as plt

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
mAP_values = [0.7142, 0.6670, 0.6034, 0.5328, 0.4297, 0.3373, 0.2085]

plt.plot(thresholds, mAP_values)
plt.xlabel('Threshold')
plt.ylabel('mAP')
plt.title('mAP at different thresholds')
plt.show()
