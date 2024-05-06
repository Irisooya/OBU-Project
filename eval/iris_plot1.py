import matplotlib.pyplot as plt

# Data for the line graph
methods = ['UM_1000', 'LACP2', 'LACP', 'WUM[12]-', 'CoLA[24]-', 'This project']
iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
mAP_values = {
    'UM_1000': [63.3, 57.1, 49.2, 10.0, 30.7, 20.4, 10.7],
    'LACP2': [74.4, 69.6, 63.4, 55.3, 44.4, 33.2, 19.5],
    'LACP': [52.2, 46.4, 40.1, 34.3, 27.6, 21.6, 13.3],
    'WUM[12]-': [67.5, 61.2, 52.3, 43.4, 33.7, 22.9, 12.1],
    'CoLA[24]-': [66.2, 59.5, 51.5, 41.9, 32.2, 22.0, 13.1],
    'This project': [67.5, 61.2, 52.3, 43.4, 33.7, 22.9, 12.1]
}

# Create the line graph
plt.figure(figsize=(10, 6))

for method in methods:
    plt.plot(iou_values, mAP_values[method], label=method)

plt.xlabel('IoU Threshold')
plt.ylabel('mAP@IoU (%)')
plt.title('mAP@IoU Performance Comparison')
plt.legend()
plt.grid(True)
plt.xticks(iou_values)
plt.yticks(range(0, 101, 10))

plt.show()
