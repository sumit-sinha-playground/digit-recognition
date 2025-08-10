# accuracy_plot.py

from matplotlib import pyplot as plt
from IPython.display import display, clear_output

class AccuracyPlot:
    def __init__(self):
        self._accuracies = []
        self._fig, self._ax = plt.subplots()

    def update(self, accuracy):
        self._accuracies.append(accuracy)
        clear_output(wait=True)
        self._ax.clear()
        self._ax.plot(self._accuracies)
        self._ax.set_xlabel('Iterations')
        self._ax.set_ylabel('Accuracy')
        self._ax.set_title(f'Training Accuracy (Iteration: {len(self._accuracies)}, Accuracy: {accuracy * 100:.2f}%)')
        self._ax.grid(True)
        self._ax.set_ylim(0, 1.05)
        display(self._fig)

    def close(self):
        plt.close(self._fig)
