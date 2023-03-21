import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

class graph:

    def __init__(self, mode, title1="Cross-Entropy", title2="Accuracy"):
        self.mode = mode
        if self.mode != "predict":
            self.fig1 = plt.figure(title1)
            self.fig2 = plt.figure(title2)
            self.ax1 = self.fig1.add_subplot()
            self.ax2 = self.fig2.add_subplot()
            self.ax1.set_xlabel("epoch")
            self.ax1.set_ylabel("Cross-Entropy")
            self.ax2.set_xlabel("epoch")
            self.ax2.set_ylabel("Accuracy")

    def add_plot(self, x1, label_x1, x2, label_x2):
        self.ax1.plot(np.arange(len(x1)), x1, label=label_x1)
        self.ax2.plot(np.arange(len(x2)), x2, label=label_x2)

    def show(self):
        if self.mode != "predict":
            self.ax1.legend()
            self.ax2.legend()
        plt.show()

def draw_matrix_confusion(confusion_matrix, title='Confusion Matrix'):
    confusion_df = pd.DataFrame(confusion_matrix, 
                             index=['True Negative', 'True Positive'], 
                             columns=['Predicted Negative', 'Predicted Positive'])

    
    # Création d'une heatmap de la matrice de confusion
    plt.figure(title,figsize=(8, 6))
    sns.set(font_scale=1.4)

    # group_names = ['True Neg','False Pos','False Neg','True Pos']
    # group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    # group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]
    # labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]
    # labels = np.asarray(labels).reshape(2,2)
    # ax = sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, annot_kws={"fontsize":12})
    
    sns.heatmap(confusion_df, annot=True, fmt='g', cmap='Blues', cbar=False)
    
    # Personnalisation de l'axe des x
    plt.xlabel('Predicted Diagnosis')
    plt.xticks(np.arange(2) + 0.5, ['Malignant', 'Begnin'], )

    # Personnalisation de l'axe des y
    plt.ylabel('True Diagnosis')
    plt.yticks(np.arange(2) + 0.5, ['Malignant', 'Begnin'])
