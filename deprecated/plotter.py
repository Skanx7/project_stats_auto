import matplotlib.pyplot as plt
import pandas as pd

class Plotter:
    """Class to plot the data for the Spaceship Titanic dataset."""
    def __init__(self, df: pd.DataFrame):
        self.df : pd.DataFrame = df

    def plot(self) -> None:
        """Plots the data using subplots."""
        columns_to_plot = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'FamilySize']
        nrows = len(columns_to_plot) // 2 + len(columns_to_plot) % 2 
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(15, nrows * 5))

        axs = axs.flatten()
        
        for i, column in enumerate(columns_to_plot):
            transported = self.df[self.df['Transported'] == True][column]
            not_transported = self.df[self.df['Transported'] == False][column]
            
            axs[i].hist(transported, alpha=0.5, label='Transported', bins=20, edgecolor='black')
            axs[i].hist(not_transported, alpha=0.5, label='Not Transported', bins=20, edgecolor='black')
            
            axs[i].set_title(f'Histogram of {column} by Transported Status')
            axs[i].set_xlabel(column)
            axs[i].set_ylabel('Frequency')
            axs[i].legend()

        plt.tight_layout()

        plt.show()
        
        if len(columns_to_plot) % 2 != 0:
            axs[-1].axis('off')
