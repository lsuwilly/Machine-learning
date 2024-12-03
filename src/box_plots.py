import pandas as pd
import matplotlib.pyplot as plt
import os

def make_box_plots():
    # Load the dataset
    data = pd.read_csv('./data/Crop_recommendation.csv')
    
    # print("Dataset Preview:")
    # print(data.head())

    # columns = N,P,K,temperature,humidity,ph,rainfall,label
    # Create the plots directory if it doesn't exist
    if not os.path.exists('./plots'):
        os.makedirs('./plots')


    label_means = data.groupby('label').mean()

    plots = []

    for column in data.columns[:-1]:
        fig, ax = plt.subplots()

        fig.set_size_inches(12, 8)
        sorted_labels = label_means[column].sort_values().index
        data['label'] = pd.Categorical(
            data['label'], categories=sorted_labels, ordered=True)
        
        # Group by 'label' and get the column values for plotting
        grouped = data.groupby('label', observed = True)[column]

        # Create a boxplot
        ax.boxplot([grouped.get_group(label).values for label in sorted_labels], 
                tick_labels=sorted_labels)
        
        ax.set_title(f'{column} vs crop')
        ax.set_xlabel('crop')
        ax.set_ylabel(column)

        plt.xticks(rotation=45, ha='right')

        # print("\n\n")
        # print(grouped.head().max())
        # print("\n\n")

        # ax.set_xticks([data['label'].unique])
        fig.savefig(f'./plots/{column}_vs_crop.png', bbox_inches='tight', dpi=300)

        plots.append(fig)

    return plots

if __name__ == '__main__':

    plots = make_box_plots()

    for plot in plots:
        plot.show()

    plt.show()

    
