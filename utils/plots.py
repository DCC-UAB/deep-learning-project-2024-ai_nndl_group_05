import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_bars(df,title,file_path):
    df_melted = df.reset_index().melt(id_vars='index', var_name='Execution', value_name='Value')
    df_melted.rename(columns={'index': 'Step'}, inplace=True)

    sns.set(style='whitegrid')
    plt.figure(figsize=(15, 9))

    # Create the bar plot
    sns.barplot(data=df_melted, x='Step', y='Value', hue='Execution', palette='Paired')

    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(title)

    # Save png
    plt.savefig(file_path, format='png')

    # Show the plot
    plt.show()



def plot_hyperparam(df, title, file_path):

    df_melted = df.reset_index().melt(id_vars='index', var_name='Execution', value_name='Value')
    df_melted.rename(columns={'index': 'Step'}, inplace=True)

    sns.set(style='whitegrid')
    plt.figure(figsize=(15, 9))

    # Create the line plot
    sns.lineplot(data=df_melted, x='Step', y='Value', hue='Execution', palette='Paired')

    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(title)

    # Save png
    plt.savefig(file_path, format='png')

    # Show the plot
    plt.show()


def drop_columns(df):
    cols_to_drop = [col for col in df.columns if col.endswith('MIN') or col.endswith('MAX')]
    cols_to_drop.append('Step')
    df = df.drop(columns=cols_to_drop)
    return df


def read_files(language):
    print(language)


    for data in ["train","val","test"]:

        for graph in ["loss","acc","wer","per"]:

            try:
                print(f'../models/{language}/wandb_graphs/{data}_{graph}.csv')
                df = pd.read_csv(f'../models/{language}/wandb_graphs/{data}_{graph}.csv')
                df = drop_columns(df)
                if data == "test":
                    plot_bars(df,title=f"{data} {graph}", file_path=f"./images/{language}/{data}_{graph}")
                else:
                    plot_hyperparam(df, title=f"{data} {graph}", file_path=f"./images/{language}/{data}_{graph}")

            except:
                print(f"CSV of {data} {graph} not found.") 


if __name__ == "__main__":
    # Print plots of hyperparameter tuning
    
    read_files(language="eng-spa")
    read_files(language="spa-eng")

    """
    CAMBIAR PLOT TEST A BARS?
    CHANGE PLOT SO THAT IT SAVES THE PNG AUTOMATICALLY
    HACER PLOTS QUE SE VEAN TRADUCCIONES? O MEJOR TABLA Y YA?
    """