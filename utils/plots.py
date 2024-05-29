import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def plot_bars(df,title,file_path):
    df_melted = df.reset_index().melt(id_vars='index', var_name='Execution', value_name='Value')
    df_melted.rename(columns={'index': 'Step'}, inplace=True)

    sns.set(style='whitegrid')
    plt.figure(figsize=(20, 9))

    # Create the bar plot
    sns.barplot(data=df_melted, x='Step', y='Value', hue='Execution', palette='Paired')

    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(title)

    # Put legend outside
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Execution')
    
    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])

    # Save png
    plt.savefig(file_path, format='png', bbox_inches='tight')

    # Show the plot
    plt.show()



def plot_hyperparam(df, title, file_path):

    df_melted = df.reset_index().melt(id_vars='index', var_name='Execution', value_name='Value')
    df_melted.rename(columns={'index': 'Step'}, inplace=True)

    sns.set(style='whitegrid')
    plt.figure(figsize=(20, 9))

    # Create the line plot
    sns.lineplot(data=df_melted, x='Step', y='Value', hue='Execution', palette='Paired')

    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(title)

    # Put legend outside
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Execution')
    
    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])

    # Save png
    plt.savefig(file_path, format='png', bbox_inches='tight')  # Ensure the legend is not cut off

    # Show the plot
    plt.show()


def drop_columns(df):
    cols_to_drop = [col for col in df.columns if col.endswith('MIN') or col.endswith('MAX')]
    cols_to_drop.append('Step')
    df = df.drop(columns=cols_to_drop)
    return df


def read_files(language):

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for data in ["train","val","test"]:

        for graph in ["loss","acc","wer","per"]:

            # Get paths
            file_path = os.path.join(base_dir, 'models', language, 'wandb_graphs', f'{data}_{graph}.csv')
            image_path = os.path.join(base_dir, 'images', language, data, f'{data}_{graph}.png')

            try:
                # Get dataframes
                df = pd.read_csv(file_path)
            except:
                print(f"CSV of {data} {graph} not found.")
            
            else:
                df = drop_columns(df)

                if data == "test":
                    plot_bars(df,title=f"{data} {graph}", file_path=image_path)
                else:
                    plot_hyperparam(df, title=f"{data} {graph}", file_path=image_path)

                 

if __name__ == "__main__":
    # Print plots of hyperparameter tuning

    read_files(language="eng-spa")
    read_files(language="spa-eng")

