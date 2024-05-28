import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    # Train
    trainloss = pd.read_csv(f'/models/{language}/wandb_graphs/train_loss.csv')
    trainacc = pd.read_csv(f'/models/{language}/wandb_graphs/train_acc.csv')
    trainwer = pd.read_csv(f'/models/{language}/wandb_graphs/train_wer.csv')
    trainper = pd.read_csv(f'/models/{language}/wandb_graphs/train_per.csv')

    for df, title in zip([trainloss,trainacc,trainwer,trainper],
        ["Train loss", "Train accuracy", "Train WER", "Train PER"]):
            plot_hyperparam(df, title)

    # Validation
    valloss = pd.read_csv(f'/models/{language}/wandb_graphs/val_loss.csv')
    valacc = pd.read_csv(f'/models/{language}/wandb_graphs/val_acc.csv')
    valwer = pd.read_csv(f'/models/{language}/wandb_graphs/val_wer.csv')
    valper = pd.read_csv(f'/models/{language}/wandb_graphs/val_per.csv')

    for df, title in zip([valloss,valacc,valwer,valper],
        ["Validation loss", "Validation accuracy", "Validation WER", "Validation PER"]):
            plot_hyperparam(df, title)

    # Test
    testloss = pd.read_csv(f'/models/{language}/wandb_graphs/test_loss.csv')
    testacc = pd.read_csv(f'/models/{language}/wandb_graphs/test_acc.csv')
    testwer = pd.read_csv(f'/models/{language}/wandb_graphs/test_wer.csv')
    testper = pd.read_csv(f'/models/{language}/wandb_graphs/test_per.csv')

    for df, title in zip([testloss,testacc,testwer,testper],
        ["Test loss", "Test accuracy", "Test WER", "Test PER"]):
            plot_hyperparam(df, title)
    


if __name__ == "__main__":
    # Print plots of hyperparameter tuning
    # Read files
    try:
        read_files("eng-spa")
        read_files("spa-eng")
    except:
        print("Error. Some file is missing or name of file is incorrect.")
    
    """
    CAMBIAR PLOT TEST A BARS?
    CHANGE PLOT SO THAT IT SAVES THE PNG AUTOMATICALLY
    HACER PLOTS QUE SE VEAN TRADUCCIONES? O MEJOR TABLA Y YA?
    """