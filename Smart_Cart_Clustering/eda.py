import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())

    print("\nDescribe:")
    print(df.describe())

    # Correlation heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
