import matplotlib.pyplot as plt
import seaborn as sns

# Plot target variable distribution (Loan Approved or Not)
def plot_target_distribution(df, target):
    """
    Shows count of approved vs rejected loans
    """
    df[target].value_counts().plot(kind='bar')
    plt.title("Loan Status Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Count")
    plt.show()


# Plot histogram of any numerical column
def plot_histogram(df, column):
    """
    Visualize distribution of a numerical feature
    """
    plt.hist(df[column], bins=30)
    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


# Plot correlation heatmap
def plot_correlation(df):
    """
    Shows relationship between features
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
