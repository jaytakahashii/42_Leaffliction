import matplotlib.pyplot as plt
import seaborn as sns


class GraphPlotter:
    """
    Visualize dataset distribution using pie and bar charts.
    """

    def __init__(self, data: dict[str, int]):
        """
        Args:
            data (dict[str, int]): A dictionary with class names as keys and image counts as values.
                eg: {'cat': 50, 'dog': 30, 'bird': 20}
        """
        self.data: dict[str, int] = data
        self.labels: list[str] = list(data.keys())
        self.values: list[int] = list(data.values())

    def plot(self) -> None:
        """
        Plot pie and bar charts for the dataset distribution.
        """
        if not self.data:
            print("No data to plot.")
            return

        # Create subplots for pie and bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- 1. Pie Chart ---
        ax1.pie(
            self.values,
            labels=self.labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("pastel")
        )
        ax1.set_title("Class Distribution (Pie)", fontsize=14)

        # --- 2. Bar Chart ---
        # Adjusting bar chart for better visibility
        sns.barplot(
            x=self.labels,
            y=self.values,
            ax=ax2,
            hue=self.labels,
            palette="pastel",
            legend=False
        )

        # Display count values on top of bars
        for i, v in enumerate(self.values):
            ax2.text(i, v + 2, str(v), ha='center')

        ax2.set_title("Class Distribution (Bar)", fontsize=14)
        ax2.set_ylabel("Number of Images")

        # Customizing x-axis labels for better readability
        ax2.set_xticks(range(len(self.labels)))

        # Rotate x-axis labels if they are too long
        ax2.set_xticklabels(self.labels, rotation=45, ha='right')

        # Adjust layout and display
        plt.tight_layout()
        plt.show()
