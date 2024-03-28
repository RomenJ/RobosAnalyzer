import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE

class RobosAnalyzer:
    """
    A class for analyzing and visualizing robbery data.

    Attributes:
        file_path (str): The path to the Excel file containing robbery data.
        df (DataFrame): The DataFrame containing the loaded data.
    """

    def __init__(self, file_path):
        """
        Initializes the RobosAnalyzer instance.

        Args:
            file_path (str): The path to the Excel file containing the data.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Loads data from the specified Excel file.

        Returns:
            bool: True if data is loaded successfully, False otherwise.
        """
        try:
            self.df = pd.read_excel(self.file_path)
            return True
        except Exception as e:
            print("Error al cargar el archivo:", str(e))
            return False

    def clean_data(self):
        """
        Cleans the loaded data by removing rows with missing values.

        Returns:
            bool: True if data is cleaned successfully, False otherwise.
        """
        try:
            self.df = self.df.dropna()
            return True
        except Exception as e:
            print("Error al limpiar los datos:", str(e))
            return False

    def plot_dendrogram(self, data, title, output_file):
        """
        Plots a dendrogram for hierarchical clustering.

        Args:
            data: The hierarchical clustering data.
            title (str): Title for the dendrogram plot.
            output_file (str): File path to save the plot.

        Returns:
            bool: True if the dendrogram is plotted successfully, False otherwise.
        """
        try:
            plt.figure(figsize=(10, 7))
            plt.title(title)
            dendrogram = sch.dendrogram(data, labels=self.df.iloc[:, 0].values, leaf_rotation=90, leaf_font_size=8)
            plt.xlabel('Comunidades Autónomas')
            plt.ylabel('Distancia')
            plt.xticks(rotation=45)
            plt.savefig(output_file)
            plt.show()
            return True
        except Exception as e:
            print("Error al generar el dendrograma:", str(e))
            return False

    def cluster_data(self, data, t):
        """
        Performs clustering on the data.

        Args:
            data: The input data for clustering.
            t (float): The threshold for forming flat clusters.

        Returns:
            array or None: An array of cluster labels if clustering is successful, None otherwise.
        """
        try:
            clusters = fcluster(data, t=t, criterion='distance')
            return clusters
        except Exception as e:
            print("Error al realizar el clustering:", str(e))
            return None

    def visualize_tsne(self, data, output_file):
        """
        Visualizes data using t-SNE (t-distributed Stochastic Neighbor Embedding).

        Args:
            data: The input data for visualization.
            output_file (str): File path to save the t-SNE visualization plot.

        Returns:
            bool: True if t-SNE visualization is successful, False otherwise.
        """
        try:
            model = TSNE(learning_rate=10, perplexity=5)
            tsne_features = model.fit_transform(data)
            xs = tsne_features[:, 0]
            ys = tsne_features[:, 1]
            plt.figure(figsize=(10, 6))
            plt.scatter(xs, ys, alpha=0.5)
            for x, y, company in zip(xs, ys, self.df.iloc[:, 0].values):
                plt.annotate(company, (x, y), fontsize=8, alpha=0.75)
            plt.title(f'Visualización t-SNE de datos ({output_file})')
            plt.xlabel('Dimensión 1')
            plt.ylabel('Dimensión 2')
            plt.grid(True)
            plt.savefig(output_file)
            plt.show()
            return True
        except Exception as e:
            print("Error al visualizar t-SNE:", str(e))
            return False

if __name__ == "__main__":
    analyzer = RobosAnalyzer('RobosYRobosV20222018SpainXLSX6.xlsx')

    if analyzer.load_data():
        print("Info 1")
        print(analyzer.df.info())
        print("Head")
        print(analyzer.df.head(5))
        print("Shape:")
        print(analyzer.df.shape)
        print("Nulls:")
        print(analyzer.df.isna().sum())

        if analyzer.clean_data():
            print("Info df2")
            print(analyzer.df.info())
            print("Head")
            print(analyzer.df.head(5))
            print("Shape:")
            print(analyzer.df.shape)
            print("Nulls:")
            print(analyzer.df.isna().sum())

            # Year 2022
            robos2022 = analyzer.df.iloc[:, [1, 2]].values
            print("Robos y Robos con violencia en 2022")
            print(robos2022)
            Cluster_jerarquico = sch.linkage(robos2022, 'ward')
            analyzer.plot_dendrogram(Cluster_jerarquico, "Dendrograma de Clustering Jerárquico de Robos y Robos con violencia 2022", "dendrograma_2022.png")
            clusters = analyzer.cluster_data(Cluster_jerarquico, t=2)
            if clusters is not None:
                print("clusters")
                print(clusters)
            analyzer.visualize_tsne(robos2022, "t-SNE_Robos_2022.png")

            # Year 2021
            robos2021 = analyzer.df.iloc[:, [3, 4]].values
            print("Robos y Robos con violencia en 2021")
            print(robos2021)
            Cluster_jerarquico2 = sch.linkage(robos2021, 'ward')
            analyzer.plot_dendrogram(Cluster_jerarquico2, "Dendrograma de Clustering Jerárquico de Robos y Robos con violencia 2021", "dendrograma_2021.png")
            clusters2 = analyzer.cluster_data(Cluster_jerarquico2, t=5)
            if clusters2 is not None:
                print("clusters")
                print(clusters2)
            analyzer.visualize_tsne(robos2021, "t-SNE_Robos_2021.png")

            # Year 2020
            robos2020 = analyzer.df.iloc[:, [5, 6]].values
            print("Robos y Robos con violencia en 2020")
            print(robos2020)
            Cluster_jerarquico3 = sch.linkage(robos2020, 'ward')
            analyzer.plot_dendrogram(Cluster_jerarquico3, "Dendrograma de Clustering Jerárquico de Robos y Robos con violencia 2020", "dendrograma_2020.png")
            clusters3 = analyzer.cluster_data(Cluster_jerarquico3, t=8)
            if clusters3 is not None:
                print("clusters")
                print(clusters3)
            analyzer.visualize_tsne(robos2020, "t-SNE_Robos_2020.png")
        else:
            print("Error al limpiar los datos.")
    else:
        print("Error al cargar los datos.")
