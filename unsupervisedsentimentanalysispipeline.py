from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
import os


class UnsupervisedSentimentAnalysisPipeline:

    @staticmethod
    def train_k_means(n_clusters, sentence_embeddings, label, save_model):
        """
        This method is used to train k-means model using 2 clusters, evaluate the model using Shilhouette Score and
        Saves the model only if the save flag is True.

        :param n_clusters: of clusters the model should train on.
        :param sentence_embeddings: A matrix of shape (514, 300) which contains 300 features for each sentence.
        :param label: It is a string which specifies if the model should train in tweets or responses.
        :param save_model: It's a boolean data which is used if the trained model should be saved or not.
        :return: k_means A trained model.
        """
        print("Training ", label.lower().replace(" ", "_"), "clustering model")
        k_means = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = k_means.fit_predict(sentence_embeddings)
        silhouette_avg = silhouette_score(sentence_embeddings, cluster_labels)
        print("Model Evaluation on " + label + " Data using Silhouette Score on", n_clusters, "clusters:",
              silhouette_avg)

        if save_model:
            if not os.path.exists("models"):
                os.mkdir("models")
            print("Saving ", label.lower().replace(" ", "_"), "clustering model")
            dump(k_means, 'models/' + label.lower().replace(" ", "_") + '_clustering.joblib')

        return k_means

    @staticmethod
    def visualize_clusters(n_clusters, sentence_embeddings, k_means, label):
        """
        This method is used to visualize the clusters using t-SNE.

        :param n_clusters: No. of clusters the model is trained on.
        :param sentence_embeddings: A matrix of shape (514, 300) which contains 300 features for each sentence.
        :param k_means: A trained model.
        :param label: It is a string which specifies if the method should visualize tweets or responses.
        :return: None
        """
        tsne = TSNE(n_components=n_clusters, random_state=42)
        embeddings_tsne = tsne.fit_transform(sentence_embeddings)
        plt.figure(figsize=(10, 5))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=k_means.labels_)
        plt.title("KMeans on " + label + " data")
        # plt.show()
        plt.savefig("plots\\k_means_" + label.replace(" ", "_") + ".png", dpi=250)

    @staticmethod
    def load_model(label):
        """
        This method is used to load the saved model.

        :param label: It is a string which specifies which model should load. Human or ChatGPT.
        :return: model: A model which is loaded.
        """
        print("Loading ", label.lower().replace(" ", "_"), "clustering model")
        model = load('models/' + label.lower().replace(" ", "_") + '_clustering.joblib')
        return model

    @staticmethod
    def get_cluster_scores_and_coeff(data, k_means, label):
        """
        This method used to calculate the sentiment scores and cluster values (0 or 1)for each data point.

        :param data: The entire data set that was collected and prepared.
        :param matrix: A matrix which contains 300 features for each data point.
        :param k_means: k-means model.
        :param label: It is a string which specifies which sentiment scores to extract.
        :return: df: A pandas data frame or table which contains sentiment_scores, cluster_values (1 or 0),
                     closeness_score etc.
        """
        df = pd.DataFrame()
        if label == 'words':
            words = list(data.keys())
            vectors = list(data.values())

            clusters = k_means.predict(vectors)
            closeness_score = [1 / (k_means.transform([vector]).min()) for vector in vectors]

            cluster_value = [1 if i == 0 else -1 for i in clusters]
            sentiment_coeff = [cs * cv for cs, cv in zip(closeness_score, cluster_value)]

            updated_data = {"words": words, "clusters": clusters, "cluster_value": cluster_value,
                            "closeness_score": closeness_score, "sentiment_coeff": sentiment_coeff}

            df = pd.DataFrame(updated_data, columns=list(updated_data.keys()))

        if "sentence" in label.lower():
            clusters, closeness_score = [], []
            sentences, sentences_cleaned = [], []
            if "hce" in data:
                clusters = k_means.predict(data['hce'])
                closeness_score = [1 / (k_means.transform([vector]).min()) for vector in data['hce']]
                sentences = data['hc']
                sentences_cleaned = data['hcc']
                del data['hce']

            elif "cre" in data:
                clusters = k_means.predict(data['cre'])
                closeness_score = [1 / (k_means.transform([vector]).min()) for vector in data['cre']]
                sentences = data['cr']
                sentences_cleaned = data['crc']
                del data['cre']

            cluster_value = [1 if i == 0 else -1 for i in clusters]
            sentiment_coeff = [cs * cv for cs, cv in zip(closeness_score, cluster_value)]

            updated_data = {"sentences": sentences, "sentences_cleaned": sentences_cleaned, "clusters": clusters,
                            "cluster_value": cluster_value,
                            "closeness_score": closeness_score, "sentiment_coeff": sentiment_coeff}

            df = pd.DataFrame(updated_data, columns=list(updated_data.keys()))

        return df
