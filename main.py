from cleaners import Cleaners
from unsupervisedsentimentanalysispipeline import UnsupervisedSentimentAnalysisPipeline
from exploratorydataanalysis import ExploratoryDataAnalysis
import pandas as pd
import datetime as dt
import numpy as np
from transformers import BertModel, BertTokenizer
import torch


class Main:
    def __init__(self):
        self.cleaners_obj = Cleaners()
        self.unsupervised_pl_obj = UnsupervisedSentimentAnalysisPipeline()
        self.eda_obj = ExploratoryDataAnalysis()
        self.word_embeddings = {}
        self.word_embeddings_dim = 100

    @staticmethod
    def parse_date(date_string):
        """
        This method is used to convert the date format from string to date.

        :param date_string: The dates represented in string format.
        :return: date_object: A object which contains dates that converted to date data type from string.
        """
        date_string_without_suffix = date_string.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        try:
            date_object = dt.datetime.strptime(date_string_without_suffix, "%B %d, %Y").date()
        except ValueError:
            date_object = None
        return date_object

    def read_data(self, path):
        """
        This method takes the input CSV file path and reads the data using pandas read() method.

        :param path: Input CSV file path to read the data and perform sentiment analysis.
        :return: data: The data of type pandas data frame which includes 'parsed date' along with other columns.
        """
        data = pd.read_csv(path)
        print("Data to be processed is", data.shape[0])
        parsed_data = [self.parse_date(d) for d in data['Date']]
        data['Formatted Date'] = parsed_data
        return data

    @staticmethod
    def transform_data(sentences):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        input_ids = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]

        max_len = max([len(sent) for sent in input_ids])
        input_ids = [sent + [0] * (max_len - len(sent)) for sent in input_ids]
        input_ids = torch.tensor(input_ids)

        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.eval()

        with torch.no_grad():
            outputs = bert_model(input_ids)
            embeddings = outputs.last_hidden_state

        word_embeddings_dict = {}
        for i, sent in enumerate(sentences):
            tokens = tokenizer.tokenize(sent)
            for j, token in enumerate(tokens):
                word_embeddings_dict[token] = embeddings[i, j].numpy()

        sentence_embeddings = embeddings[:, 0, :].numpy()

        return sentence_embeddings, word_embeddings_dict

    def format_human_comments(self, comments):
        comments_unwind = []
        comments_unwind_cleaned = []
        for each in comments.values:
            if type(each) is str:
                for each_c in each.split("-$-"):
                    if each_c.strip():
                        comments_unwind.append(each_c)
                        comments_unwind_cleaned.append(self.cleaners_obj.human_comment_cleaner(each_c))

        print(len(comments_unwind), len(comments_unwind_cleaned))

        return comments_unwind, comments_unwind_cleaned

    def k_means_analysis(self, human_comments_data, chat_gpt_responses_data, details):
        """
        This method trains and saves k-means model. Later, it also visualizes the clusters of k-means using t-SNE and
        matplotlib.

        :param human_tweet_embeddings: Sentence embeddings extracted from humans tweets
        :param chat_gpt_responses_embeddings: Sentence embeddings extracted from ChatGPT responses.
        :param chat_gpt_responses_word_embeddings: Word embeddings extracted from ChatGPT words.
        :param data: The entire dataset that was collected and prepared.
        :param chat_gpt_words: The list of ChatGPT words.
        :param details: It's a dictionary which is used to performs only required operations. For example, if you just
                        wants to load the saved model, you can keep train parameter as False in this parameter.
        :return: human_sentences_analysis: A pandas data frame or a table that contains the cluster_value, closeness_score,
                                           sentiment_score and the group that the sentence belongs to for Human Tweets.
                 chat_gpt_sentences_analysis: A pandas data frame or a table that contains the cluster_value,
                                              closeness_score, sentiment_score, group that the sentence belongs to for
                                              ChatGPT Sentences.
                 chat_gpt_words_analysis: A pandas data frame or a table that contains the cluster_value, closeness_score,
                                          sentiment_score, group that the sentence belongs to for ChatGPT Words.
        """
        n_clusters = 2

        model_human, model_chat_gpt, model_human_words, model_chat_gpt_words = "", "", "", ""

        if details['train_model']:
            print("Training K-Means model.")
            model_human = self.unsupervised_pl_obj.train_k_means(n_clusters, human_comments_data['hce'],
                                                                 'Human Comments',
                                                                 save_model=details['save_model'])

            model_chat_gpt = self.unsupervised_pl_obj.train_k_means(n_clusters, chat_gpt_responses_data['cre'],
                                                                    'ChatGPT Responses',
                                                                    save_model=details['save_model'])

            model_human_words = self.unsupervised_pl_obj.train_k_means(n_clusters,
                                                                       list(human_comments_data['hcew'].values()),
                                                                       'Human Words',
                                                                       save_model=details['save_model'])

            model_chat_gpt_words = self.unsupervised_pl_obj.train_k_means(n_clusters,
                                                                          list(
                                                                              chat_gpt_responses_data['crew'].values()),
                                                                          'ChatGPT Words',
                                                                          save_model=details['save_model'])

        if details['load_model']:
            print("Loading K-Means model.")
            model_human = self.unsupervised_pl_obj.load_model('Human Comments')
            model_chat_gpt = self.unsupervised_pl_obj.load_model('ChatGPT Responses')
            model_human_words = self.unsupervised_pl_obj.load_model('Human Words')
            model_chat_gpt_words = self.unsupervised_pl_obj.load_model('ChatGPT Words')

        if details['visualize_clusters']:
            print("Visualizing K-Means Clusters.")
            self.unsupervised_pl_obj.visualize_clusters(n_clusters, human_comments_data['hce'], model_human,
                                                        'Human Comments')

            self.unsupervised_pl_obj.visualize_clusters(n_clusters, chat_gpt_responses_data['cre'], model_chat_gpt,
                                                        'ChatGPT Responses')

            self.unsupervised_pl_obj.visualize_clusters(n_clusters,
                                                        np.array(list(human_comments_data['hcew'].values())),
                                                        model_human_words,
                                                        'Human Words')

            self.unsupervised_pl_obj.visualize_clusters(n_clusters,
                                                        np.array(list(chat_gpt_responses_data['crew'].values())),
                                                        model_chat_gpt_words,
                                                        'ChatGPT Words')

        chat_gpt_words_analysis = self.unsupervised_pl_obj.get_cluster_scores_and_coeff(chat_gpt_responses_data['crew'],
                                                                                        model_chat_gpt_words,
                                                                                        "words")

        human_words_analysis = self.unsupervised_pl_obj.get_cluster_scores_and_coeff(human_comments_data['hcew'],
                                                                                     model_human_words,
                                                                                     "words")

        chat_gpt_sentences_analysis = self.unsupervised_pl_obj.get_cluster_scores_and_coeff(chat_gpt_responses_data,
                                                                                            model_chat_gpt,
                                                                                            "ChatGPT Sentences")

        human_sentences_analysis = self.unsupervised_pl_obj.get_cluster_scores_and_coeff(human_comments_data,
                                                                                         model_human,
                                                                                         "Human Sentences")

        print("Processing results of both human and ChatGPT data.")
        human_sentences_analysis.to_csv("results/k_means_analysis_on_human_tweets.csv")
        print(human_sentences_analysis.shape)
        chat_gpt_sentences_analysis.to_csv("results/k_means_analysis_on_chat_gpt_responses.csv")
        print(chat_gpt_sentences_analysis.shape)
        chat_gpt_words_analysis.to_csv("results/k_means_analysis_on_chat_gpt_words.csv")
        print(chat_gpt_words_analysis.shape)
        human_words_analysis.to_csv("results/k_means_analysis_on_human_words.csv")
        print(human_words_analysis.shape)

        human_sentences_analysis, chat_gpt_sentences_analysis = [], []

        return human_sentences_analysis, chat_gpt_sentences_analysis, chat_gpt_words_analysis, human_words_analysis

    @staticmethod
    def analyze_results(human_sentences_analysis, chat_gpt_sentences_analysis, chat_gpt_words_analysis, human_df,
                        chat_gpt_df, chat_gpt_words_df):
        """
        It takes the extracted sentiments of words and sentences from both k-means and lexicon based analysis and
        analyzes the results like top 20 pos and neg words, neg and pos ChatGPT responses etc.

        :param human_sentences_analysis: sentiments extracted from human tweets using K-means.
        :param chat_gpt_sentences_analysis: sentiments extracted from ChatGPT responses using K-means.
        :param chat_gpt_words_analysis: sentiments extracted from ChatGPT words using K-means.
        :param human_df: sentiments extracted from human tweets using lexicon analysis.
        :param chat_gpt_df: sentiments extracted from ChatGPT responses using lexicon analysis.
        :param chat_gpt_words_df: sentiments extracted from ChatGPT words using lexicon analysis.

        :return: None
        """
        print("###################### Analyzing the results of K-means ##########################\n")

        print("### Top 20 ChatGPT Positive Words ###")
        print(chat_gpt_words_analysis.sort_values(by='sentiment_coeff', ascending=False).head(20))
        print("\n")
        print("### Top 20 ChatGPT Negative Words ###")
        print(chat_gpt_words_analysis.sort_values(by='sentiment_coeff', ascending=True).head(20))
        print("\n")
        print("### Top Positive ChatGPT Responses ###")
        for each in chat_gpt_sentences_analysis.sort_values(by='sentiment_coeff', ascending=False)[:2].iterrows():
            print(each[1]['sentiment_coeff'], "--", each[1]['a_sentences'])
        print("\n")
        print("### Top Negative ChatGPT Responses ###")
        for each in chat_gpt_sentences_analysis.sort_values(by='sentiment_coeff', ascending=True)[:2].iterrows():
            print("\n")
            print(each[1]['sentiment_coeff'], "--", each[1]['a_sentences'])

    def execute_steps(self, input_path, perform_eda):
        """
        It is the main method which calls all other methods and performs sentiment analysis.
        :param input_path: Input CSV file path to read the data and perform sentiment analysis.
        :param perform_eda: It is a boolean data type. If True, performs exploratory data analysis on the data. If
                            False, does not perform exploratory data analysis on the data.
        :return: None
        """

        print("\n\n######################## STEP ONE ######################################")
        print("Reading the data to perform sentiment analysis.")
        data = self.read_data(input_path)

        print("\n\n######################## STEP TWO ######################################")
        print("Cleaning human comments.")
        human_comments, human_comments_cleaned = self.format_human_comments(data['Human Comments'])
        print("Cleaning ChatGPT Responses.")
        data['chat_gpt_responses_cleaned'] = data['ChatGPT Response'].apply(self.cleaners_obj.chat_gpt_response_cleaner)

        if perform_eda:
            print("Performing exploratory data analysis.")
            self.eda_obj.main_eda(data, human_comments_cleaned, data['chat_gpt_responses_cleaned'])

        print("\n\n######################## STEP THREE ######################################")
        print("Transforming human comments data to numerical format.")
        human_comments_embeddings, human_comments_word_embeddings = self.transform_data(human_comments_cleaned)
        print("Transforming ChatGPT responses data to numerical format.")
        chat_gpt_responses_embeddings, chat_gpt_responses_word_embeddings = self.transform_data(
            data['chat_gpt_responses_cleaned'])

        human_data = {"hc": human_comments, "hcc": human_comments_cleaned, "hce": human_comments_embeddings,
                      "hcew": human_comments_word_embeddings}

        chatgpt_data = {"cr": data['ChatGPT Response'], "crc": data['chat_gpt_responses_cleaned'],
                        "cre": chat_gpt_responses_embeddings,
                        "crew": chat_gpt_responses_word_embeddings}

        # human_df = pd.DataFrame(human_data, columns=['Actual', "Cleaned", "Embedding"])
        # chat_gpt_df = pd.DataFrame(chatgpt_data, columns=['Actual', "Cleaned", "Embedding"])

        # print(human_df.head())
        # print(chat_gpt_df.head())
        #
        # print(human_df.dtypes)
        # print(human_df['Embedding'].dtype)
        # print(chat_gpt_df.dtypes)

        # human_df.to_csv("data\\human_features.csv")
        # chat_gpt_df.to_csv("data\\chatgpt_features.csv")

        print("\n\n######################## STEP FOUR ######################################")

        # human_df = pd.read_csv("data\\human_features.csv")
        # chat_gpt_df = pd.read_csv("data\\chatgpt_features.csv")

        parameter_details = {"visualize_clusters": True, 'train_model': True, 'save_model': True, 'load_model': True}

        print("### Analyzing data using K-Means. ###")
        human_sentences_analysis, chat_gpt_sentences_analysis, chat_gpt_words_analysis, human_words_analysis = self.k_means_analysis(
            human_data,
            chatgpt_data,
            parameter_details)

        print("K-Mean analysis is completed successfully.")

        print("\n")

        print("All Steps Executed Successfully.")


if __name__ == '__main__':
    main_obj = Main()

    path = "data/data.csv"

    perform_exploratory_data_analysis = False

    main_obj.execute_steps(path, perform_exploratory_data_analysis)
