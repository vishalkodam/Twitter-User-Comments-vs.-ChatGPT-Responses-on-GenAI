import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud


class ExploratoryDataAnalysis:
    """

    """

    @staticmethod
    def data_collection_analysis(data):
        """

        :param data:
        :return: None
        """

        grouped = data.groupby(by="Formatted Date")

        dates, counts = [], []
        for _, group in grouped:
            dates.append(_)
            counts.append(group.shape[0])

        plt.figure(figsize=(30, 10))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('"%B %d, %Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.title("Data Collected On Each Day")
        plt.xlabel("Date")
        plt.ylabel("Counts")
        plt.plot(dates, counts)
        plt.gcf().autofmt_xdate()
        plt.show()

    @staticmethod
    def perform_twitter_topic_analysis(data):
        """

        :param data:
        :return: None

        """
        topics = []

        for each_comments in data['Human Comments']:
            if type(each_comments) is str:
                for each_comment in each_comments.split("-$-"):
                    for each_word in each_comment.split():
                        if each_word.strip().startswith("@") or each_word.strip().startswith("#"):
                            topics.append(each_word.lower())

        word_cloud = WordCloud(background_color="white", width=800, height=400, collocations=False).generate(
            " ".join(topics))
        plt.figure(figsize=(15, 8))
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plot_title = "Human Comments Topic Analysis"
        plt.title(plot_title)
        # plt.show()
        plt.savefig("plots\\human_comments_topic_analysis.png", dpi=500)

        chatgpt_topics = []

        for each_comments in data['ChatGPT Response']:
            if type(each_comments) is str:
                for each_word in each_comments.split():
                    # if each_word.strip().startswith("@") or each_word.strip().startswith("#"):
                    chatgpt_topics.append(each_word.lower())

        word_cloud = WordCloud(background_color="white", width=800, height=400, collocations=False).generate(
            " ".join(chatgpt_topics))
        plt.figure(figsize=(15, 8))
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plot_title = "ChatGPT Responses Topic Analysis"
        plt.title(plot_title)
        # plt.show()
        plt.savefig("plots\\chatgpt_responses_topic_analysis.png", dpi=500)

    @staticmethod
    def data_distribution(data):
        """

        :param data:
        :return: None
        """

        chat_gpt_response_data = data['ChatGPT Response'].str.split().map(lambda x: len(x))

        human_tweets_data = []
        for each_comments in data['Human Comments']:
            if type(each_comments) is str:
                for each_comment in each_comments.split("-$-"):
                    human_tweets_data.append(len(each_comment.split()))

        fig, axs = plt.subplots(1, 2, figsize=(20, 5))

        axs[0].hist(chat_gpt_response_data, bins=20, color='b', alpha=0.5)
        axs[1].hist(human_tweets_data, bins=20, color='r', alpha=0.5)

        axs[0].set_title('ChatGPT Response Distribution')
        axs[1].set_title('Human Comments Distribution')
        axs[0].set_xlabel('ChatGPT Responses')
        axs[1].set_xlabel('Human Comments')
        axs[0].set_ylabel('Frequency')

        # plt.show()
        plt.savefig("plots\\data_distribution.png", dpi=250)

    @staticmethod
    def words_frequency(data):
        """

        :param data:
        :return: None
        """

        for col in ['ChatGPT Response', 'Human Comments']:

            temp = []
            for each in data[col]:
                if type(each) is str:
                    temp.extend(each.lower().split())

            word_cloud = WordCloud(background_color="white", width=800, height=400, collocations=False).generate(
                " ".join(temp))
            plt.figure(figsize=(15, 8))
            plt.imshow(word_cloud, interpolation='bilinear')
            plt.axis("off")
            plot_title = col + " Word Frequency"
            plt.title(plot_title)
            # plt.show()
            plt.savefig("plots\\word_frequency_" + col + ".png", dpi=250)

    @staticmethod
    def raw_vs_cleaned_data(data, human_comments_cleaned, chat_gpt_responses_cleaned):
        """

        :param data:
        :param human_tweet_cleaned:
        :param chat_gpt_responses_cleaned:
        :return: None
        """

        chat_gpt_response_data = data['ChatGPT Response'].str.split().map(lambda x: len(x))
        # human_tweets_data = data['Human Comments'].str.split().map(lambda x: len(x))

        human_tweets_data = []
        for each_comments in data['Human Comments']:
            if type(each_comments) is str:
                for each_comment in each_comments.split("-$-"):
                    human_tweets_data.append(len(each_comment.split()))

        chat_gpt_responses_cleaned_data = chat_gpt_responses_cleaned.str.split().map(lambda x: len(x))
        # human_tweet_cleaned_data = human_comments_cleaned.str.split().map(lambda x: len(x))

        human_tweet_cleaned_data = []
        for each_comments in human_comments_cleaned:
            if type(each_comments) is str and each_comments.strip():
                for each_comment in each_comments.split("-$-"):
                    human_tweet_cleaned_data.append(len(each_comment.split()))

        # print(human_tweets_data.values)
        # print(human_tweet_cleaned_data.values)
        # print(chat_gpt_responses_cleaned_data)

        fig, axs = plt.subplots(1, 2, figsize=(20, 5))

        axs[0].hist(chat_gpt_response_data, bins=20, color='r', alpha=0.5)
        axs[0].hist(chat_gpt_responses_cleaned_data, bins=20, color='b', alpha=0.5)
        axs[1].hist(human_tweets_data, bins=20, color='r', alpha=0.5)
        axs[1].hist(human_tweet_cleaned_data, bins=20, color='b', alpha=0.5)

        axs[0].set_title('ChatGPT Response Raw vs Cleaned')
        axs[1].set_title('Human Comments Raw vs Cleaned')
        axs[0].set_xlabel('ChatGPT Responses')
        axs[1].set_xlabel('Human Comments')
        axs[0].set_ylabel('Frequency')

        plt.savefig("plots\\raw_vs_cleaned.png", dpi=250)

    def main_eda(self, data, human_comments_cleaned, chat_gpt_responses_cleaned):
        """

        :param data:
        :param human_tweet_cleaned:
        :param chat_gpt_responses_cleaned:
        :return: None
        "
        """
        self.data_collection_analysis(data)
        self.perform_twitter_topic_analysis(data)
        self.data_distribution(data)
        self.words_frequency(data)
        self.raw_vs_cleaned_data(data, human_comments_cleaned, chat_gpt_responses_cleaned)
