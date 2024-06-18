import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')


class Cleaners:

    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()

        self.neutral_words = ['ai, language', 'model', 'chatgpt']

    def general_text_cleaner(self, text):
        """
        This method is used to perform general cleaning which includes removing white spaces, non-ascii etc.

        :param text: Raw uncleaned text.
        :return: text: cleaned text.
        """

        text = text.lower()
        pattern = r'[^a-zA-z\s]'
        text = re.sub(pattern, ' ', text)
        # print("Cleaned Text::", text)
        text = re.sub('\s\s+', " ", text)
        # print("Cleaned Text::", text)
        text = " ".join([word for word in nltk.word_tokenize(text) if word not in self.stopwords])
        # print("Cleaned Text::", text)
        # text = " ".join([self.lemmatizer.lemmatize(token) for token in nltk.word_tokenize(text)])
        # print("Cleaned Text::", text)

        return text

    def chat_gpt_response_cleaner(self, text):
        # print("###################################################")
        """
        This method is used to perform specific cleaning for ChatGPT responses which includes removing most frequently
        repeated neutral words.

        :param text: Raw uncleaned text.
        :return: text: Cleaned text.
        """

        self.neutral_words = ['ai, language', 'model', 'chatgpt']

        # print("Original Text::", text)
        # print("-START-", text, "-END-")

        text = self.general_text_cleaner(text)
        text = " ".join([word for word in text.split() if word not in self.neutral_words])
        # print("Cleaned Text::", text)
        return text

    def human_comment_cleaner(self, text):
        """
        This method is used to perform specific cleaning for Human Tweets which includes removing url's, @mentions and
        emojis.

        :param text: Raw uncleaned text.
        :return: text: Cleaned text
        """

        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        mentions_pattern = re.compile(r'@[^\s]+')
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", re.UNICODE)

        text = url_pattern.sub('', text)
        text = mentions_pattern.sub('', text)
        text = emoji_pattern.sub(" ", text)
        text = text.replace("#", "")

        text = self.general_text_cleaner(text)
        return text
