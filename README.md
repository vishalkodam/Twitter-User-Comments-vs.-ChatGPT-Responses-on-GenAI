# Twitter-User-Comments-vs.-ChatGPT-Responses-on-GenAI

### Prerequisites

Before running the project, ensure that you have the following installed and properly configured on your system:

1. **Python**:
   - Version: Python 3.10 or higher.
   - Ensure that Python is added to your system PATH.

2. **Required Libraries**:
   - The project requires several Python libraries for data manipulation, machine learning, and visualization.
   - Install the necessary libraries using the `requirements.txt` file provided with the project:
     
     pip install -r requirements.txt

3. **NLTK Data**:
   - The project uses the Natural Language Toolkit (NLTK) for text processing.

5. **Hugging Face's Transformers**:
   - The project uses BERT embeddings from the Hugging Face Transformers library.
   - Ensure you have the transformers library installed and configured.

6. **Data Files**:
   - Ensure that the necessary data files (e.g., human comments, ChatGPT responses) are available in the appropriate directories as specified in the code.
  

### Project Overview

This project, conducted by researchers from the Department of Computer Science at SUNY Polytechnic Institute, aims to explore the emotional intelligence of ChatGPT, a large language model developed by OpenAI, by comparing its responses to human comments on Twitter. The study employs sentiment analysis techniques to assess and compare the emotional content of ChatGPT’s responses and human comments, focusing on the ability of ChatGPT to engage and communicate effectively with humans.

### Objectives
The primary objectives of this study are:

To measure the emotional content and tone of ChatGPT's responses.
To compare these responses with human Twitter comments.
To investigate how ChatGPT's emotional intelligence affects its ability to engage and communicate with humans.


### Methodology

The study uses a combination of various sentiment analysis techniques and tools to achieve its objectives. The approach includes:

**Data Collection:** Gathering real-time human comments from Twitter using hashtags like #chatgpt, #llms, and #openai, resulting in a dataset of 429 unique comments.
**Exploratory Data Analysis (EDA):** Performing initial analyses to gain insights into the data, such as topic analysis and word frequency distribution.
**Data Cleaning:** Implementing a three-step cleaning process to remove noise and irrelevant information from the data.
**Data Transformation:** Converting textual data to numerical format using pre-trained BERT embeddings with 768 dimensions.
**Unsupervised Sentiment Analysis:** Applying the K-means clustering algorithm to analyze the emotional content of the comments and responses, followed by visualization using t-SNE.

### Future Work
Future research could extend this study by:

Increasing the dataset size to include more diverse comments and responses.
Exploring additional sentiment analysis techniques and machine learning models.
Investigating the impact of different types of emotional content on user engagement and satisfaction.
