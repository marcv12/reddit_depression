<documents>
<document index="1">
   
# Impact of COVID-19 on Linguistic Expression of Depression in Online Communities

## Research Question
This project aims to investigate how the COVID-19 pandemic has impacted the linguistic expression of depression in online communities, specifically on Reddit. We focus on identifying whether deep learning models can effectively discern and differentiate between linguistic expression patterns of depression pre- and post-pandemic.


## Goal
Our goal is to detect shifts in the linguistic expression of depression in online communities (Reddit) when comparing the periods before and after the onset of the pandemic.


## Novelty and Impact
This research is novel as it delves into the specific linguistic changes in expressions of depression attributable to the COVID-19 pandemic. While there is existing research on mental health analysis using online data, the unique impact of the pandemic on mental health expression remains underexplored. This project seeks to understand how global crises influence mental health discussions, using Natural Language Processing (NLP) to identify shifts in language patterns. It aims to enhance the accuracy of mental health monitoring tools by adapting them to evolving language use. Additionally, the dataset created will be a new resource for future research.


## Methodology

### Data collection
- **Scraping Reddit Data:** Data will be collected from the r/depression subreddit, categorized into pre- and post-COVID eras based on the posting dates.
- **Pre-Processing:** Includes lemmatization, converting to lowercase, and removing stop words.
- **Embeddings:** Use same AutoTokenizer used in pretrained Roberta model.

### Analysis
- **Exploratory Data Analysis (EDA):** Perform topic classification using LDA and generate word clouds for both eras.
- **Emotion Detection Models:** Utilize pre-trained models such as Roberta and EmoBerta, fine-tuned on datasets accounting for a broader range of emotions.
- **Distribution Analysis:** Apply the fine-tuned model on the scraped Reddit dataset to detect emotions, analyzing the distribution changes pre- and post-COVID but also between control and experimental groups.

### Comparisons
Comparisons
- **Experimental vs. Control Posts:** Analyze the distribution of posts expressing depression-related sentiments before and after COVID-19.
- **Statistical Analysis:** In addition to fitting binary classifiers, perform statistical tests (chi-square, Mann-Withney U) to account for significant shifts pre vs post covid, but also in control vs experimental group to isolate depression as the sole factor in emotion change.


## Reproducibility
To ensure reproducibility of the experiments, it is important to note that the original dataset consists of three JSON files: "train.json", "validation.json", and "test.json", which correspond to the DepressionEmo datasets. The following steps outline the process of running the experiments:

1. Dataset Preprocessing:
   - Run the "dataset_creation_tuning.py" file to preprocess the original JSON files. 
   - This script will generate the following CSV files in the "data" folder:
     - "fine_tuning_train.csv"
     - "fine_tuning_val.csv"  
     - "fine_tuning_test.csv"

2. Reddit Data Scraping:
   - Scrape the Reddit data for the control group using the "reddit_scraper_control.py" script.
   - Scrape the Reddit data for the experimental group using the "reddit_scraper.py" script.
   - The scraped data will be saved as "reddit_control_posts.csv" and "reddit_depression_posts.csv" in the "data" folder.

3. Fine-tuning the RoBERTa Model:
   - Run the "roberta_model.py" file to fine-tune the RoBERTa model on the DepressionEmo datasets.
   - The script will generate predictions on the scraped Reddit data.
   - The predicted datasets will be saved as "predicted_dataset_control.csv" and "predicted_dataset.csv" in the "data" folder.

4. Statistical Tests:
   - Run the "statistical_test_new.py" file to perform the statistical tests.
   - Use a command-line argument to specify the path of either the control group dataset ("predicted_dataset_control.csv") or the experimental group dataset ("predicted_dataset.csv").
   - The script will output the results of the statistical tests for the specified group.

By following these steps and using the provided scripts in the specified order, the experiments can be reproduced using the original DepressionEmo datasets and the scraped Reddit data. The intermediate files generated during the process ensure that each step can be verified and the results can be replicated.


## Coding files description

### Scripts
- **dataset_creation_tuning.py:** This script is responsible for preparing the dataset for fine-tuning the Roberta model. It processes a GitHub dataset containing detailed negative emotions, making our analysis more expressive. The dataset is cleaned, preprocessed, and split into training and test sets to ensure the model is accurately trained and evaluated.

- **reddit_scraper.py:** This script scrapes data from the r/depression subreddit, targeting posts relevant to our experimental group. It's designed to collect data post-2020, filling the gap in available datasets and ensuring our analysis reflects current linguistic expressions of depression.

- **reddit_scraper_control.py:** Similar to reddit_scraper.py, but focused on scraping data from various mental health subreddits other than r/depression. This data forms the control group, allowing us to differentiate between depression-specific expressions and general mental health discussions.

- **roberta_model_probs.py:** Utilizes the Transformers library to import and fine-tune the Roberta model on our prepared dataset. The goal of this fine-tuning is to get our model to detect more specific negative emotions that are related to mental health disorders, rather than basic emotions. It evaluates the model's performance on a test set and applies it to the scraped Reddit dataset to predict the probability distribution of emotions in each post.

- **roberta_model.py:** Functions similarly to roberta_model_probs.py but outputs a binary matrix for each emotion instead of specific probabilities. This script offers an alternative approach to emotion detection, which may be used for comparative analysis.

- **statistical_tests_new.py:** This script performs statistical tests on the predicted datasets. It takes a command-line argument to specify the path of either the control group dataset ("predicted_dataset_control.csv") or the experimental group dataset ("predicted_dataset.csv"). The script outputs the results of the statistical tests for the specified group.

- **statistical_tests.py** You should run this script only if you go for the "roberta_model_probs.py" script.

### Data Folder
Contains several key files to start and the rest will be generated using the scripts during the project:

- **train.json, validation.json, test.json:** The original DepressionEmo dataset files used for fine-tuning the Roberta model.



</document>
</documents>
