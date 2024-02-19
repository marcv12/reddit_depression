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
- **Statistical Analysis:** In addition to fitting binary classifiers, perform statistical tests (t-test, ANOVA, chi-square...) to account for significant shifts pre vs post covid, but also in control vs experimental group to isolate depression as the sole factor in emotion change.


## Coding files description

### Scripts
- **dataset_creation_tuning.py:** This script is responsible for preparing the dataset for fine-tuning the Roberta model. It processes a GitHub dataset containing detailed negative emotions, making our analysis more expressive. The dataset is cleaned, preprocessed, and split into training and test sets to ensure the model is accurately trained and evaluated.

- **reddit_scraper.py:** This script scrapes data from the r/depression subreddit, targeting posts relevant to our experimental group. It's designed to collect data post-2020, filling the gap in available datasets and ensuring our analysis reflects current linguistic expressions of depression.

- **reddit_scraper_control.py:** Similar to reddit_scraper.py, but focused on scraping data from various mental health subreddits other than r/depression. This data forms the control group, allowing us to differentiate between depression-specific expressions and general mental health discussions.

- **roberta_model_probs.py:** Utilizes the Transformers library to import and fine-tune the Roberta model on our prepared dataset. It evaluates the model's performance on a test set and applies it to the scraped Reddit dataset to predict the probability distribution of emotions in each post.

- **roberta_model.py:** Functions similarly to roberta_model_probs.py but outputs a binary matrix for each emotion instead of specific probabilities. This script offers an alternative approach to emotion detection, which may be used for comparative analysis.

### Data Folder
Contains several key files used and generated during the project:

- **fine_tuning_data.csv:** Compiled dataset ready for model fine-tuning, including both training and test data.

- **fine_tuning_test.csv, fine_tuning_train.csv:** Separated test and training datasets derived from fine_tuning_data.csv, specifically structured for model training and evaluation phases.

- **predicted_dataset.csv:** Contains predictions from the fine-tuned model applied to the Reddit dataset, detailing the emotional content of each post.

- **reddit_control_posts.csv:** Dataset of posts from control subreddits, used to compare against the experimental group (depression-related posts).

- **reddit_depression_posts.csv:** Dataset of posts from the r/depression subreddit, forming the core of our experimental data.

- **reddit_predictions.csv:** Results of applying emotion detection models to the scraped Reddit datasets, offering insights into the prevalence and distribution of emotions in pre- and post-pandemic expressions of depression.


### Additional Files
**Miniconda3-latest-MacOSX-arm64.sh:** A script to install Miniconda, facilitating the use of a GPU for the Roberta model training. This tool is crucial for efficiently handling the computational demands of deep learning models.
