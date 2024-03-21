import praw
import pandas as pd
from datetime import datetime
import regex as re
import random
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities.engine import RecognizerResult, OperatorConfig


analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


# Initialize PRAW with Reddit application credentials (enter secret key in client_secret)
reddit = praw.Reddit(client_id='lfzq1sMXmgnKvLP_DEGR7A',
                     client_secret='Enter secret key',
                     user_agent='Depression Linguist Analyzer/1.0 by Ok-Leading-6463')




# Define the subreddit you want to scrape
subreddit_name = "depression"

# Define the timestamp for the start of the pandemic
pandemic_start = 1583884800  # March 11, 2020, in Unix timestamp

# Lists to store scraped data
posts_data = []

# Define the maximum number of posts you want to scrape
max_posts = 6000

# Initialize a set to store post IDs
post_ids = set()

# Initialize counters for pre-pandemic and post-pandemic posts
pre_pandemic_count = 0
post_pandemic_count = 0

# Define a function to fetch posts from a category
def fetch_posts(category, period_start, period_end, period_label):
    global post_pandemic_count, pre_pandemic_count
    count = 0
    for submission in getattr(reddit.subreddit(subreddit_name), category)(limit=None):
        # Skip the post if its ID is already in the set
        if submission.id in post_ids:
            continue

        # Convert submission created_utc to datetime
        submission_date = datetime.utcfromtimestamp(submission.created_utc)
        
        # Only add posts from the specified period
        if period_start <= submission.created_utc < period_end:
            if period_label == "pre-pandemic" and pre_pandemic_count < max_posts:
                pre_pandemic_count += 1
            elif period_label == "post-pandemic" and post_pandemic_count < max_posts:
                post_pandemic_count += 1
            else:
                continue

            # Append post data to the list
            posts_data.append({
                "title": submission.title,
                "post": submission.selftext,
                "created_utc": submission_date,
                "pandemic_period": period_label
            })
            # Add the post ID to the set
            post_ids.add(submission.id)
            count += 1

# Define the categories you want to fetch posts from
categories = ['new', 'hot', 'top', 'controversial', 'rising']

# Fetch post-pandemic posts
for category in categories:
    fetch_posts(category, pandemic_start, float('inf'), "post-pandemic")

# Fetch pre-pandemic posts
for category in categories:
    fetch_posts(category, 0, pandemic_start, "pre-pandemic")



# Convert the list to a DataFrame
df = pd.DataFrame(posts_data)

# Split the DataFrame into two separate DataFrames
df_pre_pandemic = df[df['pandemic_period'] == 'pre-pandemic']
df_post_pandemic = df[df['pandemic_period'] == 'post-pandemic']

# Get the minimum count of posts between the two periods
min_count = min(len(df_pre_pandemic), len(df_post_pandemic))

# Sample an equal number of rows from each DataFrame
df_pre_pandemic = df_pre_pandemic.sample(min_count, random_state=1)
df_post_pandemic = df_post_pandemic.sample(min_count, random_state=1)

# Concatenate the two DataFrames back together
df = pd.concat([df_pre_pandemic, df_post_pandemic])

# Shuffle the DataFrame to mix post-pandemic and pre-pandemic posts
df = df.sample(frac=1, random_state=1).reset_index(drop=True)




# data preprocessing
# Function to remove URLs
# Updated function to remove URLs, dates, and specific locations
def remove_urls_dates_locations(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Basic pattern to match dates (e.g., "March 5", "2020-03-11", "03/11/2020")
    text = re.sub(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2})[\/\-,\s]*(?:\d{1,2}[\/\-,\s]*)?(?:\d{2,4})?\b', '[DATE]', text)
    # This is a very basic and not comprehensive way to handle locations and dates. For more sophisticated anonymization, consider NER or other methods.
    return text


#Function to anonymize named entities like persons, locations, and organizations
# Load the language model
nlp = spacy.load("en_core_web_md")

# def anonymize_named_entities(text):
#     doc = nlp(text)
#     anonymized_text = text
#     for ent in doc.ents:
#         if ent.label in ["PERSON", "GPE", "ORG", "LOC"]:
#             anonymized_text = anonymized_text.replace(ent.text, '[' + ent.label_ + ']')
#     return anonymized_text

def anonymize_named_entities(text):
    # Analyze the text to find entities
    analysis_results = analyzer.analyze(text=text, language='en')
    
    # Specify that all detected entities should be redacted
    operators = {
    "PERSON": OperatorConfig("redact", {}),
    "LOCATION": OperatorConfig("replace", {"new_value": "Springfield"}),
    "ORGANIZATION": OperatorConfig("replace", {"new_value": "ACME Corp"})
}
    
    # Redact entities from the text based on the analysis results
    anonymized_result = anonymizer.anonymize(
        text=text, 
        analyzer_results=analysis_results,
        operators=operators
    )
    return anonymized_result.text




# Apply the function to each text entry
df['post'] = df['post'].apply(remove_urls_dates_locations)
df['post'] = df['post'].apply(anonymize_named_entities)

# remove special characters and lowercase the text
df['post'] = df['post'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x).lower()) 

# remove extra whitespaces
df['post'] = df['post'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())


# Remove rows with empty post text
df = df[df['post'].str.len() > 15]

# Find the minimum count of pre-pandemic and post-pandemic posts
min_count = df['pandemic_period'].value_counts().min()

# Create a balanced dataframe
df_balanced = df.groupby('pandemic_period').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

df = df_balanced[["post", "pandemic_period"]]

# Save the DataFrame to a CSV file
df.to_csv("data/reddit_depression_posts.csv", index=False)

