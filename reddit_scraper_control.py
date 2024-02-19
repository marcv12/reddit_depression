import praw
import pandas as pd
from datetime import datetime
import regex as re
import random


# Initialize PRAW with Reddit application credentials (enter secret key in client_secret)
reddit = praw.Reddit(client_id='lfzq1sMXmgnKvLP_DEGR7A',
                     client_secret='Enter secret key',
                     user_agent='Depression Linguist Analyzer/1.0 by Ok-Leading-6463')




# Define the subreddits you want to scrape
subreddit_names = ["EDAnonymous", "addiction", "alcoholism", 
                   "adhd", "anxiety", "autism", "bipolarreddit", "bpd", 
                   "healthanxiety", "lonely", "ptsd", "schizophrenia", "socialanxiety", "suicidewatch"]

# Define the timestamp for the start of the pandemic
pandemic_start = 1583884800  # March 11, 2020, in Unix timestamp

# Lists to store scraped data
posts_data = []

# Define the maximum number of posts you want to scrape
max_posts = 2500

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
            if period_label == "pre-pandemic" and pre_pandemic_count < max_posts/len(subreddit_names):
                pre_pandemic_count += 1
            elif period_label == "post-pandemic" and post_pandemic_count < max_posts/len(subreddit_names):
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

# Define the maximum number of posts you want to scrape per subreddit
max_posts_per_subreddit = max_posts // len(subreddit_names)

# Loop over the subreddit names
for subreddit_name in subreddit_names:
    print(subreddit_name)
    # Reset the counters for each subreddit
    pre_pandemic_count = 0
    post_pandemic_count = 0

    # Fetch post-pandemic posts
    for category in categories:
        print(category, "post-pandemic")
        fetch_posts(category, pandemic_start, float('inf'), "post-pandemic")

    # Fetch pre-pandemic posts
    for category in categories:
        print(category, "pre-pandemic")
        fetch_posts(category, 0, pandemic_start, "pre-pandemic")




# Convert the list to a DataFrame
df = pd.DataFrame(posts_data)

# Split the DataFrame into two separate DataFrames
df_pre_pandemic = df[df['pandemic_period'] == 'pre-pandemic']
df_post_pandemic = df[df['pandemic_period'] == 'post-pandemic']

# Find the minimum count of posts between pre-pandemic and post-pandemic
min_count = min(len(df_pre_pandemic), len(df_post_pandemic))

# Sample min_count number of rows from each DataFrame
df_pre_pandemic = df_pre_pandemic.sample(min_count, random_state=2)
df_post_pandemic = df_post_pandemic.sample(min_count, random_state=2)

# Concatenate the two DataFrames and shuffle
df = pd.concat([df_pre_pandemic, df_post_pandemic])
df = df.sample(frac=1, random_state=2).reset_index(drop=True)




# data preprocessing
# Function to remove URLs
def remove_urls(text):
    # Regular expression pattern for matching URLs
    url_pattern = r'https?://\S+|www\.\S+'
    # Substitute found URLs with an empty string
    no_url_text = re.sub(url_pattern, '', text)
    return no_url_text

# Apply the function to each text entry
df['post'] = df['post'].apply(remove_urls)


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
df.to_csv("data/reddit_control_posts.csv", index=False)

