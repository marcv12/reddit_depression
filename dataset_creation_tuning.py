import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import json
import regex as re


# Modified function to load a JSON file with multiple JSON objects
def load_dataset(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

df_train = load_dataset('data/train.json')
df_val = load_dataset('data/val.json')
df_test = load_dataset('data/test.json')

# Set the option to display all columns
pd.set_option('display.max_columns', None)

# Function to remove URLs
def remove_urls(text):
    # Regular expression pattern for matching URLs
    url_pattern = r'https?://\S+|www\.\S+'
    # Substitute found URLs with an empty string
    no_url_text = re.sub(url_pattern, '', text)
    return no_url_text


# Function to preprocess the DataFrame
def preprocess_df(df):
    df = df[["post", "emotions"]]
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Transform the 'emotions' column to a binary matrix
    binary_matrix = mlb.fit_transform(df['emotions'])

    # Create a new DataFrame from the binary matrix
    emotions_df = pd.DataFrame(binary_matrix, columns=mlb.classes_)

    # Concatenate the original DataFrame with the new DataFrame
    df = pd.concat([df, emotions_df], axis=1)

    df = df.drop(["emotions"], axis=1)

    # Apply the function to each text entry
    df['post'] = df['post'].apply(remove_urls)

    # remove special characters and lowercase the text
    df['post'] = df['post'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x).lower()) 

    # remove extra whitespaces
    df['post'] = df['post'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    return df

# Load the datasets
train_df = load_dataset('data/train.json')
val_df = load_dataset('data/val.json')
test_df = load_dataset('data/test.json')



# Preprocess the datasets
train_df = preprocess_df(train_df)
val_df = preprocess_df(val_df)
test_df = preprocess_df(test_df)


train_df.to_csv("data/fine_tuning_train.csv", index=False)
val_df.to_csv("data/fine_tuning_val.csv", index=False)
test_df.to_csv("data/fine_tuning_test.csv", index=False)


