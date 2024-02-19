import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import json
import regex as re
from sklearn.model_selection import train_test_split


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

df = load_dataset('data/train.json')

# Set the option to display all columns
pd.set_option('display.max_columns', None)


df = df[["post", "emotions"]]

print(df.head())

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Transform the 'emotions' column to a binary matrix
binary_matrix = mlb.fit_transform(df['emotions'])

# Create a new DataFrame from the binary matrix
emotions_df = pd.DataFrame(binary_matrix, columns=mlb.classes_)

print(emotions_df.head())

# Concatenate the original DataFrame with the new DataFrame
df = pd.concat([df, emotions_df], axis=1)

df = df.drop(["emotions"], axis=1)

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


#Save the DataFrame to a CSV file
df.to_csv("data/fine_tuning_data.csv", index=False) 

df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42)

train_df.to_csv("data/fine_tuning_train.csv", index=False)
val_df.to_csv("data/fine_tuning_test.csv", index=False)


