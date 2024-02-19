from dataset_creation_tuning import train_df, val_df
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup, RobertaConfig
import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm
import pandas as pd


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

# Load the model as is, without adjusting the classifier
model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion", 
                                                         problem_type="multi_label_classification")


# Adjust the model's classifier for 8 labels
model.classifier.out_proj = torch.nn.Linear(model.classifier.out_proj.in_features, 8)

#Prepare the dataset
# Define the EmotionDataset class
class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.post.tolist()
        self.targets = self.data.iloc[:, 1:].values
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.targets[index], dtype=torch.float)
        }




# Create dataset objects
train_dataset = EmotionDataset(train_df, tokenizer, max_length=128)
val_dataset = EmotionDataset(val_df, tokenizer, max_length=128)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)




# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = BCEWithLogitsLoss()

# Number of training epochs
epochs = 10

# Total number of training steps is [number of batches] * [number of epochs]
total_steps = len(train_loader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Training loop, for mac conda install pytorch torchvision -c pytorch-nightly
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)
model.to(device)

for epoch_i in range(epochs):
    print(f"Epoch {epoch_i + 1} of {epochs}")
    total_loss = 0  

    model.train()

    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss if outputs.loss is not None else loss_fn(outputs.logits, batch["labels"])

        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients
        loss.backward() # Calculate the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip the norm of the gradients to 1.0 to prevent the "exploding gradients" problem

        optimizer.step() # Update the model's parameters
        scheduler.step() # Update the learning rate
        model.zero_grad() # Clear the gradients

    avg_train_loss = total_loss / len(train_loader) # Calculate the average loss over all of the batches
    print(f"Average training loss: {avg_train_loss}")   


# Evaluation loop
def evaluate(model, val_loader):
    model.eval()
    val_loss = 0
    predictions, true_labels = [], []

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)


        logits = outputs.logits
        loss = loss_fn(logits, batch["labels"])
        val_loss += loss.item()

        pred_labels = logits.sigmoid().cpu().numpy()
        predictions.append(pred_labels)
        true_labels.append(batch["labels"].cpu().numpy())

    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)

    # Calculate metrics
    mae = mean_absolute_error(true_labels, predictions)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))

    return val_loss / len(val_loader), mae, rmse


val_loss, mae, rmse = evaluate(model, val_loader)
print(f"Validation Loss: {val_loss}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")




# Now, use our fine-tuned model to predict the emotions in the Reddit data
reddit_dataset = pd.read_csv('data/reddit_depression_posts.csv')

# Tokenise the Reddit posts
class RedditDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten()
        }
    
# Create a DataLoader for the Reddit dataset
reddit_data = RedditDataset(reddit_dataset.post, tokenizer, max_length=128)
reddit_loader = DataLoader(reddit_data, batch_size=16, shuffle=False)


# Make predictions
def predict(model, data_loader):
    model.eval()
    predictions = []
    print(f'Making predictions using device: {device}')


    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            pred_labels = logits.sigmoid().cpu().numpy()
            predictions.append(pred_labels)

    return np.vstack(predictions)


# Get the predictions
reddit_predictions = predict(model, reddit_loader)


# Convert the predictions to a DataFrame
reddit_predictions_df = pd.DataFrame(reddit_predictions, columns=train_df.columns[1:])
print(reddit_predictions_df.head())

# Concatenate the original dataset with the predictions
result_df = pd.concat([reddit_dataset, reddit_predictions_df], axis=1)





# Save the result to a CSV file
result_df.to_csv('data/predicted_dataset_probs.csv', index=False)

