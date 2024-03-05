from torch.utils.data import Dataset
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

def load_data(group):
    
    if group == "experimental":
        file = "../data/predicted_dataset_probs.csv"
    else:
        file = "../data/predicted_dataset_control_probs.csv"
        
    dataset = pd.read_csv(file, sep=",")
    X = dataset[["anger","brain dysfunction (forget)","emptiness","hopelessness","loneliness","sadness","suicide intent","worthlessness"]]
    
    #Q1 = X.quantile(0.25)
    #Q3 = X.quantile(0.75)
    #IQR = Q3 - Q1
    #X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
    #X_idx = X.index
    #y = dataset['pandemic_period'][X_idx]

    encoder = LabelEncoder()
    y = encoder.fit_transform(dataset['pandemic_period'])
    
    return X.values, y

class CovidDataset(Dataset):
    
    def __init__(self, group = "experimental"):
        super().__init__()
        X, y = load_data(group)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
