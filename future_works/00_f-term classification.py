
if __name__ == '__main__':
    
    import os
    import sys
    import pandas as pd
    import numpy as np     
    import warnings
    import pickle 
    
    
    warnings.filterwarnings("ignore")
    
    # 서브모듈 경로 설정
    directory = os.path.dirname(os.path.abspath(__file__))
        
    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory_ = 'D:/data/F-TERM/'
    
    directory = directory.replace("\\", "/") # window
    sys.path.append(directory+'/submodule')
    import preprocess
    
    file_list = os.listdir(directory_)
    data = pd.DataFrame()
    
    for file in file_list : 
        if '.csv' in file : 
            temp_data = pd.read_csv(directory_ + file, skiprows = 4)
            temp_data['file'] = file
            data = pd.concat([data, temp_data], axis = 0).reset_index(drop =1)
               
    data_ = preprocess.wisdomain_prep_JP(data)    
    
    #%% 1. F-term, IPC 장전
    from collections import Counter 
    data_ = data_.dropna(subset = ['F-Term', 'IPC','title', 'abstract', 'claims_rep']).reset_index(drop = 1)
    
    data_input = data_.sample(500).reset_index(drop = 1)
    data_input = data_
    #%% preprocess and generate train, text X, y 
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer
    
    # cpc_sc
    IPC_mg = pd.DataFrame([Counter(x) for x in data_input['IPC_mg']]).fillna(0)   
    IPC_sg = pd.DataFrame([Counter(x) for x in data_input['IPC_sg']]).fillna(0)   
    IPC_mc = pd.DataFrame([Counter(x) for x in data_input['IPC_mc']]).fillna(0)   
    IPC_sc = pd.DataFrame([Counter(x) for x in data_input['IPC_sc']]).fillna(0)   
    
    #%%
    
    from umap import UMAP

    umap_model = UMAP(n_neighbors=15, 
                      n_components= 200, min_dist=0.0, metric='cosine', random_state=42)
    
    cpc_vectors = pd.concat([IPC_sg, IPC_mc, IPC_sc],axis = 1)
    cpc_vectors = umap_model.fit_transform(cpc_vectors)
    
    # X = pd.concat([IPC_sg],axis = 1)
    
    c = Counter(x for xs in data_input['IPC_sg'] for x in set(xs))
    
    #%% 2. embedding text
    from sentence_transformers import SentenceTransformer
    
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
    # embeddings_title = embedding_model.encode(data_input['title'] ,device='cuda')
    embeddings_abstract = embedding_model.encode(data_input['abstract'] ,device='cuda')
    embeddings_claims_rep = embedding_model.encode(data_input['claims'] ,device='cuda')
    
    #%%
    # F_TERM_targets = ["2H036QA22","2H036QA32","2H036QA33","2H036QA34","2H036RA23","2H036RA24","2H036RA25","2H036RA26"]
    F_TERM_targets = ["2H036QA22","2H036QA32","2H036QA33","2H036QA34","2H036RA23","2H036RA24","2H036RA25"]
    
    # F_TERM_targets = ["2H036QA21", "2H036QA31","2H036RA22"]
    
    mlb = MultiLabelBinarizer()
    # sc 1:3
    res = pd.DataFrame(mlb.fit_transform(data_input['F-Term']),
                       columns=mlb.classes_,
                       index=data_input.index)
    
    res = res[F_TERM_targets]
    
    y = res
    y.sum()  # 각 1642, 738, 980, 205, 255, 144, 581, 75
    #%% 합성
    umap_model = UMAP(n_neighbors=15, 
                      n_components= 100, min_dist=0.0, metric='cosine', random_state=42)
    
    embeddings_abstract = umap_model.fit_transform(embeddings_abstract)
    embeddings_claims_rep = umap_model.fit_transform(embeddings_claims_rep)
    
    X = np.concatenate((cpc_vectors, embeddings_abstract, embeddings_claims_rep), axis= 1)
    # X = np.concatenate((X, embeddings_abstract, embeddings_claims_rep), axis= 1)
    
    #%%
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    

    
    #%% loading data
    import torch
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd
    
    class MultiLabelDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X.values, dtype=torch.float32)
            self.y = torch.tensor(y.values, dtype=torch.float32)
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
        
    import torch
    from torch.utils.data import Dataset
    import pandas as pd
    import numpy as np
    
    class MultiLabelDataset(Dataset):
        def __init__(self, X, y):
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.DataFrame):
                y = y.values
            
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    
    dataset = MultiLabelDataset(X_train, y_train)
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    #%% model classification
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MultiLabelNN(nn.Module):
        def __init__(self, input_size, num_classes):
            
            super(MultiLabelNN, self).__init__()
            
                                                
            self.layer1 = nn.Linear(input_size, 256)
            self.layer2 = nn.Linear(256, num_classes)
            
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = torch.sigmoid(self.layer2(x))
            return x
        
    class EnhancedMultiLabelNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(EnhancedMultiLabelNN, self).__init__()
            
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, num_classes)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.sigmoid(self.fc4(x))
            return x
    
    class EnhancedMultiLabelNNWithDropout(nn.Module):
        def __init__(self, input_size, num_classes):
            super(EnhancedMultiLabelNNWithDropout, self).__init__()
            
            self.fc1 = nn.Linear(input_size, 256)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 128)
            self.dropout2 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, num_classes)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = F.relu(self.fc3(x))
            x = torch.sigmoid(self.fc4(x))
            return x
        
    class EnhancedMultiLabelNNWithBN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(EnhancedMultiLabelNNWithBN, self).__init__()
            
            # self.fc1 = nn.Linear(input_size, 256)
            # self.bn1 = nn.BatchNorm1d(256)
            # self.fc2 = nn.Linear(256, 128)
            # self.bn2 = nn.BatchNorm1d(128)
            # self.fc3 = nn.Linear(128, 64)
            # self.fc4 = nn.Linear(64, num_classes)
            
            self.fc1 = nn.Linear(input_size, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, num_classes)
            
        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.fc3(x))
            x = torch.sigmoid(self.fc4(x))
            return x
        
    import torch.optim as optim

    # model = MultiLabelNN(input_size=X.shape[1], num_classes=y.shape[1])
    model = EnhancedMultiLabelNN(input_size=X.shape[1], num_classes=y.shape[1])
    model = EnhancedMultiLabelNNWithDropout(input_size=X.shape[1], num_classes=y.shape[1])
    model = EnhancedMultiLabelNNWithBN(input_size=X.shape[1], num_classes=y.shape[1])

    #%%
    model = EnhancedMultiLabelNNWithBN(input_size=X.shape[1], num_classes=y.shape[1])
    # model = EnhancedMultiLabelNNWithDropout(input_size=X.shape[1], num_classes=y.shape[1])
    
    class WeightedBCELoss(nn.Module):
        def __init__(self, positive_weight=1.0, negative_weight=1.0):
            super(WeightedBCELoss, self).__init__()
            self.positive_weight = positive_weight
            self.negative_weight = negative_weight
        
        def forward(self, input, target):
            # Calculate the loss for each element in the batch
            loss = F.binary_cross_entropy(input, target, reduction='none')
            
            # Apply weights
            weights = target * self.positive_weight + (1 - target) * self.negative_weight
            loss = loss * weights
            
            # Return the mean loss
            return loss.mean()

    
    # Example usage
    criterion = WeightedBCELoss(positive_weight=10.0, negative_weight=1.0)
    # criterion = WeightedBCELoss(positive_weight=5.0, negative_weight=1.0)
    # criterion = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    X_val = X_test  # Validation features
    y_val = y_test  # Validation targets
    
    val_dataset = MultiLabelDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np
    
    # Set the model to evaluation mode
    model.eval()
    
    # Lists to store true and predicted labels
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():  # No need to track gradients
        for inputs, labels in val_loader:
            outputs = model(inputs)
            
            # Apply threshold to output probabilities
            predicted = (outputs > 0.2).float()
            
            true_labels.append(labels)
            pred_labels.append(predicted)
    
    # Concatenate all batches
    true_labels = torch.cat(true_labels, dim=0).numpy()
    pred_labels = torch.cat(pred_labels, dim=0).numpy()
    
    # Calculate scores
    accuracy = accuracy_score(true_labels, pred_labels)
    
    f1 = f1_score(true_labels, 
                  pred_labels, 
                  average='macro')  # 'macro' average for imbalanced classes
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    from sklearn.metrics import classification_report
    
    print(classification_report(true_labels, pred_labels))
    