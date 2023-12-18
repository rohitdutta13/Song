import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

songPopularity = pd.read_csv('./Song/song_data.csv')

scaler = StandardScaler()


features = ['song_duration_ms', 
            'acousticness', 'danceability',
            'energy', 'instrumentalness',
            'key', 'liveness', 'loudness',
            'audio_mode', 'speechiness',
            'tempo', 'time_signature', 'audio_valence']

target = 'song_popularity'

songPopularityFeatures = songPopularity[features]
songPopularityTarget = songPopularity[target]
scaler.fit(songPopularityFeatures)

X_train, X_test, y_train, y_test = train_test_split(songPopularityFeatures, songPopularityTarget, test_size = 0.2)

import torch

def dataframe_to_tensor(df):
    return torch.tensor(df.values, dtype = torch.float32)

# Transform Dataframes into PyTorch tensors using the function
X_train = dataframe_to_tensor(X_train)
X_test = dataframe_to_tensor(X_test)
y_train = dataframe_to_tensor(y_train)
y_test = dataframe_to_tensor(y_test)

from torch import nn

class LinearRegressionModel(nn.Module):
    '''
    Torch Module class
    Initiates weight randomly and gets trained via train method.
    '''
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer

        # Initialize Weights and Bias
        self.weights = nn.Parameter(
            torch.randn(1, 13, dtype=torch.float),
            requires_grad = True)
        
        self.bias = nn.Parameter(
            torch.randn(1, dtype=torch.float),
            requires_grad=True)
        
    ## To optimize these weights via backpropagation — for that we need to setup our linear layer, consisting of the regression formula:
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return(self.weights * x + self.bias).sum(axis=1)
    
    # The trainModel method will help us perform backpropagation and weight adjustment:

    # def trainModel(
    #         self,
    #         epochs: int,
    #         X_train: torch.Tensor,
    #         X_test: torch.Tensor,
    #         y_train: torch.Tensor,
    #         y_test: torch.Tensor,
    #         lr: float
    #         ):
    def trainModel(
            self,
            epochs: int,
            X_train: torch.Tensor,
            X_test: torch.Tensor,
            y_train: torch.Tensor,
            y_test: torch.Tensor,
            lr: float
            ):
        '''
        Trains linear model using pytorch.
        Evaluates the model against test set for every epoch.
        '''
        torch.manual_seed(42)
        # Create empty loss lists to track values
        self.train_loss_values = []
        self.test_loss_values = []

        loss_fn = nn.L1Loss()

        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=lr
                )
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=lr
                )

        for epoch in range(epochs):
            self.train()
            y_pred = self(X_train)
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Set the model in evaluation mode
            self.eval()
            with torch.inference_mode():
                self.evaluate(X_test, y_test, epoch, loss_fn, loss)

    #The final touch is revealing how our model will be evaluated using the evaluate method:
    
    def evaluate(self, X_test, y_test, epoch_nb, loss_fn, train_loss):
        '''
        Evaluates current epoch performance on the test set.
        '''
        test_pred = self(X_test)
        test_loss = loss_fn(test_pred, y_test.type(torch.float))
        if epoch_nb % 10 == 0:
            self.train_loss_values.append(train_loss.detach().numpy())
            self.test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch_nb} - MAE Train Loss: {train_loss} - MAE Test Loss: {test_loss} ")


# Fitting the Model

adam_model = LinearRegressionModel('Adam')

adam_model.trainModel(500, X_train, X_test, y_train, y_test, 0.01)

# sgd_model = LinearRegressionModel('SGD')
# sgd_model.trainModel(500, X_train, X_test, y_train, y_test, 0.01)