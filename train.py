import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os
from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.animation as animation
from IPython.display import HTML


kospi_daily = pd.read_csv("data/KOSPI_daily.csv")
usdkrw_daily = pd.read_csv('data/USDKRW_daily.csv')
usdkrw_daily = usdkrw_daily[["Date","Close"]]
usdkrw_daily["Date"]=pd.to_datetime(usdkrw_daily["Date"])
kospi_daily["Date"] = pd.to_datetime(kospi_daily["Date"])
kospi_daily = kospi_daily.merge(usdkrw_daily, how='inner',on="Date")

numRows,numColumns = kospi_daily.shape
last_date, first_date = kospi_daily.iloc[0].Date, kospi_daily.iloc[-1].Date
na_cols = kospi_daily.columns[kospi_daily.isna().any()].tolist()

print(f"There are {numRows} rows and {numColumns} columns in the initial dataset.")
print(f"The data represents the time frame between the dates '{last_date}' and '{first_date}'.")
if not na_cols:
    print("There are no NA rows.")
else:
    print(f"Columns in the dataset which include NA rows: {na_cols}.")

column_names = ["Close_x", "Open", "High", "Low"]
for column in column_names:
    kospi_daily[column] = pd.to_numeric(kospi_daily[column])

kospi_daily.Date = pd.to_datetime(kospi_daily.Date)
kospi_daily.sort_values(by="Date", ignore_index=True,inplace=True)
kospi_daily.set_index(pd.DatetimeIndex(kospi_daily["Date"]), inplace=True)

kospi_daily.ta.log_return(cumulative=True, append=True)
kospi_daily.ta.percent_return(cumulative=True, append=True)

ind_list = kospi_daily.ta.indicators(as_list=True)

kospi_daily.ta.rsi(append=True)
kospi_daily.ta.macd(append=True)

sma_values = [5, 10, 15] 
for i in sma_values:
    kospi_daily['SMA'+str(i)] = kospi_daily['Close_x'].rolling(window=i).mean()

kospi_daily.dropna(inplace=True)

vols = kospi_daily['Volume'].to_list()
indexToRemove = kospi_daily.iloc[list(map(itemgetter(0),filter(lambda vol: "-" in vol,enumerate(vols))))].index
kospi_daily.drop(indexToRemove,inplace=True)

device = torch.device("cpu")
print("device :", device)

batch_size = 1

optimizer_betas = (0.9, 0.999)
learning_rate = 5.125e-4

num_epochs = 5000

evaluation_epoch_num = 500

class TimeseriesDataset(Dataset):
    def __init__(self, data_frame, sequence_length=2):
        self.data = torch.tensor(data_frame.values)
        self.sequence_length = sequence_length

    def __len__(self):
        return self.data.shape[0] - self.sequence_length + 1

    def __getitem__(self, index):
        return self.data[index: index + self.sequence_length].float()

columns_used_in_training = ["Close_x", "Open", "High", "Low", "CUMLOGRET_1", "RSI_14", "MACD_12_26_9", "SMA5"]
data_dimension = len(columns_used_in_training)
sequence_length = 7

class Generator(nn.Module):
    def __init__(self, hidden_size):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=data_dimension, hidden_size=hidden_size, num_layers=1, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(hidden_size, data_dimension)
        self.drop = nn.Dropout(0.2)

    def forward(self, input_sequences):
        input_sequences = self.drop(input_sequences)
        lstm_output, hidden_cell = self.lstm(input_sequences)
        res = self.linear(hidden_cell[0][-1])
        res = res.view(res.shape[0], 1, -1)
        return res

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(sequence_length*data_dimension, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_sequences):
        input_sequences_flattened = input_sequences.view(input_sequences.shape[0], -1)
        res = self.model(input_sequences_flattened)
        return res

train_data, rest_data = train_test_split(kospi_daily[columns_used_in_training], test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
scaler.fit(train_data)
train_data[train_data.columns] = scaler.transform(train_data)
rest_data[rest_data.columns] = scaler.transform(rest_data)

validation_data, test_data = train_test_split(rest_data, test_size=0.5, shuffle=False)

train_dataset = TimeseriesDataset(train_data, sequence_length)
test_dataset = TimeseriesDataset(test_data, sequence_length)
validation_dataset = TimeseriesDataset(validation_data, sequence_length)

# create the dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
real_data_sample = next(iter(train_dataloader))
print("Real data sample shape:", real_data_sample.shape)

def model_rmse(model, dataloader, epoch, plot_graph=False, plot_title="Validation Predictions", show_preds=False):
    rmse = 0
    squared_error_list = []
    real_data_list = []
    predicted_data_list = []
    file_title = plot_title.lower().replace(" ", "_")
    for i, sequence_batch in enumerate(dataloader):
        with torch.no_grad():
            real_sequence = sequence_batch
            # Assign first t values
            generator_input_sequence = sequence_batch[:,:-1].to(device)
            real_values = sequence_batch[:,-1:]
            #  Generate (t+1)th value from first t values
            predicted_values = generator(generator_input_sequence).cpu()
            real_data_list.append(real_values)
            predicted_data_list.append(predicted_values)
    
    real_data = torch.cat(real_data_list, 0)
    predicted_data = torch.cat(predicted_data_list, 0)
    
    # Unscale data
    df_pred = pd.DataFrame(predicted_data.view(-1,len(columns_used_in_training)),columns=columns_used_in_training)
    df_pred_unscaled = pd.DataFrame(scaler.inverse_transform(df_pred),columns=columns_used_in_training)
    df_real = pd.DataFrame(real_data.view(-1,len(columns_used_in_training)),columns=columns_used_in_training)
    df_real_unscaled = pd.DataFrame(scaler.inverse_transform(df_real),columns=columns_used_in_training)
    
    if plot_graph:
        if not os.path.exists('./plots_fc_disc/'):
            os.makedirs('./plots_fc_disc/')
        
        for column in columns_used_in_training:
            # TODO: get x values and plot prediction of multiple columns
            fig = plt.figure(figsize=(16,8))
            plt.xlabel("Date")
            plt.ylabel(column)
            plt.title(plot_title + f" -{column}-")
            plt.plot(df_real_unscaled[column],label="Real")
            plt.plot(df_pred_unscaled[column],label="Predicted")
            # plt.ylim(bottom=0)
            plt.legend()
            if show_preds and column == "close":
                plt.show()
            fig.savefig(f'./plots_fc_disc/{file_title}_plt_{column}_e{epoch}.png')
    rmse_results = {}
    for column in columns_used_in_training:
        squared_errors = (df_real_unscaled[column] - df_pred_unscaled[column])**2
        rmse = np.sqrt(squared_errors.mean())
        rmse_results[column] = rmse
    return rmse_results

generator = Generator(hidden_size=data_dimension*2).to(device)
discriminator = Discriminator().to(device)
print("Generator and discriminator are initialized")

criterion = nn.BCELoss()
optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=optimizer_betas)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=optimizer_betas)

real_label = 1.
fake_label = 0.

if not os.path.exists('./models_fc_disc/'):
    os.makedirs('./models_fc_disc/')

best_predictor = None
min_close_rmse = math.inf

evaluation_metrics = {"gen_loss":[], "disc_loss":[], "rmse_values":{}}
for column in columns_used_in_training:
        evaluation_metrics["rmse_values"][column] = []



if __name__ == "__main__":
    print("Training is started")
    for epoch in range(num_epochs):
        for i, sequence_batch in enumerate(train_dataloader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Training with real batch
                discriminator.zero_grad()
                # Format batch
                real_sequence = sequence_batch.to(device)
                batch_size = real_sequence.size(0)
                real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                discriminator_output_real = discriminator(real_sequence).view(-1)
                # Calculate loss on all-real batch
                discriminator_error_real = criterion(discriminator_output_real, real_labels)
                # Calculate gradients for D in backward pass
                discriminator_error_real.backward()

                ## Training with fake batch
                # Assign first t values
                generator_input_sequence = sequence_batch[:,:-1].to(device)
                #  Generate (t+1)th value from first t values
                generated_values = generator(generator_input_sequence)
                fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
                # Concat first t real values and generated (t+1)th values
                generator_result_concat = torch.cat((generator_input_sequence, generated_values.detach()), 1)
                # Classify all fake batch with D
                discriminator_output_fake = discriminator(generator_result_concat).view(-1)
                # Calculate D's loss on the all-fake batch
                discriminator_error_fake = criterion(discriminator_output_fake, fake_labels)
                # Calculate the gradients for this batch
                discriminator_error_fake.backward()
                # Add the gradients from the all-real and all-fake batches
                discriminator_error = discriminator_error_real + discriminator_error_fake
                # Update D
                optimizer_discriminator.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                generator.zero_grad()
                real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
                # Since we just updated D, perform another forward pass of all-fake batch through D
                generator_result_concat_grad = torch.cat((generator_input_sequence, generated_values), 1)
                discriminator_output_fake = discriminator(generator_result_concat_grad).view(-1)
                # Calculate G's loss based on this output
                generator_error = criterion(discriminator_output_fake, real_labels)
                # Calculate gradients for G
                generator_error.backward()
                # Update G
                optimizer_generator.step()
        if (epoch+1) % evaluation_epoch_num == 0 or epoch+1 == 1:
            rmse_values = model_rmse(generator, validation_dataloader, epoch=(epoch+1), plot_graph=True)
            if rmse_values["Close_x"] < min_close_rmse:
                min_close_rmse = rmse_values["Close_x"]
                best_predictor = epoch+1
            for column in columns_used_in_training:
                evaluation_metrics["rmse_values"][column].append(rmse_values[column])
            evaluation_metrics["gen_loss"].append(generator_error.item())
            evaluation_metrics["disc_loss"].append(discriminator_error.item())
            print('\n[{}/{}]\tDiscriminator Loss: {:.4f}\tGenerator Loss: {:.4f}'
                      .format(epoch+1, num_epochs, discriminator_error.item(), generator_error.item()))
            for col_name, rmse in rmse_values.items():
                print(f"{col_name} RMSE: {rmse:.4f}")
            save_path = os.path.join("./models_fc_disc/","model_epoch_{}.pt".format(epoch+1))
            torch.save({
                'epoch': epoch+1,
                'generator_model_state_dict': generator.state_dict(),
                'discriminator_model_state_dict': discriminator.state_dict(),
                'optimizer_generator_state_dict': optimizer_generator.state_dict(),
                'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(), 
                'discriminator_loss': discriminator_error,
                'generator_loss': generator_error,
                }, save_path)
            print(best_predictor)