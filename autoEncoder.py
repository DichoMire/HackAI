import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math

device = torch.device('cuda')

# get distance values
#distance = pd.read_csv('./data/[REDACTED]')
distance = pd.DataFrame({}) # We are not allowed to share the data
distance = torch.tensor(distance.values, dtype=torch.float32).to(device)


# flight searches dataset
class FlightSearches(Dataset):
    def __init__(self):
        # data loading
        #xy = np.loadtxt('./data/[REDACTED]', delimiter=",", dtype=np.float32, skiprows=1)
        xy = "" # We are not allowed to share the data
        self.features = torch.from_numpy(xy)  # features
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # call an index to the data -> e.g. dataset[0]
        return self.features[index]

    def __len__(self):
        # call len(dataset)
        return self.n_samples


# events dataset
class Events(Dataset):
    def __init__(self):
        # data loading
        #xy = np.loadtxt('./data/[REDACTED]', delimiter=",", dtype=np.float32, skiprows=1)
        xy = "" # We are not allowed to share the data
        self.features = torch.from_numpy(xy)  # features
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # call an index to the data -> e.g. dataset[0]
        return self.features[index, :]

    def __len__(self):
        # call len(dataset)
        return self.n_samples


flightsData = FlightSearches()
eventsData = Events()
print(f"Events length = {len(eventsData)}")
print(f"Flight length = {len(flightsData)}")
print(f"Distance shape = {distance.shape}\n")

#df = pd.read_json('./data/[REDACTED]', orient='columns')
df = pd.DataFrame({}) # We are not allowed to share the data

geoloc = df["geolocation"]
listVal = []

for i in range(len(geoloc)):
    a = geoloc[i]["coordinates"]
    listVal.append(a)

longlatAirports = torch.FloatTensor(listVal).to(device)
print(f"Airport longlat shape = {longlatAirports.shape}\n")

# define hyperparameters
hyperparameters = {
    'num_days': 5,
    'input_size': 2048,
    'learning_rate': 0.1,
    'num_epochs': 20,
    'mini_batch': 256
}

# X is a list which contains the inputs to the auto-encoder => derivation of the X values is shown below
'''
X = torch.empty(hyperparameters['input_size'], 0).to(device)
for event in eventsData:
    bot_Th = event[2] - hyperparameters['num_days']
    top_Th = event[3] + hyperparameters['num_days']

    flights = flightsData[flightsData[:, 1] > bot_Th]
    flights = flights[flights[:, 2] < top_Th]

    length = flights.shape[0]
    if length % 2 != 0:
        mid = int(length / 2 + 1)
    else:
        mid = int(length / 2)

    flights = flights[int(mid - (hyperparameters['input_size'] / 2)):int(mid + (hyperparameters['input_size'] / 2)), :]

    X = torch.cat([X, flights[:, 0].reshape(-1, 1).to(device)], axis=1)                                             # this is the saving of the inputs for each event

X = X.reshape(-1, hyperparameters['input_size'])
print(X.shape)
inVal = X.tolist()

df = pd.DataFrame(inVal)
df.to_csv('./data/input.csv', index = False)

inVal = pd.read_csv('./data/input.csv')
X = torch.tensor(inVal.values, dtype=torch.float32).to(device)

print(X.shape)
'''

inVal = pd.read_csv('./data/input.csv')
X = torch.tensor(inVal.values, dtype=torch.float32).to(device)

temp = DataLoader(X, batch_size=hyperparameters['mini_batch'])
data = next(iter(temp))
mean = data.mean()
std = data.std()

# normalise X
X_normal = (X - mean) / std
loader = DataLoader(X_normal, batch_size=hyperparameters['mini_batch'])

print(f"There are {len(X_normal)} samples and we have a batch size of {hyperparameters['mini_batch']}, therefore, "
      f"there are ceil({len(X_normal)}/{hyperparameters['mini_batch']})={math.ceil(len(X_normal) / hyperparameters['mini_batch'])} iterations in each epoch.")


# define model
class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size)                         # could put a tanh activation after this to put values between -1 and 1
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode


#model = AutoEncoder(input_size=hyperparameters['input_size']).to(device)

# load previously defined model
PATH = 'autoencode.pth'

# Load previously trained model
model = AutoEncoder(hyperparameters['input_size']).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

# define loss and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])

# training loop
X = torch.empty(0, 2048).to(device)
reproduce = torch.empty(0, 2048).to(device)
print("Beginning training")
for epoch in range(hyperparameters['num_epochs']):
    for index, x in enumerate(loader):
        # forward pass
        reconstruct = model(x)
        loss = loss_fn(reconstruct, x)
        if (epoch+1) == hyperparameters['num_epochs']:
            for index in range(x.shape[0]):
                X = torch.cat([X, x[index, :].reshape(1, -1)], axis=0)
                reproduce = torch.cat([reproduce, reconstruct[index, :].reshape(1, -1)], axis=0)

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (index+1) % 20 == 0:
            print(f"Epoch = {epoch+1}: Iteration = {index+1} / {math.ceil(len(X_normal) / hyperparameters['mini_batch'])}: Loss = {loss:.3f}")


print(f"Reproduce = {reproduce.shape}")

torch.save(model.state_dict(), PATH)

# calculate the mean absolute error
difference = torch.mean(torch.abs(reproduce - X), 1, True)
normal_threshold = torch.mean(difference) + torch.std(difference)

difference = difference.reshape(-1).tolist()
print(f"The threshold for normality is at L = {normal_threshold:.3f}")

# the events that have a major effect are the events which have a difference > normal_threshold
Important = []
for index in range(len(difference)):
    if difference[index] > normal_threshold:
        Important.append(index)

#data = pd.read_csv('./data/[REDACTED]')
data = pd.DataFrame({}) # We are not allowed to share the data
columns = data['name']


importantEvents = [columns[i] for i in Important]
anomalies = ([columns[index] for index in Important])
anomalyLoss = [difference[index] for index in Important]
print(f"Anomalous events = {anomalies}, Loss = {anomalyLoss}")

anomalyLoss, anomalies = zip(*sorted(zip(anomalyLoss, anomalies), reverse=True))

print(f"Number of important events = {len(anomalies)}")
str = ''
for anomaly in anomalies[:100]:
    newVal = anomaly + '___'
    str += newVal

print(str)

#df = pd.DataFrame({'Anomalous Events': anomalies, 'Loss': anomalyLoss})
#df.to_csv('./data/anomalies.csv', index=False)

# draw the histogram
plt.hist(difference, bins=500)

# draw the vertical line
X = [normal_threshold.to('cpu').detach().numpy(), normal_threshold.to('cpu').detach().numpy()]
Y = [0, 50]
plt.plot(X, Y, c='r', linestyle='--', label='Threshold', linewidth=3)

plt.title("Graph showing the variation in the loss for each event.")
plt.xlabel("Loss")
plt.ylabel("Num Samples")
plt.ylim(0.5)
plt.legend()
plt.show()
