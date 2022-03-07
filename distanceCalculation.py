import torch
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# define device
device = torch.device('cuda')

# define the Airports tensor
#df = pd.read_json('./data/[REDACTED]', orient='columns')
df=pd.DataFrame({}) # We are not allowed to share the data

geoloc = df["geolocation"]
listVal = []

for i in range(len(geoloc)):
    a = geoloc[i]["coordinates"]
    listVal.append(a)

longlatAirports = torch.FloatTensor(listVal).to(device)

# define the events tensor
#df = pd.read_csv('./data/[REDACTED')
df=pd.DataFrame({}) # We are not allowed to share the data


lat = df["lat"]
long = df["lng"]

latEvents = torch.FloatTensor(lat).reshape(-1, 1).to(device)
longEvents = torch.FloatTensor(long).reshape(-1, 1).to(device)

longlatEvents = torch.cat([longEvents, latEvents], axis=1)
print(longlatEvents.shape, longlatAirports.shape)


def calculateEuclideanDistance(longlatEvents, longlatAirports):
    res = []
    for airport in longlatAirports:
        euclideanDistance = []
        for event in longlatEvents:
            lon1 = event[0].item()
            lat1 = event[1].item()
            lon2 = airport[0].item()
            lat2 = airport[1].item()

            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            # haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            # Radius of earth in kilometers is 6371
            km = 6371 * c
            euclideanDistance.append(km)
        res.append(euclideanDistance)

    distance = res
    return distance


distance = calculateEuclideanDistance(longlatEvents, longlatAirports)

df = pd.DataFrame(distance)
df.to_csv('./data/distances.csv', index = False)
