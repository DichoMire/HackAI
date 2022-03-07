import pandas as pd
import numpy as np

daysmonths = [31,28,31,30,31,30,31,31,30,31,30,31] # Days in each month

def turndateintoint(initialdate, array):
    splitdate = initialdate.split('-')
    year, month, day = int(splitdate[0]), int(splitdate[1]), int(splitdate[2])
    days = (year - 2018) * 365
    for i in range(0,month-1):
        days += array[i]
    days += day - 1
    return days

#data = pd.read_csv('./data/[REDACTED]') # Import the first Dataset
data=pd.DataFrame({}) # We are not allowed to share the data
data = data.drop('origin', axis = 1) # Drop unneeded columns
data = data.drop('destination', axis = 1)
arrivalDate = data['arrival_date'] # Create variables from the columns
returnDate = data['return_date']   # that need to be transformed
data = data.drop('arrival_date', axis = 1) # Remove the old columns
data = data.drop('return_date', axis = 1)

a = np.max(arrivalDate) # Check the earliest date
b = np.max(returnDate) # Check the last date

print(a,b)

newArrivalDates = [] # New list which will contain the transformed dates

for i in range(len(arrivalDate)):
    newValues = turndateintoint(arrivalDate[i], daysmonths)
    newArrivalDates.append(newValues)

newReturnDates = [] # Repeat

for i in range(len(returnDate)):
    a = turndateintoint(returnDate[i], daysmonths)
    newReturnDates.append(a)

data = data.assign(arrival_date=newArrivalDates) # Add both lists to the dataset
data = data.assign(return_date=newReturnDates)

data.to_csv('./data/airportsInfo.csv', index=False) # Import dataset to .csv file, index=False removes an extra column named "Unnamed: 0"

# Repeat same process for the second dataset

data = pd.read_csv('dataset2.csv')
data = data.drop('description', axis = 1)
data = data.drop('exhibitors', axis = 1)
data = data.drop('location', axis = 1)
data = data.drop('name', axis = 1)
data = data.drop('type', axis = 1)
data = data.drop('visitors', axis = 1)

startDate = data['start_date']
endDate = data['end_date']

newStartDates = []
for i in range(len(newStartDates)):
    newValues = turndateintoint(startDate[i], daysmonths)
    newStartDates.append(newValues)

newEndDates = []
for i in range(len(endDate)):
    newValues = turndateintoint(endDate[i], daysmonths)
    newEndDates.append(newValues)

data = data.drop('start_date',axis = 1)
data = data.drop('end_date',axis = 1) # Remove the initial columns

data = data.assign(start_date = newStartDates)
data = data.assign(end_date = newEndDates)

data.to_csv('/data/eventsInfo.csv', index=False)