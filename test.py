import csv
import numpy as np
import pandas as pd

df = pd.read_csv('C:\My Projects\Fake News Detection using NLP\Test Data\english_fake_news_2212.csv')
#print(df)
df[df['label'] == "Real"]
#df.head()

# Open the CSV file in read mode 
#with open('C:\My Projects\Fake News Detection using NLP\Test Data\english_fake_news_2212.csv', mode='r') as file: 
  # Create a CSV reader object 
 # csv_reader = csv.reader(file)

  # Skip the header row (if there is one) 
  #next(csv_reader, None) 

  # Iterate over each row in the CSV file 
  #for row in csv_reader: 
   # print(row) 
