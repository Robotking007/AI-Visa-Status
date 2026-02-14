# Batch 3- EHR

# Pandas Data Analysis

# import pandas as pd

# data ={patient_id- 001, 002, 003
# name- John, Smith, Bob 
# age- 39, 60, 45
# digaonsis- Asthma, Hypertension, Diabetes
# Blood pressure - 200, 150, 300
# risk categor    
# normalised bp 
# }


# dataframe= pd.DataFrame(data)

# Average age of the pateinerts: print("\n Average age of the patients: ", dataframe['age'].mean())

# print("Average BP:", dataframe["blood pressure"].mean())

# print(dataframe["Name"])

# print(dataframe['blood pressure']>150)

# dataframe['Risk']= dataframe["blood pressure"].apply(lambda x: 'High' if x>150 else 'Low')
# print("risk column", dataframe)

# Numpy for python 

# import numpy as np 

# heart_rates= np.array([72, 88, 95])

# print("Average heart rate: ", np.mean(heart_rates))

# np.max(heart_rates)- maximum heart heart_rate

# np.min(heart_rates)- minimum heart rate

# np.median(heart_rates)- median heart rate

# np.std(heart_rates) standard deviation

# pandas + numpy- for a calculation/ analysis 

# dataframe['normalised_BP'] = (dataframe['blood_pressure'] - np.mean(dataframe['blood_pressure'])) / np.std(dataframe['blood_pressure'])
# print("normalised_bf", dataframe)



# Visa Status- Batch 8/9/10

# Pandas Data Analysis


import pandas as pd
import numpy as np
from datetime import datetime

data = {
    "application_date": ["2024-01-01", "2024-02-15", "2024-03-10"],
    "decision_date": ["2024-02-01", "2024-03-20", "2024-04-05"],
    "country": ["India", "USA", "UK"],
    "visa_type": ["Student", "Tourist", "Work"]
}

df = pd.DataFrame(data)
print(df)

#Calculate processing time

df["application_date"] = pd.to_datetime(df["application_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])

df["processing_days"] = (df["decision_date"] - df["application_date"]).dt.days
print(df)


# Average processing time

print("Average processing time:", df["processing_days"].mean())

# Filter rows

print(df["processing_days"] > 25)

# create a risk column 
df["risk"] = df["processing_days"].apply(
    lambda x: "High Delay" if x > 30 else "Low Delay"
)
print(df)

# Numpy for calculations

processing_array = np.array(df["processing_days"])

print("Mean:", np.mean(processing_array))
print("Median:", np.median(processing_array))
print("Max:", np.max(processing_array))
print("Min:", np.min(processing_array))
print("Std Dev:", np.std(processing_array))

# Normalisation

df["normalized_processing"] = (
    (df["processing_days"] - np.mean(df["processing_days"])) /
    np.std(df["processing_days"])
)

print(df)



# Basics of ML: 

# data- analysis, cleaning, getting ready for the processing 
# ML- accurate predict something 
# weather data- analysis, cleaned, encoded, preprocessing for the ML pipelines

# 1. Supervised learning - input(data from the user) + ouput (data from the user)- labelled data
# predict the disease risk (Yes/no) from vitals- covid- how many got affected/not affected, age, symptms that occured, vitals(lab reports-blood test, urine test,...)
# bob, 32, fever, blood report, urin report-  extremely high -- covid 
# alice, 23, cough, blood, urine repot- extremly high -- no covid

# -- Logisitic regression, Decision tree, Random forest, Gradient boosting, neural netowrks 

# bob, 32, fever, blood report, urin report-  extremely high -- covid 
# alice, 23, cough, blood, urine repot- extremly high -- no covid


# david, 20, fever, urine+bllo report- normaal - no covid


# 2. unsupervised learning - input(given from the user), output- model trained+ tested - unlabelled data 
# -- clusters/ classifys depending upon the groups based on the syatmpos/classified outcome
# a class of 5 people, each student is aimed to select only 2 subjects, M, C, P, B, E
# 1- e, m
# 2- c, p
# 3- e, c
# 4- b, c
# 5- p, e 
# e- 3, m- 1, b- 1, p- 2, c-3

# -- Kmeans clustering, Classification, KNN classifier 

# decsion tree- 

# making a question- is my blood pressure > 140 
# yes- hypertension - pr4edicted
# no- normal- predicted ans 

# adv: easy to understand + explain
# works with both categorical + numerical data 
# is not needed to perform for scaling/normalization 

# disadv: 
# decision/ predictions can overfit (memorize trainijg data )
# if data- complex, decsion tree accuracy will be low most of the time - we use Random Forest / graident boosting 
 