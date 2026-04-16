import pandas as pd # data manipulation
import matplotlib.pyplot as plt # data visualization
import sweetviz

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN # machine learning algorithms
from sklearn.metrics import silhouette_score

from sqlalchemy import create_engine # connect to SQL database
from urllib.parse import quote

# Load Wine data set
df = pd.read_csv(r"AirTraffic_Passenger_Statistics.csv")

# Credentials to connect to Database
user = '****' # user name
pw = '*****' # password
db = 'air_routes_db' # database
# creating engine to connect MySQL database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
df.to_sql('airline_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from airline_tbl;'
df = pd.read_sql_query(sql, engine)

# Data types
df.info()
df.isnull().sum()

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***

df.describe()

df.duplicated().sum()

my_report = sweetviz.analyze([df, "df"])
my_report.show_html('Report.html')

# As we can see there are multiple columns in our dataset, 
# but for cluster analysis we will use 
# Operating Airline, Geo Region, Passenger Count and Flights held by each airline.
df1 = df[["Operating Airline", "GEO Region",  "Passenger Count"]]

airline_count = df1["Operating Airline"].value_counts()
airline_count.sort_index(inplace=True)

passenger_count = df1.groupby("Operating Airline").sum()["Passenger Count"]
passenger_count.sort_index(inplace=True)

'''So as this algorithms is working with distances it is very sensitive to outliers, 
that’s why before doing cluster analysis we have to identify outliers and remove them from the dataset. 
In order to find outliers more accurately, we will build the scatter plot.'''

df2 = pd.concat([airline_count, passenger_count], axis=1)
# x = airline_count.values
# y = passenger_count.values
plt.figure(figsize = (10,10))
plt.scatter(df2['count'], df2['Passenger Count'])
plt.xlabel("Flights held")
plt.ylabel("Passengers")
for i, txt in enumerate(airline_count.index.values):
    a = plt.gca()
    plt.annotate(txt, (df2['count'][i], df2['Passenger Count'][i]))
plt.show()

df2.index
# We can see that most of the airlines are grouped together in the bottom left part of the plot, 
# some are above them, and it has 2 outliers United Airlines and Unites Airlines — Pre 07/01/2013.
# So let’s get rid of them.

index_labels_to_drop = ['United Airlines', 'United Airlines - Pre 07/01/2013']
df3 = df2.drop(index_labels_to_drop)



# Generate clusters using Agglomerative Hierarchical Clustering
ac = AgglomerativeClustering(2, linkage = 'ward')
ac_clusters = ac.fit_predict(df3)

# Generate clusters from K-Means
km = KMeans(2)
km_clusters = km.fit_predict(df3)

# Generate clusters using DBSCAN
db_param_options = [[8000000, 2], [7500000, 2], [8200000, 2], [6800000, 3], [6500000, 3], [6000000, 3]]

for ep, min_sample in db_param_options:
    db = DBSCAN(eps = ep, min_samples = min_sample)
    db_clusters = db.fit_predict(df3)
    print("Eps: ", ep, "Min Samples: ", min_sample)
    print("DBSCAN Clustering: ", silhouette_score(df3, db_clusters))

# Generate clusters using DBSCAN
db = DBSCAN(eps = 8000000, min_samples = 2)
db_clusters = db.fit_predict(df3)

plt.figure(1)
plt.title("Airline Clusters from Agglomerative Clustering")
plt.scatter(df3['count'], df3['Passenger Count'], c = ac_clusters, s = 50, cmap = 'tab20b')
plt.show()

plt.figure(2)
plt.title("Airline Clusters from K-Means")
plt.scatter(df3['count'], df3['Passenger Count'], c = km_clusters, s = 50, cmap = 'tab20b')
plt.show()

plt.figure(3)
plt.title("Airline Clusters from DBSCAN")
plt.scatter(df3['count'], df3['Passenger Count'], c = db_clusters, s = 50, cmap = 'tab20b')
plt.show()


# Calculate Silhouette Scores
print("Silhouette Scores for Wine Dataset:\n")

print("Agg Clustering: ", silhouette_score(df3, ac_clusters))

print("K-Means Clustering: ", silhouette_score(df3, km_clusters))

print("DBSCAN Clustering: ", silhouette_score(df3, db_clusters))

## saving dbscan
import pickle
pickle.dump(db, open('db.pkl', 'wb'))

model = pickle.load(open('db.pkl', 'rb'))

res = model.fit_predict(df3)

