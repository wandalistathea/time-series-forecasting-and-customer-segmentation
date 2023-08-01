# Import Packages -----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# Data Preparation -----
transaction = pd.read_csv("Data Project Kalbe/Case Study - Transaction.csv", sep = ";")
# Change date type
transaction["Date"] = pd.to_datetime(transaction["Date"], format = "%d/%m/%Y")

customer = pd.read_csv("Data Project Kalbe/Case Study - Customer.csv", sep = ";")
product = pd.read_csv("Data Project Kalbe/Case Study - Product.csv", sep = ";")
store = pd.read_csv("Data Project Kalbe/Case Study - Store.csv", sep = ";")
# Drop some columns from "store" dataframe
store = store.drop(columns = ["Latitude", "Longitude"])

# Merge "transaction" dataframe with other tables
df = transaction.merge(customer, how = "left", on = "CustomerID").merge(store, how = "left", on = "StoreID").merge(product, how = "left", on = "ProductID")
df = df.drop(columns = ["Price_y"])
df.rename(columns = {"Price_x": "Price"}, inplace = True)

# Customer Segmentation -----

# Create a new dataframe for clustering
# Group data by the "CustomerID" column, calculate these things:
# the count of "TransactionID", the sum of "Qty", and the sum of "TotalAmount" 

cl_df = df[["CustomerID", "TransactionID", "Qty", "TotalAmount"]].groupby("CustomerID").agg({"TransactionID": "count", 
                                                                                             "Qty": "sum", 
                                                                                             "TotalAmount": "sum"})

# Boxplot
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5))
ax[0].boxplot(cl_df["TransactionID"])
ax[0].set_title("TransactionID")
ax[1].boxplot(cl_df["Qty"])
ax[1].set_title("Qty")
ax[2].boxplot(cl_df["TotalAmount"])
ax[2].set_title("TotalAmount")
plt.show()

# Feature Scaling 
# Using StandardScaler because the data have some outliers
scaler = StandardScaler()
scaler.fit_transform(cl_df[["TransactionID", "Qty", "TotalAmount"]])

cl_df_scaler = pd.DataFrame(scaler.fit_transform(cl_df[["TransactionID", "Qty", "TotalAmount"]]), columns = [col + "_stand" for col in list(cl_df.columns)])
# Change the index of "cl_df_scaler" 
cl_df_scaler.set_index(cl_df.index, inplace = True)

# Concate 2 dataframes and save it using the name of the first dataframe ("cl_df")
cl_df = pd.concat([cl_df, cl_df_scaler], axis = 1)

# Cluster the customer by the data using KMeans
# Determine the best number of clusters (k) using Elbow Method
wcss = [] # within cluster sum of square

for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(cl_df_scaler)
    wcss.append(kmeans.inertia_)
    
# Visualization
plt.plot(range(1, 11), wcss, "o-")
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Consider to use k = 3 or k = 4
def df_centroid(df_clustering, df_result, k):
    
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(df_clustering)
    
    df_result[f"label_kmeans_{k}"] = kmeans.labels_
    
    df_centroid = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                               columns = ["TransactionID_centroid", "Qty_centroid", "TotalAmount_centroid"], 
                               index = [f"Cluster {i}" for i in range(1, k + 1)])
    
    return df_centroid

centroid3 = df_centroid(df_clustering = cl_df_scaler, df_result = cl_df, k = 3)
centroid4 = df_centroid(df_clustering = cl_df_scaler, df_result = cl_df, k = 4)

fig = plt.figure(figsize = (15, 5))

# Using 3 clusters
ax1 = fig.add_subplot(121, projection = "3d")
# Scatter plot the data points for each cluster
for cluster_id in range(3):
    cluster_data = cl_df[cl_df["label_kmeans_3"] == cluster_id]
    ax1.scatter(cluster_data["TransactionID"], cluster_data["Qty"], cluster_data["TotalAmount"], 
                label = f"Cluster {cluster_id + 1}")
ax1.scatter(centroid3["TransactionID_centroid"], centroid3["Qty_centroid"], centroid3["TotalAmount_centroid"], 
            c = "black", s = 30, label = "Centroid")
ax1.set_xlabel("TransactionID")
ax1.set_ylabel("Qty")
ax1.set_zlabel("TotalAmout")
ax1.set_title("Graphic for 3 Cluters")
ax1.legend()

# Using 4 clusters
ax2 = fig.add_subplot(122, projection = "3d")
# Scatter plot the data points for each cluster
for cluster_id in range(4):
    cluster_data = cl_df[cl_df["label_kmeans_4"] == cluster_id]
    ax2.scatter(cluster_data["TransactionID"], cluster_data["Qty"], cluster_data["TotalAmount"], 
                label = f"Cluster {cluster_id + 1}")
ax2.scatter(centroid4["TransactionID_centroid"], centroid4["Qty_centroid"], centroid4["TotalAmount_centroid"], 
            c = "black", s = 30, label = "Centroid")
ax2.set_xlabel("TransactionID")
ax2.set_ylabel("Qty")
ax2.set_zlabel("TotalAmout")
ax2.set_title("Graphic for 4 Cluters")
ax2.legend()
plt.show()

# Show the interactive 3D plot (3 clusters)
fig = px.scatter_3d(cl_df, x = "TransactionID", y = "Qty", z = "TotalAmount",
                    color = "label_kmeans_3", size_max = 8, opacity = 0.8,
                    color_continuous_scale = "Viridis", title = "3D Scatter Plot of Customer",
                    labels = {"TransactionID": "TransactionID", "Qty": "Qty", "TotalAmount": "TotalAmount"})
fig.show()

# Show the interactive 3D plot (4 clusters)
fig = px.scatter_3d(cl_df, x = "TransactionID", y = "Qty", z = "TotalAmount",
                    color = "label_kmeans_4", size_max = 8, opacity = 0.8,
                    color_continuous_scale = "Viridis", title = "3D Scatter Plot of Customer",
                    labels = {"TransactionID": "TransactionID", "Qty": "Qty", "TotalAmount": "TotalAmount"})
fig.show()

print(cl_df)