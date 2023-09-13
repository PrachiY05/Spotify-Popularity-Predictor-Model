#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


track= pd.read_csv('tracks_features.csv')


# In[9]:


track.head()


# In[50]:


track.columns


# In[6]:


track.isna().sum()


# In[7]:


track.describe()


# In[11]:


# View distribution for non-numeric columns
for column in track.select_dtypes(include=['object']).columns:
    print(f"\nColumn: {column}")
    print(track[column].value_counts())


# In[12]:


import pandas as pd

# Sample dataframe
# df = pd.read_csv('your_dataset_path.csv')

# For each column, print potential outliers
for column in track.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = track[column].quantile(0.25)
    Q3 = track[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect outliers
    outliers = track[(track[column] < lower_bound) | (track[column] > upper_bound)]
    print(f"Potential outliers for column {column}: {len(outliers)}")


# In[16]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(20, 15))

# Loop through all the numeric columns and plot boxplots
for index, column in enumerate(track.select_dtypes(include=['float64', 'int64']).columns):
    plt.subplot(4, 4, index+1)  # Adjust grid values (4, 4) depending on the number of columns
    sns.boxplot(y=track[column])
    plt.title(f"Boxplot of {column}")
    plt.tight_layout()

plt.show()


# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample dataframe loading
# df = pd.read_csv('your_dataset_path.csv')

# For each numeric column, plot the boxplot and display outlier values
for column in track.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 6))
    
    # Plotting the boxplot
    sns.boxplot(x=track[column], orient='h')
    plt.title(f"Boxplot for {column}")
    
    # Calculating IQR for each column
    Q1 = track[column].quantile(0.25)
    Q3 = track[column].quantile(0.75)
    IQR = Q3 - Q1

    # Outlier boundaries
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    # Display the outlier values for each column
    outliers = track[(track[column] < lower_bound) | (track[column] > upper_bound)]
    print(f"Outliers for {column}:")
    print(outliers[column].sort_values().unique())
    
    plt.show()


# In[20]:


duplicated_ids = track[track['id'].duplicated(keep=False)]
print(duplicated_ids)


# In[22]:


track['combined_name_artist'] = track['name'] + ' ' + track['artists'].astype(str)
duplicated_names = track[track['combined_name_artist'].duplicated(keep=False)]
print(duplicated_names[['name', 'artists']])


# In[23]:


duplicated_tracks = track[track.duplicated(subset=['album_id', 'track_number'], keep=False)]
print(duplicated_tracks[['album_id', 'track_number', 'name', 'artists']])


# In[26]:


# Create a combined column for name + artists to ease the duplication check
track['combined_name_artist'] = track['name'] + ' ' + track['artists'].astype(str)

# Get duplicate rows based on 'combined_name_artist'
duplicates_df = track[track.duplicated(subset='combined_name_artist', keep=False)].sort_values(by='combined_name_artist')


# In[28]:


duplicates_df.head()


# In[29]:


duplicated_rows = track[track.duplicated(keep=False)]
print(duplicated_rows)


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Number of unique tracks, albums, artists.
num_tracks = track['id'].nunique()
num_albums = track['album_id'].nunique()
num_artists = track['artist_ids'].nunique()

print(f"Number of unique tracks: {num_tracks}")
print(f"Number of unique albums: {num_albums}")
print(f"Number of unique artists: {num_artists}")

# The range of years the tracks span.
min_year = track['year'].min()
max_year = track['year'].max()

print(f"Tracks range from the year {min_year} to {max_year}")


# In[36]:


track['year'].max()


# In[42]:


year_0_data = track[track['year'] == 0]

# Display the first few rows of this subset
print(year_0_data)


# In[39]:


num_year_0 = year_0_data.shape[0]
print(f"Number of tracks from the year 0: {num_year_0}")


# In[43]:


year_0_data = track[track['year'] == 0]

# Save this subset to a CSV
year_0_data.to_csv('year_0_data.csv', index=False)


# In[46]:


track['year'].unique()


# In[45]:


track['year'] = track['year'].replace(0, 2018)


# In[47]:


track['year'].min()


# In[48]:


year_data = track[track['year'] == 1900]

# Save this subset to a CSV
year_data


# In[53]:


import requests
import csv

# Authentication
CLIENT_ID = '88ad66c6218f4b55b3655ace633dc7a0'
CLIENT_SECRET = '6dabb129ff504a3297dbe04d96de912e'

# Get token
auth_response = requests.post('https://accounts.spotify.com/api/token', {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})



auth_response_data = auth_response.json()
access_token = auth_response_data['access_token']

headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}

track_ids = list(track['id'])
batch_size = 50  # Spotify's API allows up to 50 tracks per request

with open('track_popularity.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'popularity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(0, len(track_ids), batch_size):
        batched_ids = track_ids[i:i+batch_size]
        ids_string = ','.join(batched_ids)
        tracks_info = requests.get(f"https://api.spotify.com/v1/tracks?ids={ids_string}", headers=headers).json()

        for track_info in tracks_info['tracks']:
            writer.writerow({'id': track_info['id'], 'popularity': track_info['popularity']})

print("Popularity data saved to track_popularity.csv")


# In[54]:


track.shape


# In[61]:


# Given your list of track IDs
track_ids = list(track['id'])

# Find the index of a specific ID
specific_id = "1wsRitfRRtWyEapl0q22o8"  # Replace with the ID you're looking for
index = track_ids.index(specific_id)
print(f"The index of '{specific_id}' is: {index}")


# In[63]:


import pandas as pd

# 1. Read the CSV files into pandas DataFrames

popularity_df = pd.read_csv("track_popularity.csv")

# 2. Trim the popularity dataset down to 170,400 rows
popularity_df = popularity_df.head(170400)

# 3. Merge the two DataFrames on the `id` column
merged_df = pd.merge(track, popularity_df, on="id", how="inner")

# Save the merged dataframe to a new CSV file
merged_df.to_csv("merged_dataset.csv", index=False)


# In[66]:


print(merged_df.head())
print(merged_df.shape)


# In[67]:


merged_df.isna().sum()


# In[69]:


# Assuming merged_df is the name of your dataframe

# Popularity
pop_mean = merged_df['popularity'].mean()
pop_median = merged_df['popularity'].median()
pop_std = merged_df['popularity'].std()
pop_range = merged_df['popularity'].max() - merged_df['popularity'].min()

print(f"Popularity:\nMean: {pop_mean}\nMedian: {pop_median}\nStandard Deviation: {pop_std}\nRange: {pop_range}")


# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns

# Popularity Distribution
plt.figure(figsize=(12,6))
sns.histplot(merged_df['popularity'], kde=True, bins=30)
plt.title('Distribution of Track Popularity')
plt.xlabel('Popularity')
plt.ylabel('Number of Tracks')
plt.show()


# In[72]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
correlation_matrix = merged_df.corr()

# Sorting correlations related to 'popularity' in descending order
popularity_correlation = correlation_matrix['popularity'].sort_values(ascending=False)

print(popularity_correlation)

# Visualizing the correlations using a heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Track Attributes')
plt.show()


# In[73]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' with your data

# Step 1: Filter top 10% popular tracks for each year
merged_df['rank'] = merged_df.groupby('year')['popularity'].rank(pct=True, method='first')
top_tracks = merged_df[merged_df['rank'] > 0.9]

# Step 2: Calculate mean value of attributes for top tracks each year
trends = top_tracks.groupby('year').agg({
    'danceability': 'mean',
    'tempo': 'mean',
    'instrumentalness': 'mean'
}).reset_index()

# Step 3: Plot the trends
plt.figure(figsize=(14, 6))

# Danceability trend
plt.plot(trends['year'], trends['danceability'], label='Danceability', marker='o')

# Tempo trend
plt.plot(trends['year'], trends['tempo'], label='Tempo', marker='o')

# Instrumentalness trend
plt.plot(trends['year'], trends['instrumentalness'], label='Instrumentalness', marker='o')

plt.title('Trend Analysis of Top Tracks Attributes Over the Years')
plt.xlabel('Year')
plt.ylabel('Attribute Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[74]:


missing_danceability = top_tracks['danceability'].isnull().sum()
zero_danceability = (top_tracks['danceability'] == 0).sum()

print(f"Missing danceability values in top tracks: {missing_danceability}")
print(f"Zero danceability values in top tracks: {zero_danceability}")


# In[75]:


plt.figure(figsize=(14, 6))

# Danceability trend
plt.plot(trends['year'], trends['danceability'], label='Danceability', marker='o')

plt.title('Danceability Trend Analysis of Top Tracks Over the Years')
plt.xlabel('Year')
plt.ylabel('Danceability Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[77]:


# Sorting the data by 'danceability' in descending order and taking the top 10
top_danceability_tracks = top_tracks.sort_values(by='danceability', ascending=False).head(10)

# Displaying the top 10 tracks with their 'danceability', 'year', and track title (assuming there's a 'title' column)
print(top_danceability_tracks[['name', 'danceability', 'year']])


# In[81]:


import matplotlib.pyplot as plt

# Sorting the data and fetching the top 10 tracks by danceability
top_10_danceable_tracks = merged_df.sort_values(by='danceability', ascending=False).head(10)

# Plotting
plt.figure(figsize=(12, 8))
plt.barh(top_10_danceable_tracks['name'], top_10_danceable_tracks['danceability'], color='skyblue')
plt.xlabel('Danceability')
plt.ylabel('Track Title')
plt.title('Top 10 Songs by Danceability')
plt.gca().invert_yaxis()  # To have the song with the highest danceability at the top
plt.tight_layout()
plt.show()


# In[79]:


import matplotlib.pyplot as plt

# Sorting the data and fetching the top 10 tracks by popularity
top_10_tracks = merged_df.sort_values(by='popularity', ascending=False).head(10)

# Plotting
plt.figure(figsize=(12, 8))
plt.barh(top_10_tracks['name'], top_10_tracks['danceability'], color='skyblue')
plt.xlabel('Danceability')
plt.ylabel('Track Title')
plt.title('Danceability of Top 10 Songs')
plt.gca().invert_yaxis()  # To have the song with the highest popularity at the top
plt.tight_layout()
plt.show()


# In[84]:


import pandas as pd

# Assuming you have a DataFrame called 'df' with columns 'album_name' and 'popularity'
album_popularity = merged_df.groupby('album')['popularity'].mean().sort_values(ascending=False)


# In[94]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
album_popularity.head(20).plot(kind='bar', color='skyblue')
print(album_popularity.head(20))
plt.title('Top 20 Albums by Average Track Popularity')
plt.ylabel('Average Popularity')
plt.xlabel('Album Name')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[89]:


import scipy.stats as stats

# One-way ANOVA
fval, pval = stats.f_oneway(*(merged_df['popularity'][merged_df['album'] == album] for album in merged_df['album'].unique()))
print("F-value:", fval)
print("P-value:", pval)


# In[93]:


# Average popularity by artist
artist_popularity = merged_df.groupby('artists')['popularity'].mean().sort_values(ascending=False)

# Average popularity by release year
year_popularity = merged_df.groupby('release_date')['popularity'].mean()

# Average popularity by genre
genre_popularity = merged_df.groupby('genre')['popularity'].mean().sort_values(ascending=False)


# In[105]:


import pandas as pd



# Filtering by track name
tracks_to_find = ['Get Lucky (feat. Pharrell Williams & Nile Rodgers) [Radio Edit]', 'Please Me', 'Goosebumps','Dancing in My Room']

filtered_df = merged_df[merged_df['name'].isin(tracks_to_find)][['name', 'artists', 'album', 'release_date','danceability']]

print(filtered_df)


# In[107]:


merged_df.columns


# In[108]:


#Which artists have the highest average track popularity?
# Group by artists and calculate mean popularity, then sort in descending order
artist_popularity = merged_df.groupby('artists')['popularity'].mean().sort_values(ascending=False)

# Visualization
plt.figure(figsize=(15, 8))
artist_popularity.head(10).plot(kind='bar', color='lightblue')  # Displaying top 10 artists
plt.title('Top 10 Artists by Average Track Popularity')
plt.ylabel('Average Popularity')
plt.xlabel('Artist Name')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()



# In[109]:


# Group by year and calculate mean popularity
yearly_popularity = merged_df.groupby('year')['popularity'].mean()

# Visualization
plt.figure(figsize=(15, 8))
yearly_popularity.plot(kind='line', color='green', marker='o')
plt.title('Average Track Popularity Over the Years')
plt.ylabel('Average Popularity')
plt.xlabel('Year')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[110]:


# Scatter plot to visualize relationship
plt.figure(figsize=(15, 8))
plt.scatter(merged_df['duration_ms'], merged_df['popularity'], alpha=0.5, color='orange')
plt.title('Relationship between Track Duration and Popularity')
plt.ylabel('Popularity')
plt.xlabel('Duration (in ms)')
plt.tight_layout()
plt.show()


# In[111]:


import matplotlib.pyplot as plt
import seaborn as sns

# Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='danceability', y='popularity', data=merged_df, hue='energy', palette="coolwarm", size='energy', sizes=(10, 200))
plt.title('Danceability vs Popularity colored by Energy')
plt.show()


# In[112]:


# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='explicit', y='popularity', data=merged_df)
plt.title('Popularity Distribution based on Explicit Content')
plt.show()


# In[116]:


import seaborn as sns
import matplotlib.pyplot as plt

# Update mode column to have string values instead of numerical
merged_df['mode'] = merged_df['mode'].replace({1: 'Major', 0: 'Minor'})

# Barplot
plt.figure(figsize=(15, 7))
sns.barplot(x='key', y='popularity', hue='mode', data=merged_df, ci=None)
plt.title('Average Popularity by Key and Mode')
plt.show()


# In[114]:


# Lineplot
plt.figure(figsize=(15, 6))
sns.lineplot(x='year', y='acousticness', data=merged_df, estimator='mean', ci=None)
plt.title('Trend of Acousticness Over the Years')
plt.show()


# In[120]:


# Scatter plot for track number vs. popularity
plt.figure(figsize=(15,7))
sns.scatterplot(x='track_number', y='popularity', data=merged_df, alpha=0.6, color='coral')
plt.title('Popularity vs Track Number within an Album')
plt.ylabel('Popularity')
plt.xlabel('Track Number')
plt.tight_layout()
plt.show()


# In[123]:


merged_df['explicit_count'] = merged_df['explicit'].astype(int)
explicit_proportion = merged_df.groupby('year')['explicit_count'].mean()
explicit_proportion.plot(kind='bar', figsize=(15,7))
plt.title('Proportion of Explicit Tracks Over Time')
plt.ylabel('Proportion')
plt.xlabel('Year')
plt.tight_layout()
plt.show()


# In[124]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,7))
sns.scatterplot(x='speechiness', y='popularity', data=merged_df, alpha=0.6, color='blue')
plt.title('Popularity vs Speechiness')
plt.ylabel('Popularity')
plt.xlabel('Speechiness')
plt.tight_layout()
plt.show()


# In[125]:


# Filter datasets
instrumental_tracks = merged_df[merged_df['instrumentalness'] > 0.8]
non_instrumental_tracks = merged_df[merged_df['instrumentalness'] <= 0.8]

# Calculate average popularity
avg_popularity_instrumental = instrumental_tracks['popularity'].mean()
avg_popularity_non_instrumental = non_instrumental_tracks['popularity'].mean()

# Bar plot
plt.figure(figsize=(10,6))
sns.barplot(x=['Instrumental', 'Non-Instrumental'], 
            y=[avg_popularity_instrumental, avg_popularity_non_instrumental], 
            palette="viridis")
plt.title('Average Popularity of Instrumental vs Non-Instrumental Tracks')
plt.ylabel('Average Popularity')
plt.tight_layout()
plt.show()


# In[126]:


# Code for visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.boxplot(x='disc_number', y='popularity', data=merged_df)
plt.title('Popularity of Tracks Based on Disc Number')
plt.xlabel('Disc Number')
plt.ylabel('Popularity')
plt.show()


# In[135]:


merged_df.columns


# Feature Engg
# 

# In[136]:


def categorize_period(year):
    if year < 1970:
        return '60s and before'
    elif year < 1980:
        return '70s'
    elif year < 1990:
        return '80s'
    elif year < 2000:
        return '90s'
    elif year < 2010:
        return '2000s'
    else:
        return '2010s and after'

merged_df['release_period'] = merged_df['year'].apply(categorize_period)


# In[137]:


def categorize_tempo(tempo):
    if tempo < 60:
        return 'Slow'
    elif tempo < 120:
        return 'Medium'
    else:
        return 'Fast'

merged_df['tempo_category'] = merged_df['tempo'].apply(categorize_tempo)


# In[138]:


def categorize_duration(duration):
    if duration < 2 * 60 * 1000:  # less than 2 minutes
        return 'Short'
    elif duration < 4 * 60 * 1000:  # less than 4 minutes
        return 'Medium'
    else:
        return 'Long'

merged_df['duration_category'] = merged_df['duration_ms'].apply(categorize_duration)


# In[139]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Compute the correlation matrix
selected_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'year', 'explicit_count', 'track_number']
corr = merged_df[selected_features].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(15, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# In[154]:


merged_df.head


# In[146]:


artist_popularity = merged_df.groupby('artists')['popularity'].mean().reset_index()
artist_popularity.columns = ['artists', 'artist_avg_popularity']


# In[147]:


merged_df = merged_df.merge(artist_popularity, on='artists', how='left')


# In[148]:


# List of required columns
required_columns = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'time_signature', 'explicit', 'track_number', 'disc_number',
    'year', 'explicit_count', 'release_period', 'tempo_category', 'duration_category',
    'artist_avg_popularity'  # Including the newly added feature
]

# Create a new dataframe with only the required columns
df = merged_df[required_columns].copy()


# In[149]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize the 'artist_avg_popularity' column
df['normalized_artist_popularity'] = scaler.fit_transform(df[['artist_avg_popularity']])

# Drop the original 'artist_avg_popularity' column
df.drop('artist_avg_popularity', axis=1, inplace=True)


# In[153]:


missing_data = df_selected.isnull().sum()
print(missing_data[missing_data > 0])


# In[155]:


df.head()


# In[156]:


#one hot encoding
# One-Hot Encoding the categorical columns
df_encoded = pd.get_dummies(df, columns=['release_period', 'tempo_category', 'duration_category'])


# In[158]:


#standardizing numeric features
from sklearn.preprocessing import StandardScaler

# List of numeric features to standardize
numeric_features = [
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 
    'tempo', 'duration_ms', 'explicit_count'
]

scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])


# In[163]:


df_encoded.columns


# In[164]:


df_encoded = df_encoded.rename(columns={"normalized_artist_popularity": "song_popularity"})


# In[165]:


from sklearn.model_selection import train_test_split

X = df_encoded.drop("song_popularity", axis=1)
y = df_encoded["song_popularity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[166]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[168]:


print(X_train.dtypes)


# In[169]:


# If mode is categorical like "Major" or "Minor", use encoding
X_train['mode'] = X_train['mode'].map({'Major': 1, 'Minor': 0})
X_test['mode'] = X_test['mode'].map({'Major': 1, 'Minor': 0})


# In[170]:


# Convert the explicit column to integers
X_train['explicit'] = X_train['explicit'].astype(int)
X_test['explicit'] = X_test['explicit'].astype(int)


# In[171]:


model.fit(X_train, y_train)


# In[172]:


train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R^2 Score: {train_score:.4f}")
print(f"Test R^2 Score: {test_score:.4f}")


# In[173]:


from sklearn.ensemble import RandomForestRegressor

# Create the model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Evaluate the model
train_score = rf_regressor.score(X_train, y_train)
test_score = rf_regressor.score(X_test, y_test)

print(f"Training R^2 Score: {train_score:.4f}")
print(f"Test R^2 Score: {test_score:.4f}")


# In[174]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameters to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor()

# Grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Train a new random forest using the best parameters
best_rf = RandomForestRegressor(**best_params)
best_rf.fit(X_train, y_train)

# Check the R^2 scores again
train_score = best_rf.score(X_train, y_train)
test_score = best_rf.score(X_test, y_test)

print(f"Training R^2 Score: {train_score:.4f}")
print(f"Test R^2 Score: {test_score:.4f}")


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Assuming you're using a RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# Define the parameter distributions
param_dist = {
    'n_estimators': randint(10, 200),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': randint(1, 20),
    'bootstrap': [True, False]
}

# Create the random search object
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, verbose=1, n_jobs=-1, random_state=42)

# Fit the data
random_search.fit(X_train, y_train)

# Check the best parameters
print(random_search.best_params_)

# Check the model's performance on the test set
print(random_search.score(X_test, y_test))


# In[ ]:




