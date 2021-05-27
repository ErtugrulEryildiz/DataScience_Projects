"""
    Autor: Ertugrul Eryildiz
    SBUID: 112495660
    CSE 351-01 Assingment #1
    Airbnb Project
"""

import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import scipy.stats
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Following Libraries may not be necessary
# import dash
# import dash_core_components as dcc
# import dash_html_components as html

"""
    Task 1
    Only missing data are in these columns
    - name (post name) can be missing (FIX: "not-available")
    - host name can be missing (FIX: NO-NAME)
    - last review can be missing (FIX: "1970-01-01")
    - reviews_per_month can be missing (FIX: 0)
"""
raw_data = pd.read_csv('archive/AB_NYC_2019.csv')
columns = raw_data.columns

# If post name or host name data is not available, just put not-available since they are not a crucial part
# of the data for our purposes. When we are doing word clouds, we can simply ignore them
for row in range(len(raw_data['name'])):
    if pd.isna(raw_data['name'][row]):
        raw_data.at[row, 'name'] = 'not-available'

for row in range(len(raw_data['host_name'])):
    if pd.isna(raw_data['host_name'][row]):
        raw_data.at[row, 'host_name'] = 'not-available'

# If last review date is not available, just assign it an outlier value.
for row in range(len(raw_data['last_review'])):
    if pd.isna(raw_data['last_review'][row]):
        raw_data.at[row, 'last_review'] = '1970-01-01'

# If last review per month is not available, just assign it an outlier value.
for row in range(len(raw_data['reviews_per_month'])):
    if pd.isna(raw_data['reviews_per_month'][row]):
        raw_data.at[row, 'reviews_per_month'] = 0

# Write data to a file.
raw_data.to_csv('airbnb_cleaned.csv')

airbnb_data = pd.read_csv('airbnb_cleaned.csv')

'''
    task 2
'''
# Group by neighborhood
# sort each group by its price and find top and bottom 5 neighborhood

# save neighbourhoods and data frames in tuples
top_5s = list()
bottom_5s = list()

neighborhood_gp = airbnb_data.groupby('neighbourhood')

for neighborhood, gp in neighborhood_gp:
        if len(gp.index) > 5:
            sorted_df = gp.sort_values(by='price', ascending=False)
            top_5s.append((neighborhood, sorted_df.head(5)))
            bottom_5s.append((neighborhood, sorted_df.tail(5)))


# Groupby borough sort each group by its price and show it on box-whisker plot to analyze.
boroughs = airbnb_data.groupby('neighbourhood_group')

# Box=whisker plot figure to analyze 5 boroughs with their prices
fig = go.Figure()

for b_name, borough in boroughs:
    fig.add_trace(go.Box(y=borough['price'], name=b_name))

fig.update_layout(dict(title='NYC Borough Prices',
                       xaxis=dict(title=dict(text='Boroughs', font=dict(size=18))),
                       yaxis=dict(title=dict(text='Prices', font=dict(size=18))))
                  )
fig.show()
'''
    task 3 - Pearson correlation between two interesting feature (# of reviews and price) 
'''

# interesting set of features
features = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
corr = [ [ [col1, scipy.stats.pearsonr(airbnb_data[col1], airbnb_data[col2])[0] ] for col2 in features ] for col1 in features]

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=[[corr_pair[1] for corr_pair in feature_corr] for feature_corr in corr],
        x=features,
        y=features
    )
)

fig.update_layout(dict(title=dict(text='Listing Correlation', font=dict(size=32)),
                       xaxis=dict(title=dict(text='Features', font=dict(size=18))),
                       yaxis=dict(title=dict(text='Features', font=dict(size=18))))
                  )

fig.show()
'''
    task 4 - The Latitude and Longitude of all the Airbnb listings are provided in the dataset. 
'''
fig1 = go.Figure()
fig2 = go.Figure()
filtered_data = airbnb_data[airbnb_data['price'] <= 1000]
boroughs = filtered_data.groupby(by='neighbourhood_group')

for borough in boroughs:
    fig1.add_trace(go.Scatter(
        x=borough[1]['longitude'],
        y=borough[1]['latitude'],
        name=borough[0],
        mode='markers'))

fig1.update_layout(dict(
    title=dict(text="NYC Airbnb Listing Locations", font=dict(size=32)),
    xaxis=dict(title=dict(text="Longitude", font=dict(size=18))),
    yaxis=dict(title=dict(text="Latitude", font=dict(size=18)))
))

fig1.show()

for borough in boroughs:
    fig2.add_trace(go.Scatter(
        x=borough[1]['longitude'],
        y=borough[1]['latitude'],
        name=borough[0],
        mode='markers',
        marker=dict(
            color=borough[1]['price'],
            colorscale='Viridis',
            showscale=True if len(fig2.data) == 0 else False
        ),
        text=borough[1]['price'],
        showlegend=False
    ))

fig2.show()

'''
    task 5 - Word Cloud
'''
all_words = ''
stopwords = set(STOPWORDS)

for post in airbnb_data['name']:
    post = str(post)
    words = post.split()

    for index in range(len(words)):
        words[index] = words[index].lower()

    all_words += " ".join(words) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(all_words)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()
'''
    task 6 - Finding the busiest (hosts with high number of listings) areas.
'''
# Group by boroughs
# then Group by host name
# then Sort by number of listings
# Analyze why these hosts can be the busiest one
# For each host, compare global mean and median values for reviews_per_month, price, availability with users.

topHosts = []
figures = [go.Figure() for i in range(3)]
features = ['reviews_per_month', 'price', 'availability_365']
titles = ['Busiest Hosts by Reviews per Month', 'Busiest Hosts by Listing Price', 'Busiest Hosts by Availability']

boroughs = airbnb_data.sort_values(by='calculated_host_listings_count', ascending=False)\
    .groupby('neighbourhood_group', sort=False)

for name, pd in boroughs:
    index = 0
    hosts = pd.groupby('host_id', sort=False)
    for h_id, df in hosts:
        hosts_listing = airbnb_data.loc[airbnb_data['host_id'] == h_id]
        host_name = hosts_listing['host_name'].iloc[0]
        topHosts.append([name, host_name,
                         (hosts_listing['reviews_per_month'].mean(), hosts_listing['reviews_per_month'].median()),
                         (df['price'].mean(), df['price'].median()),
                         (df['availability_365'].mean(), df['availability_365'].median()),
                         ])
        break

# Add Global mean and median values
for i in range(3):
    figures[i].add_trace(go.Bar(
        name='Global',
        x=['mean', 'median'],
        y=[airbnb_data[features[i]].mean(), airbnb_data[features[i]].median()]
    ))

# Add top hosts' mean and median values
for i in range(3):
    for j in range(5):
        figures[i].add_trace(go.Bar(
            name=str(topHosts[j][0]) + str(topHosts[j][1]),
            x=['mean', 'median'],
            y=[topHosts[j][2+i][0], topHosts[j][2+i][1]]
        ))

for i in range(3):
    figures[i].update_layout(dict(
        title=dict(text=titles[i], font=dict(size=32)),
        xaxis=dict(title=dict(text="Mean - Median Values", font=dict(size=18))),
        yaxis=dict(title=dict(text=features[i], font=dict(size=18)))
    ))

for i in range(3):
    figures[i].show()
'''
    task 7 - 2 Unique Plots
'''
# Figure 1 - types of rooms by each borough
fig1 = go.Figure()
data = dict()
for borough, b_df in airbnb_data.groupby(by='neighbourhood_group'):
    data[borough] = dict()
    for room_type, r_df in b_df.groupby(by='room_type'):
        data[borough][room_type] = len(r_df.index)


for borough, room_types in data.items():
    fig1.add_trace(
        go.Bar(
            x=[x for x in room_types.keys()],
            y=[x for x in room_types.values()],
            name=borough
        )
    )


fig1.show()
# Figure 2 - neighborhood median price and borough correlation (Are certain boroughs are more expensive??)
features = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
corr = [ [ [col1, scipy.stats.pearsonr(airbnb_data[col1], airbnb_data[col2])[0] ] for col2 in features ] for col1 in features]

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=[[corr_pair[1] for corr_pair in feature_corr] for feature_corr in corr],
        x=features,
        y=features
    )
)

fig.update_layout(dict(title=dict(text='Listing Correlation', font=dict(size=32)),
                       xaxis=dict(title=dict(text='Features', font=dict(size=18))),
                       yaxis=dict(title=dict(text='Features', font=dict(size=18))))
                  )

fig.show()
