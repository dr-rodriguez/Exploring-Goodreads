# Map of the authors I've read

import pickle
from utilities import geo_api_search
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# Read in the author data
with open('data/author_info.pkl', 'r') as f:
    df_author = pickle.load(f)

# Parse the hometown with the Google API
data = list()
for i, row in df_author.iterrows():
    try:
        print('{} {}'.format(row['author'].encode('utf-8').decode('ascii', errors='ignore'),
                            row['hometown'].encode('utf-8').decode('ascii', errors='ignore')))
    except AttributeError:
        print('{} {}'.format(row['author'].encode('utf-8').decode('ascii', errors='ignore'),
                             row['hometown']))

    if row['hometown'] is None:
        status = None
    else:
        status, geo = geo_api_search(row['hometown'])

    if status is None:
        geo = [None, None]
        print('Author: {}  Status: FAIL'.format(row['author'].encode('utf-8').decode('ascii', errors='ignore')))
    else:
        print('Author: {}  Status: {} Location: {} {}'.format(row['author'].encode('utf-8').
                                                              decode('ascii', errors='ignore'), status, geo[0], geo[1]))

    data.append([geo[0], geo[1]])
    time.sleep(0.5)

df_coords = pd.DataFrame(data, columns=['lat', 'lon'])
full_data = df_author.join(df_coords)
full_data.dropna(inplace=True, axis=0, subset=['lon', 'lat'])

# Plot on map
fig = plt.figure()
themap = Basemap(projection='robin', lat_0=0, lon_0=-100,
                 resolution='l', area_thresh=10000.0)

# themap.drawcountries()
themap.fillcontinents(color='gainsboro')
themap.drawmapboundary()

# Add the data points
x, y = themap(full_data['lon'].values, full_data['lat'].values)
themap.plot(x, y, 'o', color='Green', markersize=6)
for label, xpt, ypt, lat, lon in zip(full_data['author'], x, y, full_data['lat'], full_data['lon']):
    if lat < 20 or lon > 100:
        plt.text(xpt, ypt+100000, label, ha='center', va='bottom')

plt.savefig('figures/fullmap.png')


# Zoom to US
fig = plt.figure()
# US_BOUNDING_BOX = "-125.00,24.94,-66.93,49.59"
themap = Basemap(projection='merc', lat_0=37, lon_0=-95,
                 resolution='l', area_thresh=10000.0,
                 llcrnrlon=-128.00, llcrnrlat=24.00,
                 urcrnrlon=-66.00, urcrnrlat=52.00)

# themap.drawcoastlines()
themap.drawcountries()
themap.drawstates()
themap.fillcontinents(color='gainsboro')
themap.drawmapboundary()

x, y = themap(full_data['lon'].values, full_data['lat'].values)
themap.plot(x, y, 'o', color='Green', markersize=10)

# Add labels
for label, xpt, ypt, lat in zip(full_data['author'], x, y, full_data['lat']):
    if lat < 35 or lat > 43.5:
        if label == 'Karl Schroeder':
            plt.text(xpt+10000, ypt+100000, label, ha='left')
        elif label == 'Guy Gavriel Kay':
            plt.text(xpt, ypt+100000, label, ha='right')
        else:
            plt.text(xpt, ypt+100000, label, ha='center')
plt.savefig('figures/usmap.png')



# Zoom to Europe
fig = plt.figure()
themap = Basemap(projection='merc', lat_0=54.5, lon_0=15.2,
                 resolution='l', area_thresh=10000.0,
                 llcrnrlon=-13.00, llcrnrlat=35.00,
                 urcrnrlon=40.00, urcrnrlat=60.00)

# themap.drawcoastlines()
themap.drawcountries()
themap.fillcontinents(color='gainsboro')
themap.drawmapboundary()

# themap.drawmeridians(np.arange(-10, 40, 10))
# themap.drawparallels(np.arange(35, 55, 5))

x, y = themap(full_data['lon'].values, full_data['lat'].values)
themap.plot(x, y, 'o', color='Green', markersize=10)

# Add labels
for label, xpt, ypt, lat in zip(full_data['author'], x, y, full_data['lat']):
    if lat < 51 or lat > 53:
        plt.text(xpt, ypt+80000, label, ha='center')
plt.savefig('figures/europemap.png')
