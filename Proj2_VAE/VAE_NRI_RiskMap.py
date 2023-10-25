import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

column14 = pd.read_csv('Proj2_VAE/column14.csv')


# Path to the shapefile of US counties
shapefile_path = 'Proj2_VAE/cb_2018_us_county_5m/cb_2018_us_county_5m.shp'
# Read the shapefile into a GeoDataFrame
gdf_counties = gpd.read_file(shapefile_path)
gdf_counties['COUNTY'] = gdf_counties['NAME'].str.upper()

print(gdf_counties.columns)
print(column14.columns)

gdf_counties = gdf_counties.merge(column14, left_on='COUNTY', right_on='COUNTY', how='left')
print(gdf_counties.columns)
print(gdf_counties.head())

# Plot the data
# Create a figure and axis with specified size
fig, ax = plt.subplots()
# Plot the counties with color-coded RISK
gdf_counties.plot(column='RISK', cmap='viridis', linewidth=0.1, ax=ax, edgecolor='0.8')
# Customize the plot
ax.set_title('NRI Natural Disaster Risk Rating by County')
ax.axis('off')
# Add a colorbar with a smaller size
sm = plt.cm.ScalarMappable(cmap='viridis')
sm.set_array(gdf_counties['RISK'])
cbar = plt.colorbar(sm, ax=ax, fraction=0.02)  # Specify the ax argument
# Show and save the plot
plt.savefig('RISKMAP.png')
plt.show()


'''
# Plot the data
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the counties with color-coded RISK
gdf_counties.plot(column='RISK', cmap='viridis', linewidth=0.1, ax=ax, edgecolor='0.8')
# Customize the plot
ax.set_title('NRI Natural Disaster Risk Rating by County')
ax.axis('off')
# Adjust the aspect ratio of the plot to match the aspect ratio of the data
ax.set_aspect('equal')
# Add a colorbar with a smaller size
sm = plt.cm.ScalarMappable(cmap='viridis')
sm.set_array(gdf_counties['RISK'])
cbar = plt.colorbar(sm, ax=ax, fraction=0.02)  # Specify the ax argument
# Show and save the plot
plt.savefig('RISKMAP.png')
plt.show()
'''