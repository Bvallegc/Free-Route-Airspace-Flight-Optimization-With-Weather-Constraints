import netCDF4 as nc
import numpy as np

# Open a NetCDF file
ds = nc.Dataset('/Users/beltran/Documents/University Year 4/Capstone Project/Airmap/datasets/2022-01-31.nc', 'r')

# Print out all the variables in the file
print(ds.variables.keys())

wind_u = ds.variables['wind_u'][:]
wind_v = ds.variables['wind_v'][:]

wind_u_units = ds.variables['wind_u'].units
wind_v_units = ds.variables['wind_v'].units

level = ds.variables['level'].units

temp = ds.variables['temperature'].units

print("Units of wind_u:", wind_u_units)
print("Units of wind_v:", wind_v_units)
print("Units of level:", level)
print("Units of temperature:", temp)

# Access the latitude, longitude, and level values
latitudes = ds.variables['lat'][:]
longitudes = ds.variables['lon'][:]
levels = ds.variables['level'][:]

# Find the index of the closest latitude, longitude, and level to the ones specified
lat_index = (np.abs(latitudes - 30)).argmin()
lon_index = (np.abs(longitudes - 65)).argmin()
level_index = (np.abs(levels - 500)).argmin()  # for example, if you want the level closest to 500

# Access the wind components at the specified coordinates
wind_u_value = wind_u[0, level_index, lat_index, lon_index]
wind_v_value = wind_v[0, level_index, lat_index, lon_index]

# Calculate the wind speed
wind_speed = np.sqrt(wind_u_value**2 + wind_v_value**2)

print("Wind speed at coordinates (30,60) and level 500:", wind_speed)

wind_direction = (270 - (np.arctan2(wind_v_value, wind_u_value) * (180/np.pi))) % 360

print("Wind direction at coordinates (30,60) and level 500:", wind_direction)

print("Shape of wind_u:", wind_u.shape)
print("Shape of wind_v:", wind_v.shape)
print("Shape of latitudes:", latitudes.shape)
print("Shape of longitudes:", longitudes.shape)
print("Shape of levels:", levels.shape)

ds.close()
