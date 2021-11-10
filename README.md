# Competition1-6390
IFT 6390-2021 Competition1
Autor: Abdiel Fernandez

PARA ELIER y Abdiel

	This competition is about detection of extreme weather events from atmospherical data. The goal is to automatically classify a set of climate variables corresponding to a time point and location, latitude and longitude, into one of three classes:

Standard background conditions
Tropical cyclone
Atmospheric river
Models that are capable of accurately detecting such events are crucial for our understanding of how they may evolve under various climate change scenarios.

The data set for this competition is a relatively small portion of a bigger data set, ClimateNet. The complete data set amounts to nearly 30 GB because it contains climate variables at almost 900,000 locations around the globe. For The subset for this competition contains just 120 locations, while keeping the data at all time points.

The training set contains 47,760 data points, from 1996 to 2010, and the test set contains 7,320 data points, from 2011 to 2013. Each data point consists of 16 atmospheric variables such as pressure, temperature and humidity, besides the latitude, longitude and time. The complete set of variables is the following:

lat: latitude

lon: longitude

TMQ: total (vertically integrated) precipitable water [kg/m^2]

U850: zonal wind at 850 mbar pressure surface [m/s]

V850: meridional wind at 850 mbar pressure surface [m/s]

UBOT: lowest level zonal wind [m/s]

VBOT: lowest model level meridional wind [m/s]

QREFHT: reference height humidity [kg/kg]

PSL: sea level pressure [Pa]

T200: temperature at 200 mbar pressure surface [K]

T500: temperature at 500 mbar pressure surface [K]

PRECT: total (convective and large-scale) precipitation rate (liq + ice) [m/s]

TS: surface temperature (radiative) [K]

TREFHT: reference height temperature [K]

Z1000: geopotential Z at 1000 mbar pressure surface [m]

Z200: geopotential Z at 200 mbar pressure surface [m]

ZBOT: lowest modal level height [m]

time [string]
