#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

epw_header = ['year', 'month', 'day', 'hour', 'minute', 'u_flag', 'Dry bulb T [C]', 'Dew point T [C]',
              'Rel Humidity [%]', 'Atm pressure [Pa]', 'GHIe [Wh/m2]', 'DNIe [Wh/m2]', 'Infrared horiz rad [Wh/m2]',
              'GHI [Wh/m2]', 'DNI [Wh/m2]', 'DHI [Wh/m2]', 'GHE [lx]', 'DNE [lx]', 'DHE [lx]', 'Zenith lum [cd/m2]',
              'Wind direction [deg]', 'Wind speed [m/s]', 'Total sky cover', 'Opaque sky cover', 'Visibility [km]',
              'Ceiling height [m]', 'Present wea observation', 'Present wea codes', 'Precipitable water [mm]',
              'Aerosol optical depth', 'Snow depth [cm]', 'Days since last snow', 'Albedo',
              'Liquid precipitation depth [mm]',
              'Liquid precipitation quantity [hr]']


def read_epw(fn, year=None, printcoord=False):
    """
    Reads an EPW format climate file and saves it as a DataFrame. Unless specified, the first year found in the file is assigned to all data.
    :param fn: file name string
    :return: pandas DataFrame with header and datetime index, [lat [+N], lon [+E], tz [+E], altitude]
    """
    with open(fn) as f:
        line = f.readline()
        print(line)

    lat = float(line.split(',')[6])
    lon = float(line.split(',')[7])
    tz = float(line.split(',')[8])
    alt = float(line.split(',')[9])

    # na_values = ['999999999', '999999999.0', 999999999, 999999999.0]
    epw = pd.read_csv(fn, skiprows=8, header=None, names=epw_header, na_values=[999999999])
    epw['hour'] = epw['hour'] - 1
    if year is not None:
        epw['year'] = year
    else:
        epw['year'] = epw['year'][0]
    epw.index = pd.to_datetime(epw[['year', 'month', 'day', 'hour']])
    epw = epw.iloc[:, 5:]

    if printcoord:
        return epw, lat, lon, tz, alt
    else:
        return epw
