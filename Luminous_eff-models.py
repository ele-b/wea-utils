#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import datetime
import pandas as pd
import numpy as np
from math import sin, cos, tan, asin, acos, atan, radians, degrees, pi, log, exp, atan2, floor, sqrt


# ===================== Solar position models =================================

def limit_360(angle):
    angle /= 360.
    limited = 360. * (angle - floor(angle))
    if limited < 0:
        limited += 360
    return limited


def AST_lowacc(lon, lsm, year, month, day, hour, mins=0, secs=0, v=False):
    """ Low accuracy calculation of solar geometry fundamentals

    Code taken from 'Solar radiation and daylight models' (Muneer, 2004).
    EOT: p.6 - AST and DEC: p.10 - SHA: Prog.1-6.For

    :param year: int format yyyy
    :param month: int format MM
    :param day: int format dd
    :param hour: int format hh 0-23 (local civil time)
    :param mins: int format mm
    :param lon: longitude in degrees [+W; -E]
    :param lsm: longitude of the standard time meridian
    :param v: verbose option, if set to True prints strings with results (default False)
    :return: eot (equation of time, in hours), dec (solar declination, in radians), ast (apparent solar time, in hours),
    sha (solar hour angle, in degrees)
    """

    dn = datetime.datetime(year, month, day).timetuple().tm_yday
    xdn = 2. * pi * (dn - 1) / 365.242
    eot = -0.1236 * sin(xdn) + 0.0043 * cos(xdn) - 0.1538 * sin(2 * xdn) - 0.0608 * cos(2 * xdn)
    dec = asin(0.39795 * cos(radians(0.98563 * (dn - 173))))
    ast = hour + (mins / 60.) + (secs / 3600.) + eot + ((lsm - lon) / 15)
    sha = (ast - 12) * 15
    if v:
        print('Equation of time: %.4f h (%.4f min)' % (eot, eot * 60))
        print('Solar declination: %.4f degrees (%.4f rad)' % (degrees(dec), dec))
        print('Apparent solar time: %.4f h' % ast)
        print('Solar hour angle: %.4f degrees' % sha, end='\n\n')
    return eot, dec, ast, sha


def AST_highacc(lon, lsm, year, month, day, hour, mins=0, secs=0, v=False):
    """ High accuracy calculation of solar geometry fundamentals

    Calculations based on Yallop (1992) and code taken from 'Solar radiation and daylight models' (Muneer, 2004), p.11 and Prog.1-6.For
    Valid for the period 1980-2050.

    :param year: int format yyyy
    :param month: int format MM
    :param day: int format dd
    :param hour: int format hh 0-23 (local civil time)
    :param mins: int format mm
    :param lon: longitude in degrees [+W; -E]
    :param lsm: longitude of the standard time meridian
    :param v: verbose option, if set to True prints strings with results (default False)
    :return: eot, dec, ast, sha
    Equation of time (in hours), Solar declination (in radians), Apparent solar time (in hours), Solar hour angle (in degrees)
    """

    xlct = hour + mins / 60. + secs / 3600.
    ut = xlct + lsm / 15.
    if month > 2:
        year1 = year
        month1 = month - 3
    else:
        year1 = year - 1
        month1 = month + 9
    int1 = int(30.6 * (month1) + 0.5)
    int2 = int(365.25 * (year1 - 1976))
    smlt = ((ut / 24) + day + int1 + int2 - 8707.5) / 36525.

    epsiln = 23.4393 - 0.013 * smlt
    capg = 357.528 + 35999.05 * smlt
    if capg > 360:
        g360 = capg - int(capg / 360.) * 360
    else:
        g360 = capg
    capc = 1.915 * sin(radians(g360)) + 0.020 * sin(radians(2 * g360))
    capl = 280.460 + 36000.770 * smlt + capc

    if capl > 360:
        xl360 = capl - int(capl / 360.) * 360
    else:
        xl360 = capl
    alpha = xl360 - 2.466 * sin(radians(2 * xl360)) + 0.053 * sin(radians(4 * xl360))
    eot = (xl360 - capc - alpha) / 15.
    gha = 15 * ut - 180 - capc + xl360 - alpha

    if gha > 360:
        gha360 = gha - int(gha / 360.) * 360
    else:
        gha360 = gha
    dec = atan(tan(radians(epsiln)) * sin(radians(alpha)))
    sha = gha360 - lon
    ast = 12 + (sha / 15.)
    if v:
        print('Equation of time: %.4f h (%.4f min)' % (eot, eot * 60))
        print('Solar declination: %.4f degrees (%.4f rad)' % (degrees(dec), dec))
        print('Apparent solar time: %.4f h' % ast)
        print('Solar hour angle: %.4f degrees' % sha, end='\n\n')
    return eot, dec, ast, sha


def solpos(lat, dec, sha, v=False):
    """ Estimates solar position (altitude and azimuth)

    Estimates solar position (altitude and azimuth), based on latitude, solar declination and solar hour angle.
    Code taken from 'Solar radiation and daylight models' (Muneer, 2004), p.13 and Prog.1-6.For

    :param lat: latitude, in degrees [+N; -S]
    :param dec: solar declination, in radians
    :param sha: solar hour angle, in degrees
    :param v: verbose option, if set to True prints strings with results (default False)
    :return: solalt, solazm
    solar altitude (in degrees) and azimuth (in degrees)
    """
    solalt1 = sin(radians(lat)) * sin(dec)
    solalt2 = cos(radians(lat)) * cos(dec) * cos(radians(sha + 180))
    solalt = asin(solalt1 - solalt2)
    solalt = degrees(solalt)
    # print(solalt)

    solazm1 = cos(radians(lat)) * tan(dec)
    solazm2 = sin(radians(lat)) * cos(radians(sha + 180))
    # solazm2 = sin(radians(lat)) * cos(radians(sha))
    solazm3 = cos(radians(solalt))
    solazm = acos((cos(dec) * (solazm1 + solazm2) / solazm3))
    solazm = degrees(solazm)
    # solazm = solazm + 180
    # solazm = limit_360(solazm)
    if sha > 0 or sha < -180:
        solazm = 360 - solazm
    if v:
        print('Solar altitude: %.4f degrees' % solalt)
        print('Solar azimuth: %.4f degrees' % solazm, end='\n\n')
    return solalt, solazm

    # solazm = -atan2((cos(sdec) * sin(stime * pi / 12.)),
    #                 (-cos(radians(lat)) * sin(sdec) - sin(radians(lat)) * cos(sdec) * cos(stime * pi / 12.)))
    # solazm = 180 + degrees(solazm)
    # solazm = limit_360(solazm)


L_0 = pd.DataFrame(data=[[175347046.0, 0., 0.], [3341656.0, 4.6692568, 6283.07585], [34894.0, 4.6261, 12566.1517],
                         [3497.0, 2.7441, 5753.3849], [3418.0, 2.8289, 3.5231], [3136.0, 3.6277, 77713.7715],
                         [2676.0, 4.4181, 7860.4194], [2343.0, 6.1352, 3930.2097], [1324.0, 0.7425, 11506.7698],
                         [1273.0, 2.0371, 529.691], [1199.0, 1.1096, 1577.3435], [990, 5.233, 5884.927],
                         [902, 2.045, 26.298], [857, 3.508, 398.149], [780, 1.179, 5223.694], [753, 2.533, 5507.553],
                         [505, 4.583, 18849.228], [492, 4.205, 775.523], [357, 2.92, 0.067], [317, 5.849, 11790.629],
                         [284, 1.899, 796.298], [271, 0.315, 10977.079], [243, 0.345, 5486.778],
                         [206, 4.806, 2544.314], [205, 1.869, 5573.143], [202, 2.458, 6069.777], [156, 0.833, 213.299],
                         [132, 3.411, 2942.463], [126, 1.083, 20.775], [115, 0.645, 0.98], [103, 0.636, 4694.003],
                         [102, 0.976, 15720.839], [102, 4.267, 7.114], [99, 6.21, 2146.17], [98, 0.68, 155.42],
                         [86, 5.98, 161000.69], [85, 1.3, 6275.96], [85, 3.67, 71430.7], [80, 1.81, 17260.15],
                         [79, 3.04, 12036.46], [75, 1.76, 5088.63], [74, 3.5, 3154.69], [74, 4.68, 801.82],
                         [70, 0.83, 9437.76], [62, 3.98, 8827.39], [61, 1.82, 7084.9], [57, 2.78, 6286.6],
                         [56, 4.39, 14143.5], [56, 3.47, 6279.55], [52, 0.19, 12139.55], [52, 1.33, 1748.02],
                         [51, 0.28, 5856.48], [49, 0.49, 1194.45], [41, 5.37, 8429.24], [41, 2.4, 19651.05],
                         [39, 6.17, 10447.39], [37, 6.04, 10213.29], [37, 2.57, 1059.38], [36, 1.71, 2352.87],
                         [36, 1.78, 6812.77], [33, 0.59, 17789.85], [30, 0.44, 83996.85], [30, 2.74, 1349.87],
                         [25, 3.16, 4690.48]], columns=['A', 'B', 'C'])

L_1 = pd.DataFrame(data=[[628331966747.0, 0, 0],
                         [206059.0, 2.678235, 6283.07585],
                         [4303.0, 2.6351, 12566.1517],
                         [425.0, 1.59, 3.523],
                         [119.0, 5.796, 26.298],
                         [109.0, 2.966, 1577.344],
                         [93, 2.59, 18849.23],
                         [72, 1.14, 529.69],
                         [68, 1.87, 398.15],
                         [67, 4.41, 5507.55],
                         [59, 2.89, 5223.69],
                         [56, 2.17, 155.42],
                         [45, 0.4, 796.3],
                         [36, 0.47, 775.52],
                         [29, 2.65, 7.11],
                         [21, 5.34, 0.98],
                         [19, 1.85, 5486.78],
                         [19, 4.97, 213.3],
                         [17, 2.99, 6275.96],
                         [16, 0.03, 2544.31],
                         [16, 1.43, 2146.17],
                         [15, 1.21, 10977.08],
                         [12, 2.83, 1748.02],
                         [12, 3.26, 5088.63],
                         [12, 5.27, 1194.45],
                         [12, 2.08, 4694],
                         [11, 0.77, 553.57],
                         [10, 1.3, 6286.6],
                         [10, 4.24, 1349.87],
                         [9, 2.7, 242.73],
                         [9, 5.64, 951.72],
                         [8, 5.3, 2352.87],
                         [6, 2.65, 9437.76],
                         [6, 4.67, 4690.48]], columns=['A', 'B', 'C'])

L_2 = pd.DataFrame(data=[[52919.0, 0, 0],
                         [8720.0, 1.0721, 6283.0758],
                         [309.0, 0.867, 12566.152],
                         [27, 0.05, 3.52],
                         [16, 5.19, 26.3],
                         [16, 3.68, 155.42],
                         [10, 0.76, 18849.23],
                         [9, 2.06, 77713.77],
                         [7, 0.83, 775.52],
                         [5, 4.66, 1577.34],
                         [4, 1.03, 7.11],
                         [4, 3.44, 5573.14],
                         [3, 5.14, 796.3],
                         [3, 6.05, 5507.55],
                         [3, 1.19, 242.73],
                         [3, 6.12, 529.69],
                         [3, 0.31, 398.15],
                         [3, 2.28, 553.57],
                         [2, 4.38, 5223.69],
                         [2, 3.75, 0.98]], columns=['A', 'B', 'C'])

L_3 = pd.DataFrame(data=[[289.0, 5.844, 6283.076],
                         [35, 0, 0],
                         [17, 5.49, 12566.15],
                         [3, 5.2, 155.42],
                         [1, 4.72, 3.52],
                         [1, 5.3, 18849.23],
                         [1, 5.97, 242.73]], columns=['A', 'B', 'C'])

L_4 = pd.DataFrame(data=[[114.0, 3.142, 0],
                         [8, 4.13, 6283.08],
                         [1, 3.84, 12566.15]], columns=['A', 'B', 'C'])

L_5 = pd.DataFrame(data=[[1, 3.14, 0]], columns=['A', 'B', 'C'])

B_0 = pd.DataFrame(data=[[280.0, 3.199, 84334.662],
                         [102.0, 5.422, 5507.553],
                         [80, 3.88, 5223.69],
                         [44, 3.7, 2352.87],
                         [32, 4, 1577.34]], columns=['A', 'B', 'C'])

B_1 = pd.DataFrame(data=[[9, 3.9, 5507.55], [6, 1.73, 5223.69]], columns=['A', 'B', 'C'])

R_0 = pd.DataFrame(data=[[100013989.0, 0, 0],
                         [1670700.0, 3.0984635, 6283.07585],
                         [13956.0, 3.05525, 12566.1517],
                         [3084.0, 5.1985, 77713.7715],
                         [1628.0, 1.1739, 5753.3849],
                         [1576.0, 2.8469, 7860.4194],
                         [925.0, 5.453, 11506.77],
                         [542.0, 4.564, 3930.21],
                         [472.0, 3.661, 5884.927],
                         [346.0, 0.964, 5507.553],
                         [329.0, 5.9, 5223.694],
                         [307.0, 0.299, 5573.143],
                         [243.0, 4.273, 11790.629],
                         [212.0, 5.847, 1577.344],
                         [186.0, 5.022, 10977.079],
                         [175.0, 3.012, 18849.228],
                         [110.0, 5.055, 5486.778],
                         [98, 0.89, 6069.78],
                         [86, 5.69, 15720.84],
                         [86, 1.27, 161000.69],
                         [65, 0.27, 17260.15],
                         [63, 0.92, 529.69],
                         [57, 2.01, 83996.85],
                         [56, 5.24, 71430.7],
                         [49, 3.25, 2544.31],
                         [47, 2.58, 775.52],
                         [45, 5.54, 9437.76],
                         [43, 6.01, 6275.96],
                         [39, 5.36, 4694],
                         [38, 2.39, 8827.39],
                         [37, 0.83, 19651.05],
                         [37, 4.9, 12139.55],
                         [36, 1.67, 12036.46],
                         [35, 1.84, 2942.46],
                         [33, 0.24, 7084.9],
                         [32, 0.18, 5088.63],
                         [32, 1.78, 398.15],
                         [28, 1.21, 6286.6],
                         [28, 1.9, 6279.55],
                         [26, 4.59, 10447.39]], columns=['A', 'B', 'C'])

R_1 = pd.DataFrame(data=[[103019.0, 1.10749, 6283.07585],
                         [1721.0, 1.0644, 12566.1517],
                         [702.0, 3.142, 0],
                         [32, 1.02, 18849.23],
                         [31, 2.84, 5507.55],
                         [25, 1.32, 5223.69],
                         [18, 1.42, 1577.34],
                         [10, 5.91, 10977.08],
                         [9, 1.42, 6275.96],
                         [9, 0.27, 5486.78]], columns=['A', 'B', 'C'])

R_2 = pd.DataFrame(data=[[4359.0, 5.7846, 6283.0758],
                         [124.0, 5.579, 12566.152],
                         [12, 3.14, 0],
                         [9, 3.63, 77713.77],
                         [6, 1.87, 5573.14],
                         [3, 5.47, 18849.23]], columns=['A', 'B', 'C'])

R_3 = pd.DataFrame(data=[[145.0, 4.273, 6283.076], [7, 3.92, 12566.15]], columns=['A', 'B', 'C'])

R_4 = pd.DataFrame(data=[[4, 2.56, 6283.08]], columns=['A', 'B', 'C'])

Y = pd.DataFrame(data=[[0, 0, 0, 0, 1],
                       [-2, 0, 0, 2, 2],
                       [0, 0, 0, 2, 2],
                       [0, 0, 0, 0, 2],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [-2, 1, 0, 2, 2],
                       [0, 0, 0, 2, 1],
                       [0, 0, 1, 2, 2],
                       [-2, -1, 0, 2, 2],
                       [-2, 0, 1, 0, 0],
                       [-2, 0, 0, 2, 1],
                       [0, 0, -1, 2, 2],
                       [2, 0, 0, 0, 0],
                       [0, 0, 1, 0, 1],
                       [2, 0, -1, 2, 2],
                       [0, 0, -1, 0, 1],
                       [0, 0, 1, 2, 1],
                       [-2, 0, 2, 0, 0],
                       [0, 0, -2, 2, 1],
                       [2, 0, 0, 2, 2],
                       [0, 0, 2, 2, 2],
                       [0, 0, 2, 0, 0],
                       [-2, 0, 1, 2, 2],
                       [0, 0, 0, 2, 0],
                       [-2, 0, 0, 2, 0],
                       [0, 0, -1, 2, 1],
                       [0, 2, 0, 0, 0],
                       [2, 0, -1, 0, 1],
                       [-2, 2, 0, 2, 2],
                       [0, 1, 0, 0, 1],
                       [-2, 0, 1, 0, 1],
                       [0, -1, 0, 0, 1],
                       [0, 0, 2, -2, 0],
                       [2, 0, -1, 2, 1],
                       [2, 0, 1, 2, 2],
                       [0, 1, 0, 2, 2],
                       [-2, 1, 1, 0, 0],
                       [0, -1, 0, 2, 2],
                       [2, 0, 0, 2, 1],
                       [2, 0, 1, 0, 0],
                       [-2, 0, 2, 2, 2],
                       [-2, 0, 1, 2, 1],
                       [2, 0, -2, 0, 1],
                       [2, 0, 0, 0, 1],
                       [0, -1, 1, 0, 0],
                       [-2, -1, 0, 2, 1],
                       [-2, 0, 0, 0, 1],
                       [0, 0, 2, 2, 1],
                       [-2, 0, 2, 0, 1],
                       [-2, 1, 0, 2, 1],
                       [0, 0, 1, -2, 0],
                       [-1, 0, 1, 0, 0],
                       [-2, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 1, 2, 0],
                       [0, 0, -2, 2, 2],
                       [-1, -1, 1, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, -1, 1, 2, 2],
                       [2, -1, -1, 2, 2],
                       [0, 0, 3, 2, 2],
                       [2, -1, 0, 2, 2]], dtype=int, columns=['Y0', 'Y1', 'Y2', 'Y3', 'Y4'])

delta_psi_eps = pd.DataFrame(data=[[-171996, -174.2, 92025, 8.9],
                                   [-13187, -1.6, 5736, -3.1],
                                   [-2274, -0.2, 977, -0.5],
                                   [2062, 0.2, -895, 0.5],
                                   [1426, -3.4, 54, -0.1],
                                   [712, 0.1, -7, 0],
                                   [-517, 1.2, 224, -0.6],
                                   [-386, -0.4, 200, 0],
                                   [-301, 0, 129, -0.1],
                                   [217, -0.5, -95, 0.3],
                                   [-158, 0, 0, 0],
                                   [129, 0.1, -70, 0],
                                   [123, 0, -53, 0],
                                   [63, 0, 0, 0],
                                   [63, 0.1, -33, 0],
                                   [-59, 0, 26, 0],
                                   [-58, -0.1, 32, 0],
                                   [-51, 0, 27, 0],
                                   [48, 0, 0, 0],
                                   [46, 0, -24, 0],
                                   [-38, 0, 16, 0],
                                   [-31, 0, 13, 0],
                                   [29, 0, 0, 0],
                                   [29, 0, -12, 0],
                                   [26, 0, 0, 0],
                                   [-22, 0, 0, 0],
                                   [21, 0, -10, 0],
                                   [17, -0.1, 0, 0],
                                   [16, 0, -8, 0],
                                   [-16, 0.1, 7, 0],
                                   [-15, 0, 9, 0],
                                   [-13, 0, 7, 0],
                                   [-12, 0, 6, 0],
                                   [11, 0, 0, 0],
                                   [-10, 0, 5, 0],
                                   [-8, 0, 3, 0],
                                   [7, 0, -3, 0],
                                   [-7, 0, 0, 0],
                                   [-7, 0, 3, 0],
                                   [-7, 0, 3, 0],
                                   [6, 0, 0, 0],
                                   [6, 0, -3, 0],
                                   [6, 0, -3, 0],
                                   [-6, 0, 3, 0],
                                   [-6, 0, 3, 0],
                                   [5, 0, 0, 0],
                                   [-5, 0, 3, 0],
                                   [-5, 0, 3, 0],
                                   [-5, 0, 3, 0],
                                   [4, 0, 0, 0],
                                   [4, 0, 0, 0],
                                   [4, 0, 0, 0],
                                   [-4, 0, 0, 0],
                                   [-4, 0, 0, 0],
                                   [-4, 0, 0, 0],
                                   [3, 0, 0, 0],
                                   [-3, 0, 0, 0],
                                   [-3, 0, 0, 0],
                                   [-3, 0, 0, 0],
                                   [-3, 0, 0, 0],
                                   [-3, 0, 0, 0],
                                   [-3, 0, 0, 0],
                                   [-3, 0, 0, 0]], columns=['a', 'b', 'c', 'd'])


def helio_coord(table, jme):
    vector = []
    for i in range(len(table)):
        v_i = table['A'][i] * np.cos(table['B'][i] + table['C'][i] * jme)
        vector.append(v_i)
    v_sum = np.sum(np.array(vector))
    return v_sum


def nutation(X, jce):
    delta_psi = []
    delta_eps = []
    for i in range(len(Y)):
        xy_sum = 0
        for j in range(4):
            xy_sum = xy_sum + X[j] * Y.iloc[i, j]
        delta_psi_i = (delta_psi_eps['a'][i] + delta_psi_eps['b'][i] * jce) * sin(xy_sum)
        delta_eps_i = (delta_psi_eps['c'][i] + delta_psi_eps['d'][i] * jce) * cos(xy_sum)
        delta_psi.append(delta_psi_i)
        delta_eps.append(delta_eps_i)
    delta_psi_sum = np.sum(np.array(delta_psi)) / 36000000.
    delta_eps_sum = np.sum(np.array(delta_eps)) / 36000000.
    return delta_psi_sum, delta_eps_sum


def SPA(lat, lon, lsm, alt, P, T, year, month, day, hour, mins=0, secs=0):
    """ Estimates solar position (altitude and azimuth)

    Estimates solar position (altitude and azimuth), based on the Solar Position Algorithm (Reda and Andreas 2004)
    Copied from C code obtained from http://midcdmz.nrel.gov/spa/ [20/04/2018]

    :param year: int format yyyy
    :param month: int format MM
    :param day: int format dd
    :param hour: int format hh 0-23 (local civil time)
    :param mins: int format mm
    :param lon: longitude in degrees [+W; -E]
    :param lsm: longitude of the standard time meridian
    :param alt: elevation (m)
    :param P: annual average local pressure (mbar)
    :param T: annual average local temperature (degrees celsius)
    :param v: verbose option, if set to True prints strings with results (default False)
    :return: solalt, solazm
    solar altitude (in degrees) and azimuth (in degrees)
    """
    delta_t = 70  # s - this value changes every year (http://asa.usno.navy.mil/SecK/DeltaT.html)

    d = day + hour / 24 + mins / 60 / 24 + secs / 60 / 60 / 24
    b = 2 - int(year / 100) + int(int(year / 100) / 4)  # only Gregorian calendar considered in this code
    lct = lsm / 15. / 24.
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + d + b - 1524.5 - lct

    jde = jd + delta_t / 86400.
    jc = (jd - 2451545) / 36525.
    jce = (jde - 2451545) / 36525.
    jme = jce / 10.

    # Calculate the Earth heliocentric longitude, latitude, and radius vector (L, B, and R)
    L_0_sum = helio_coord(L_0, jme)
    L_1_sum = helio_coord(L_1, jme)
    L_2_sum = helio_coord(L_2, jme)
    L_3_sum = helio_coord(L_3, jme)
    L_4_sum = helio_coord(L_4, jme)
    L_5_sum = helio_coord(L_5, jme)

    L = (L_0_sum + L_1_sum * jme + L_2_sum * jme ** 2 + L_3_sum * jme ** 3 +
         L_4_sum * jme ** 4 + L_5_sum * jme ** 5) / 1e9
    L = degrees(L)
    L = limit_360(L)

    B_0_sum = helio_coord(B_0, jme)
    B_1_sum = helio_coord(B_1, jme)

    B = (B_0_sum + B_1_sum * jme) / 1e9
    B = degrees(B)
    B = limit_360(B)

    R_0_sum = helio_coord(R_0, jme)
    R_1_sum = helio_coord(R_1, jme)
    R_2_sum = helio_coord(R_2, jme)
    R_3_sum = helio_coord(R_3, jme)
    R_4_sum = helio_coord(R_4, jme)

    R = (R_0_sum + R_1_sum * jme + R_2_sum * jme ** 2 + R_3_sum * jme ** 3 + R_4_sum * jme ** 4) / 1e9
    R = degrees(R)
    R = limit_360(R)

    # Calculate the geocentric longitude and latitude (theta and beta)
    theta = L + 180
    theta = limit_360(theta)

    beta = -B

    # Calculate the nutation in longitude and obliquity (delta_psi and delta_eps)
    X_0 = 297.85036 + 445267.111480 * jce - 0.0019142 * jce ** 2 + (jce ** 3) / 189474.
    X_1 = 357.52772 + 35999.050340 * jce - 0.0001603 * jce ** 2 - (jce ** 3) / 300000.
    X_2 = 134.96298 + 477198.867398 * jce + 0.0086972 * jce ** 2 + (jce ** 3) / 56250.
    X_3 = 93.27191 + 483202.017538 * jce - 0.0036825 * jce ** 2 + (jce ** 3) / 327270.
    X_4 = 125.04452 - 1934.136261 * jce + 0.0020078 * jce ** 2 + (jce ** 3) / 450000.
    X = [X_0, X_1, X_2, X_3, X_4]

    delta_psi, delta_eps = nutation(X, jce)

    # Calculate the true obliquity of the ecliptic, eps (in degrees)
    U = jme / 10.
    eps_0 = 84381.448 - 4680.93 * U - 1.55 * U ** 2 + 1999.25 * U ** 3 - 51.38 * U ** 4 - \
            249.67 * U ** 5 - 39.05 * U ** 6 + 7.12 * U ** 7 + 27.87 * U ** 8 + 5.79 * U ** 9 + 2.45 * U ** 10
    eps = (eps_0 / 3600.) + delta_eps

    delta_tau = -(20.4898 / (3600 * R))

    lam = theta + delta_psi + delta_tau

    # Calculate the apparent sidereal time at Greenwich at any given time, nu (in degrees)
    nu_0 = 280.46061837 + 360.98564736629 * (jd - 2451545) + 0.000387933 * jc ** 2 - (jc ** 3) / 38710000.
    nu_0 = limit_360(nu_0)
    nu = nu_0 + delta_psi * cos(eps)

    # Calculate the geocentric sun right ascension, alpha (in degrees)
    alpha = atan2((sin(lam) * cos(eps) - tan(beta) * sin(eps)), cos(lam))
    alpha = degrees(alpha)
    alpha = limit_360(alpha)

    # Calculate the geocentric sun declination, delta (in degrees)
    delta = asin(sin(beta) * cos(eps) + cos(beta) * sin(eps) * sin(lam))
    delta = degrees(delta)

    # Calculate the observer local hour angle, H (in degrees)
    H = nu + lon - alpha
    H = limit_360(H)

    # Calculate the topocentric sun right ascension alpha' (in degrees)
    xi = 8.794 / (3600 * R)
    u = atan(0.99664719 * tan(lat))
    x = cos(u) + alt * cos(lat) / 6378140.
    y = 0.99664719 * sin(u) + alt * sin(lat) / 6378140.

    delta_alpha = atan2((-x * sin(xi) * sin(H)), (cos(delta) - x * sin(xi) * cos(H)))
    delta_alpha = degrees(delta_alpha)
    alpha_1 = alpha + delta_alpha

    delta_1 = atan2(((sin(delta) - y * sin(xi)) * cos(delta_alpha)), (cos(delta) - y * sin(xi) * cos(H)))

    H_1 = H - delta_alpha

    # Calculate the topocentric zenith angle (in degrees)

    e_0 = asin(sin(lat) * sin(delta_1) + cos(lat) * cos(delta_1) * cos(H_1))
    e_0 = degrees(e_0)

    delta_e = P / 1010. * 283 / (273 + T) * 1.02 / (60 * tan(e_0 + 10.3 / (e_0 + 5.11)))

    solalt = e_0 + delta_e
    zenith = 90 - solalt

    # Calculate the topocentric azimuth angle (in degrees - eastward from North)
    gamma = atan2((sin(H_1)), (cos(H_1) * sin(lat) - tan(delta_1) * cos(lat)))
    gamma = degrees(gamma)
    gamma = limit_360(gamma)

    solazm = gamma + 180
    solazm = limit_360(solazm)

    return solalt, solazm


def epsranges(eps):
    if 1. <= eps <= 1.065:
        return 0
    if 1.065 < eps <= 1.230:
        return 1
    if 1.230 < eps <= 1.5:
        return 2
    if 1.5 < eps <= 1.95:
        return 3
    if 1.95 < eps <= 2.8:
        return 4
    if 2.8 < eps <= 4.5:
        return 5
    if 4.5 < eps <= 6.2:
        return 6
    if eps > 6.2:
        return 7


# DATA GCOEFS
a_g = [96.6251298, 107.5371163, 98.7277012, 92.7209667, 86.7266487, 88.3516199, 78.6239624, 99.6451808]
b_g = [-0.4702617, 0.7866356, 0.6971843, 0.5590546, 0.9762500, 1.3891286, 1.4699236, 1.8569185]
c_g = [11.5009654, 1.7899069, 4.4045790, 8.3578727, 7.1032884, 6.0641078, 4.9305411, -4.4554996]
d_g = [-9.1555100, -1.1892307, -6.9483420, -8.3062845, -10.9361060, -7.5966858, -11.3703391, -3.1464722]
# DATA BCOEFS
a_b = [57.20, 98.99, 109.83, 110.34, 106.36, 107.19, 105.75, 101.18]
b_b = [-4.55, -3.46, -4.90, -5.84, -3.97, -1.25, 0.77, 1.58]
c_b = [-2.98, -1.21, -1.71, -1.99, -1.75, -1.51, -1.26, -1.10]
d_b = [117.12, 12.38, -8.81, -4.56, -6.16, -26.73, -34.44, -8.29]
# DATA DCOEFS
a_d = [97.2375084, 107.2129343, 104.9660402, 102.3944705, 100.71, 106.42, 141.88, 152.23]
b_d = [-0.4597402, 1.1508313, 2.9604528, 5.5889779, 5.94, 3.83, 1.90, 0.35]
c_d = [11.9962113, 0.5840355, -5.5334007, -13.9510061, -22.75, -36.15, -53.24, -45.27]
d_d = [-8.9149195, -3.9489905, -8.7792705, -13.9051830, -23.74, -28.83, -14.03, -7.98]
# DATA ZCOEFS
a_z = [40.8646191, 26.5790229, 19.3462484, 13.2424953, 14.4716310, 19.7664897, 28.3923317, 42.9198246]
b_z = [26.7765737, 14.7297831, 2.2894950, -1.3986604, -5.0931879, -3.8842762, -9.6633967, -19.6247400]
c_z = [-29.5863114, 58.4661858, 100.0028947, 124.7992123, 160.0932252, 154.6061497, 151.5770258, 130.8071959]
d_z = [-45.7561792, -21.2447353, 0.2546900, 15.6529071, 9.1254758, -19.2028401, -69.3940780, -164.0794043]


# =========== Luminous efficacy models =========================================

def irr2ill_P90(ghi, dhi, solalt, dpt=10.9735, alt=0, v=False):
    """ Derive illuminance from irradiance using the Perez (1990) model

    Code taken from 'Solar radiation and daylight models' (Muneer, 2004), p.108 and Prog.3-4.For

    :param ghi: global horizontal irradiance in W/m2
    :param dhi: diffuse horizontal irradiance in W/m2
    :param solalt: solar altitude in degrees
    :param dpt: dew point temperature [C]
    :param alt: site elevation in m above sea level, if known (default is 0)
    :param v: verbose option, if set to True prints strings with results (default False)
    :return: ghe, dne, dhe, zlum
    global horizontal illuminance (klx), direct normal illuminance (klx), direct horizontal illuminance (klx), zenith luminance (cd/m2)
    """

    if np.isnan(ghi) or np.isnan(dhi):
        ghe = np.nan
        dne = np.nan
        dhe = np.nan
        zlum = np.nan
        return ghe, dne, dhe, zlum

    if solalt <= 0:
        dhe = 0
        ghe = 0
        dne = 0
        zlum = 0
    elif solalt <= 2.5:
        dhe = dhi * 120
        ghe = dhe
        dne = 0
        zlum = 0
        if v:
            print('Low solar angle, fixed luminous efficacy for ghi and dhi of 120 lm/W')
    elif solalt > 2.5 and (ghi <= 0 or dhi <= 0):
        ghe = np.nan
        dne = np.nan
        dhe = np.nan
        zlum = np.nan
        if v:
            print('GHI or DHI = 0 when solar altitude > 2.5, output saved with error code NaN')
    else:
        trm11 = sin(radians(solalt))
        b2 = 1.041
        dni = (ghi - dhi) / trm11
        zenith = 90 - solalt
        z = radians(zenith)
        airmass = 1 / (trm11 + 0.15 * (93.9 - zenith) ** (-1.253))

        # reduce atmospheric pressure for known altitudes
        p = exp(-0.0001184 * alt)
        airmass *= p

        # Perez coefficients: sky clearness (eps) and sky brightness (delta)

        skyclearinf = 1.0  # limitations for the variation of the Perez parameters
        skyclearsup = 12.01
        skybriginf = 0.01
        skybrigsup = 0.6

        delta = dhi * airmass / 1367.
        if delta < skybriginf:
            delta = skybriginf
        if delta > skybrigsup:
            delta = skybrigsup
        # print(delta)

        ldel = log(delta)
        t = z ** 3
        eps1 = (dni + dhi) / dhi
        eps = (eps1 + t * b2) / (1 + t * b2)

        if eps < skyclearinf:
            eps = skyclearinf
        if eps > skyclearsup:
            eps = skyclearsup
        # print(eps)

        epsbin = epsranges(eps)

        cz = cos(z)
        z2 = exp(5.729 * z - 5)
        z3 = exp(-3 * z)
        if epsbin > 0:
            lcz = cz
            lz3 = z3
        else:
            if cz > 0:
                lcz = min(0.825, cz)
            else:
                lcz = 0
            if z3 > 0.00899:
                lz3 = min(0.165, z3)
            else:
                lz3 = 0.00899

        # atmospheric precipitable water content lw
        if not np.isnan(dpt):
            lw = exp(0.07 * dpt - 0.075)
        else:
            lw = 2
        # a constant value of 2 cm does not influence the Perez model (Muneer 2004, p.114)
        # lw = 2

        zeff = a_z[epsbin] + b_z[epsbin] * lcz + c_z[epsbin] * lz3 + d_z[epsbin] * delta
        geff = a_g[epsbin] + b_g[epsbin] * lw + c_g[epsbin] * cz + d_g[epsbin] * ldel
        beff = a_b[epsbin] + b_b[epsbin] * lw + c_b[epsbin] * z2 + d_b[epsbin] * delta
        deff = a_d[epsbin] + b_d[epsbin] * lw + c_d[epsbin] * cz + d_d[epsbin] * ldel

        ghe = geff * ghi
        dne = max(0, beff * dni)
        dhe = deff * dhi
        zlum = zeff * dhi

        if v:
            print('Global horizontal illuminance: %.2f lx' % ghe)
            print('Direct normal illuminance: %.2f lx' % dne)
            print('Diffuse horizontal illuminance: %.2f lx' % dhe)
            print('Zenith luminance: %.2f cd/m2' % zlum)

    return ghe, dne, dhe, zlum


def irr2ill_KM98(ghi, dhi, solalt, year, month, day, v=False):
    """ Derive illuminance from irradiance using the Kinghorn-Muneer (1998) model

    Formulae taken from Annex to CIBSE Guide A Chapter 2, p.52-55

    :param ghi: global horizontal irradiance (W/m2)
    :param dhi: diffuse horizontal irradiance (W/m2)
    :param solalt: solar altitude (degrees)
    :param year: format yyyy
    :param month: format MM
    :param day: format dd
    :param v: verbose option, if set to True prints strings with results (default False)
    :return: ghe, dne, dhe
    global horizontal illuminance (klx), direct normal illuminance (klx), direct horizontal illuminance (klx)
    """

    if solalt <= 2.5:
        if ghi < 0:
            ghe = -999
            dne = -999
            dhe = -999
            if v:
                print('Error in input values, output saved with error code -999')
        else:
            dhe = dhi * 120
            # TODO why is ghe equal to ghi??
            # ghe = ghi
            ghe = dhe
            dne = 0
            if v:
                print('Low solar angle, fixed luminous efficacy for dhi of 120 lm/W')
    else:
        dn = datetime.datetime(year, month, day).timetuple().tm_yday
        j = dn * 360 / 365.242
        msd = 1 + 0.03344 * cos(j - 2.8)
        kth = ghi / (msd * 1367 * sin(radians(solalt)))
        geff = 136.6 - 74.54 * kth + 57.34 * kth ** 2
        deff = 130.2 - 39.82 * kth + 49.97 * kth ** 2

        ghe = geff * ghi
        dhe = deff * dhi
        dne = (ghe - dhe) / sin(radians(solalt))
        if v:
            print('Global horizontal illuminance: %.2f lx' % ghe)
            print('Direct normal illuminance: %.2f lx' % dne)
            print('Diffuse horizontal illuminance: %.2f lx' % dhe)

    return ghe, dne, dhe
