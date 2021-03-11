#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import datetime
import pandas as pd
import numpy as np
from math import sin, cos, tan, asin, acos, atan, radians, degrees, pi, log, exp, atan2, floor, sqrt


def ih_extraterr(solalt, dn):
    Iex = 1367.
    if solalt > 0:
        Ieh = Iex * (1 + 0.033 * cos(0.0172024 * dn)) * sin(radians(solalt))
    else:
        Ieh = 0
    return Ieh


# Separation models ============================================================================================

def Erbs82(ghi, solalt, dn):
    """
    Separation model to obtain DNI and DHI from GHI, based on Erbs et al. (1982)

    Code taken from Radiance gendaylit.c (W. Sprenger, 2013)

    :param ghi: global horizontal irradiance [W/m2]
    :param solalt: solar altitude in degrees
    :param dn: day of the year
    :return: GHI, DNI, DHI
    """
    Ieh = ih_extraterr(solalt, dn)

    if solalt <= 0:
        ghi = 0
        dni = 0
        dhi = 0
        return ghi, dni, dhi

    if ghi is np.nan:
        ghi = np.nan
        dni = np.nan
        dhi = np.nan
        return ghi, dni, dhi

    if ghi > Ieh:
        ghi = Ieh * 0.999

    kt = ghi / Ieh

    if kt <= 0.22:
        dhi = ghi * (1 - 0.09 * kt)
    elif kt <= 0.8:
        dhi = ghi * (0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4)
    else:
        dhi = ghi * 0.165

    dni = (ghi - dhi) / sin(radians(solalt))

    return ghi, dni, dhi


def Reindl90(ghi, t, rh, solalt, dn):
    """ Separation model to obtain DNI and DHI from GHI, based on Reindl et al. (1990)

    Formulae taken from 'Solar radiation and daylight models' (Muneer, 2004), p.103

    :param ghi: global horizontal irradiance [W/m2]
    :param t: dry bulb temperature [C]
    :param rh: relative humidity [%]
    :param solalt: solar altitude in degrees
    :param dn: day of the year
    :return: GHI, DNI, DHI
    """

    rh_frac = rh / 100.

    Ieh = ih_extraterr(solalt, dn)

    if solalt <= 0:
        ghi = 0
        dni = 0
        dhi = 0
        return ghi, dni, dhi

    if ghi is np.nan:
        ghi = np.nan
        dni = np.nan
        dhi = np.nan
        return ghi, dni, dhi

    if ghi > Ieh:
        ghi = Ieh * 0.999

    kt = ghi / Ieh

    if kt <= 0.3:
        dhi = ghi * (1 - 0.232 * kt + 0.0239 * sin(radians(solalt)) - 0.000682 * t + 0.0195 * rh_frac)
    elif kt < 0.78:
        dhi = ghi * (1.329 - 1.716 * kt + 0.2670 * sin(radians(solalt)) - 0.003570 * t + 0.1060 * rh_frac)
    else:
        dhi = ghi * (0.426 * kt - 0.2560 * sin(radians(solalt)) + 0.00349 * t + 0.0734 * rh_frac)

    dni = (ghi - dhi) / sin(radians(solalt))

    return ghi, dni, dhi


def CIBSEJ02(ghi, solalt, dn):
    """ Separation model to obtain DNI and DHI from GHI, based on CIBSE Guide J (2002)

    :param ghi: global horizontal irradiance [W/m2]
    :param solalt: solar altitude in degrees
    :param dn: day of the year
    :return: GHI, DNI, DHI
    """

    Ieh = ih_extraterr(solalt, dn)

    if solalt <= 5:
        ghi = 0
        dni = 0
        dhi = 0
        return ghi, dni, dhi

    if ghi is np.nan:
        ghi = np.nan
        dni = np.nan
        dhi = np.nan
        return ghi, dni, dhi

    if ghi > Ieh:
        ghi = Ieh * 0.999

    kt = ghi / Ieh

    if kt > 0.2:
        dhi = ghi * (0.687 + 2.932 * kt - 8.546 * kt ** 2 + 5.227 * kt ** 3)
    else:
        dhi = ghi * (0.98 * kt)

    dni = (ghi - dhi) / sin(radians(solalt))

    return ghi, dni, dhi


def Skartveit86(ghi, solalt, dn):
    """ Separation model to obtain DNI and DHI from GHI, based on Skartveit and Olseth (1986)

    :param ghi: global horizontal irradiance [W/m2]
    :param solalt: solar altitude in degrees
    :param dn: day of the year
    :return: GHI, DNI, DHI
    """

    Ieh = ih_extraterr(solalt, dn)

    if solalt <= 2:
        ghi = 0
        dni = 0
        dhi = 0
        return ghi, dni, dhi

    if ghi is np.nan:
        ghi = np.nan
        dni = np.nan
        dhi = np.nan
        return ghi, dni, dhi

    if ghi > Ieh:
        ghi = Ieh * 0.999

    kt = ghi / Ieh
    k0 = 0.2
    k1 = 0.87 - 0.56 * exp(-0.06 * solalt)
    alpha = 1.09
    d1 = 0.15 + 0.43 * exp(-0.06 * solalt)
    a = 0.27

    if kt < k0:
        dni = 0
    elif kt <= (alpha * k1):
        K = 0.5 * (1 + sin(pi * (((kt - k0) / (k1 - k0)) - 0.5)))
        phi = 1 - (1 - d1) * (a * sqrt(K) + (1 - a) * K ** 2)
        dni = ghi * (1 - phi) / sin(radians(solalt))
    else:
        a2 = alpha * k1 - k0
        K1 = 0.5 * (1 + sin(pi * ((a2 / (k1 - k0)) - 0.5)))
        eps = 1 - (1 - d1) * (a * sqrt(K1) + (1 - a) * K1 ** 2)
        phi = 1 - (alpha * k1 * (1 - eps) / kt)
        dni = ghi * (1 - phi) / sin(radians(solalt))

    dhi = ghi - dni * sin(radians(solalt))

    return ghi, dni, dhi


def Maxwell87(ghi, solalt, dn):
    """ Separation model to obtain DNI and DHI from GHI, based on Maxwell (1987)

    Formulae taken from '﻿A quasi-physical model for converting hourly global to direct normal insolation'

    :param ghi: global horizontal irradiance [W/m2]
    :param solalt: solar altitude in degrees
    :param dn: day of the year
    :return: GHI, DNI, DHI
    """

    Ieh = ih_extraterr(solalt, dn)

    if solalt <= 0:
        ghi = 0
        dni = 0
        dhi = 0
        return ghi, dni, dhi

    if ghi is np.nan:
        ghi = np.nan
        dni = np.nan
        dhi = np.nan
        return ghi, dni, dhi

    if ghi > Ieh:
        ghi = Ieh * 0.999

    kt = ghi / Ieh

    zenith = 90 - solalt
    m_kasten = 1 / (cos(radians(zenith)) + 0.15 * ((93.885 - zenith) ** (-1.253)))

    knc = 0.866 - 0.122 * m_kasten + 0.0121 * m_kasten ** 2 - 0.000653 * m_kasten ** 3 + 0.000014 * m_kasten ** 4

    if kt <= 0.6:
        a = 0.512 - 1.560 * kt + 2.286 * kt ** 2 - 2.222 * kt ** 3
        b = 0.370 + 0.962 * kt
        c = -0.28 + 0.932 * kt - 2.048 * kt ** 2
    else:
        a = -5.743 + 21.77 * kt - 27.49 * kt ** 2 + 11.56 * kt ** 3
        b = 41.40 - 118.50 * kt + 66.05 * kt ** 2 + 31.90 * kt ** 3
        c = -47.01 + 184.2 * kt - 222.0 * kt ** 2 + 73.81 * kt ** 3

    dni = Ieh * (knc - (a + b * exp(m_kasten * c)))
    dhi = ghi - dni * sin(radians(solalt))

    return ghi, dni, dhi
