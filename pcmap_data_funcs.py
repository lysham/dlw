"""
All functions to download and process NOAA ISH weather data.
"""
# import all packages
from datetime import datetime, timedelta
import pytz
import csv
import numpy as np
import pandas as pd
import ftplib
import io
import gzip
import shutil
import os
from constants import SURF_ASOS


# # ASOS stations
# ASOS_SITES = [
#     'UNIVERSI OF WILLARD APT', 'BOULDER MUNICIPAL AIRPORT',
#     'DESERT ROCK AIRPORT', 'L M CLAYTON AIRPORT', 'OXFORD UNIV',
#     'UNIVERSITY PARK AIRPORT', 'JOE FOSS FIELD AIRPORT'
# ]
# # INDEX_N = [20085, 17433, 19277, 17446, 21205, 19962, 20756]
# USAF = [725315, 720533, 723870, 720541, 727686, 725128, 726510]
# WBAN = [94870, 160, 3160, 53806, 94017, 54739, 14944]


def get_asos_stations(year, local_dir):
    """
    Function to retrieve one year of data for multiple ASOS stations.
    Deprecated: Function to output station ID for US stations.
    Author: Hannah Peterson 10/08/2018
    Modified: Lysha 03/03/2023

    Parameters
    ----------
    year: required data year, int
    local_dir: local directory to save the downloaded data.
    Returns
    -------
    US_year_files: Saved file names in a list.
    Save unzipped US data files to local directory indicated by 'local_dir'.
    """
    # file_address = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.csv'
    # # read csv instead
    # df = pd.read_csv(file_address)  # total of 29705 stations worldwide
    # US_stations = df[df.CTRY == "US"]  # total of 7320 US stations
    df = pd.DataFrame.from_dict(SURF_ASOS, orient="index")
    USAF = df.usaf.values
    WBAN = df.wban.values
    US_list = [''] * len(USAF)  # initialize list of filenames of US stations: USAF-WBAN
    for i in range(0, len(USAF)):
        US_list[i] = str(USAF[i]) + '-' + str(WBAN[i])

    ftp_host = "ftp.ncdc.noaa.gov"
    with ftplib.FTP(host=ftp_host, user='ftp', passwd='') as ftpconn:
        year_dir = "pub/data/noaa/{YEAR}".format(YEAR=year)
        file_list = ftpconn.nlst(year_dir)  # returns list of file names in year directory of format 'pub/data/noaa/YEAR/USAFID-WBAN#-YEAR.gz'
        US_year_files = []
        for filename in file_list:  # TODO should revert to run through US_list
            if filename[19:31] in US_list:  # if is US station, then download and unzip and save to local directory
                US_year_files.append(filename[19:36])
                response = io.BytesIO()  # store the file in memory
                try:
                    ftpconn.retrbinary('RETR ' + filename, response.write)
                except ftplib.error_perm as err:
                    if str(err).startswith('550 '):
                        print('ERROR:', err)
                    else:
                        raise
                response.seek(0)  # jump back to the beginning of the stream
                # unzip and save to local directory
                f = os.path.join(local_dir, filename[19:36])
                with gzip.open(response, 'rb') as f_in:
                    with open(f, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f)
                ish2csv(f)  # covert to CSV and save
    return None


def ish2csv(fileName):
    """
    function to convert an ISH raw data file to a csv file.
    Author: Mengying Li 10/08/2018
    Last modified: Mengying Li 10/10/2018
    Parameters
    ----------
    fileName: string
        The name of the ish data file.
    Returns
    -------
    save a csv file with the same name to the same directory.
    """
    # open the ISH file
    with open(fileName) as fp:
        data = fp.read()  # read the file as one string
    file_csv = fileName + '.csv'

    # open an empty csv file
    with open(file_csv, mode='w') as csv_fp:
        csv_writer = csv.writer(csv_fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        USAF = data[4:10]  # station ID
        WBAN = data[10:15]  # station WBAN ID
        title = ['USAF', USAF, 'WBAN', WBAN]
        csv_writer.writerow(title)  # write title

        headers = ['UTC', 'DIR', 'SPD', 'CLG', 'VSB', 'TEMP', 'DEWP', 'SLP',
                   'CF1', 'CBH1', 'CF2', 'CBH2', 'CF3', 'CF4', 'CF5', 'CBH5',
                   'CT_low', 'CT_mid', 'CT_high', 'W_TYPE', 'PERCP']
        csv_writer.writerow(headers)  # write headers
        # write data
        for noaa_string in data.split("\n"):  # read+parse+write line by line
            try:
                csv_line = parse_ish_line(noaa_string)  # parse the string
            except:
                csv_line = ''
            csv_writer.writerow(csv_line)  # write the string to csv files
    return None


def parse_ish_line(noaa_string):
    """
    function to parse an ISH line of data, select useful data and apply data quality control.
    Author: Mengying Li 10/08/2018
    Last modified: Mengying Li 10/10/2018

    Parameters
    ----------
    noaa_string: string
        One line of ISH data.

    Returns
    -------
    csv_line: string
        Selected data separated by ',' in a csv foremat.
    """
    # scales used for some variables -- get from ISH info files
    TEMPERATURE_SCALE = 10.0
    PRESSURE_SCALE = 10.0
    SPEED_SCALE = 10
    GEO_SCALE = 1000

    actual_length = len(noaa_string)

    # change the format of time to Python datatime in UTC time
    try:
        utc_time = datetime.strptime(noaa_string[15:27], '%Y%m%d%H%M')
    except:
        """ some cases, we get 2400 hours, which is really the next day, so 
        this is a workaround for those cases """
        utc_time = noaa_string[15:27]
        print(utc_time)
        utc_time = utc_time.replace("2400", "2300")
        print(utc_time)
        utc_time = datetime.strptime(utc_time, '%Y%m%d%H%M')
        utc_time += timedelta(hours=1)
    utc_time = utc_time.replace(tzinfo=pytz.UTC)  # in UTC timezone

    UTC = utc_time  # convert to UTC datatime format
    # only data of QC==1 or 5 is recorded, otherwise a nan is recorded
    DIR = int(noaa_string[60:63]) * QC_value(noaa_string[63])  # wind direction angle in degree
    DIR = np.nan if (DIR == 999) else DIR

    SPD = int(noaa_string[65:69]) / float(SPEED_SCALE) * QC_value(noaa_string[69])  # wind speed
    SPD = np.nan if (SPD == 9999) else SPD

    CLG = int(noaa_string[70:75]) * QC_value(noaa_string[75])  # cloud ceiling
    CLG = np.nan if (CLG == 99999 or CLG == 9999) else CLG

    VSB = int(noaa_string[78:84]) * QC_value(noaa_string[84])  # visibility
    VSB = np.nan if (VSB == 999999 or VSB == 9999) else VSB

    TEMP = int(noaa_string[87:92]) / TEMPERATURE_SCALE * QC_value(noaa_string[92])  # air temperature in oC
    TEMP = np.nan if (TEMP == 999.9) else TEMP

    DEWP = int(noaa_string[93:98]) / TEMPERATURE_SCALE * QC_value(noaa_string[98])  # dew point in oC
    DEWP = np.nan if (DEWP == 999.9) else DEWP

    SLP = int(noaa_string[99:104]) / PRESSURE_SCALE * QC_value(noaa_string[104])  # air pressure relative to mean sea level (MSL)
    SLP = np.nan if (SLP == 9999.9) else SLP
    # handling the addtional data about sky cover
    CF1, CBH1, CF2, CBH2, CF3, CF4, CF5, CBH5, CT_low, CT_mid, CT_high = ['' for i in
                                                                          range(0, 11)]  # initialize variables
    try:
        CF1, CBH1, CF2, CBH2, CF3, CF4, CF5, CBH5, CT_low, CT_mid, CT_high = get_cloudInfo(noaa_string)
        W_TYPE=get_weatherInfo(noaa_string)
        PERCP=get_percipationInfo(noaa_string)
    except:
        pass

    csv_line = [UTC, DIR, SPD, CLG, VSB, TEMP, DEWP, SLP,
                CF1, CBH1, CF2, CBH2, CF3, CF4, CF5, CBH5,
                CT_low, CT_mid, CT_high, W_TYPE, PERCP]
    return csv_line


def QC_value(chars):
    """
    function to get data quality controller value.
    Author: Mengying Li 10/10/2018
    Last modified: Mengying Li 10/10/2018

    Parameters
    ----------
    chars: string
        QC indicator from raw data.

    Returns
    -------
    QC: int
        either nan or 1.
    """
    QC = np.nan if chars == ('2' or '3' or '6' or '7') else 1
    return QC


def cloud_frac(chars):
    """
    function to decode cloud fraction.
    Author: Mengying Li 10/10/2018
    Last modified: Mengying Li 10/10/2018
    Returns:
    CF: float
        numerical value of cloud fraction
    """
    if (len(chars) == 1):  # one char indicator 0~9
        CF = int(chars) / 4.0
    else:
        CF = int(chars) / 8.0
    CF = np.nan if CF > 1 else CF
    return CF


def get_cloudInfo(string):
    """
    function to get sky (cloud) cover data, and apply data quality control.
    Author: Mengying Li 10/08/2018
    Last modified: Mengying Li 10/10/2018

    Parameters
    ----------
    string: string
        ISH data string, which contains sky (cloud) cover data.

    Returns
    -------
    CF1, CF2, CF3, CF4, CF5: float,float,float,float,float
        Cloud fractions [0-1] for discrete cloud, total cloud, total cloud, opaque cloud, low cloud
    CBH1,CBH2,CBH5: float,float,float
        cloud ceiling corresponds to the previous four fractions.
    CT_low,CT_mid,CT_high: int,int,int
        code for low, middle, high cloud type.
    """
    MAP = {'GA1': ['SKY-COVER-LAYER', 13, 'discrete cloud, CF1'],
           'GD1': ['SKY-COVER-SUMMATION', 12, 'total cloud, CF2'],
           'GF1': ['SKY-COVER-SUMMATION', 23, 'total cloud and three cloud types, CF3 - CF5']
           }
    CF1, CF2, CF3, CF4, CF5 = ['' for i in range(0, 5)]  # initialize parameters
    CBH1, CBH2, CBH5 = ['' for i in range(0, 3)]  # initialize parameters
    CT_low, CT_mid, CT_high = ['' for i in range(0, 3)]  # initialize parameters
    keys = ['GA1', 'GD1', 'GF1']
    for k in range(0, len(keys)):  # loop for different products
        useable_map = MAP[keys[k]]
        pos_init = string.find(keys[k])
        if (pos_init > -1):  # if the field exists
            pos_init += len(keys[k])
            chars_to_read = string[pos_init:pos_init + useable_map[1]]
            if (k == 0):
                CF1 = cloud_frac(chars_to_read[0:2]) * QC_value(chars_to_read[2])
                CBH1 = int(chars_to_read[3:9]) * QC_value(chars_to_read[9])
                CBH1 = np.nan if (CBH1 == 99999) else CBH1
            elif (k == 1):
                CF2_1 = cloud_frac(chars_to_read[0]) * QC_value(chars_to_read[3])
                CF2_2 = cloud_frac(chars_to_read[1:3]) * QC_value(chars_to_read[3])
                CF2 = CF2_1 if np.isnan(CF2_2) else CF2_2
                CBH2 = int(chars_to_read[4:10]) * QC_value(chars_to_read[10])
                CBH2 = np.nan if (CBH2 == 99999) else CBH2
            else:
                CF3 = cloud_frac(chars_to_read[0:2]) * QC_value(chars_to_read[4])
                CF4 = cloud_frac(chars_to_read[2:4]) * QC_value(chars_to_read[4])
                CF5 = cloud_frac(chars_to_read[5:7]) * QC_value(chars_to_read[7])
                CBH5 = int(chars_to_read[11:16]) * QC_value(chars_to_read[16])
                CBH5 = np.nan if (CBH5 == 99999) else CBH5
                CT_low = int(chars_to_read[8:10]) * QC_value(chars_to_read[10])
                CT_mid = int(chars_to_read[17:19]) * QC_value(chars_to_read[19])
                CT_high = int(chars_to_read[20:22]) * QC_value(chars_to_read[22])
    # cloud fraction, cloud base height, cloud type
    return CF1, CBH1, CF2, CBH2, CF3, CF4, CF5, CBH5, CT_low, CT_mid, CT_high


def get_weatherInfo(string):
    """
    function to get weather type, and apply data quality control.
    Author: Mengying Li 11/26/2018
    Last modified: Mengying Li 11/26/2018

    Parameters
    ----------
    string: string
        ISH data string, which contains weather type.

    Returns
    -------
    W_TYPE: int
        weather type indicator.
    """
    keys=['AT1','AT2','AT3','AT4','AT5','AT6','AT7','AT8'] # upto 8 repeating field
    W_TYPE='' # empty string, allow to append
    for k in range(len(keys)):
        pos_init = string.find(keys[k]) # AT1~AT8, up to 8 repeatable field
        if (pos_init > -1):  # if the field exists
            pos_init += 5 # ATx+AU, 3 field indicator, 2 data source
            temp=int(string[pos_init:pos_init+2])
            temp2=str(temp*QC_value(string[pos_init+5]))
            W_TYPE+=str(temp2)+'/'
    return W_TYPE # weather type indicator


def get_percipationInfo(string):
    """
    function to get percipation, and apply data quality control.
    Author: Mengying Li 11/26/2018
    Last modified: Mengying Li 11/26/2018

    Parameters
    ----------
    string: string
        ISH data string, which contains percipation info.

    Returns
    -------
    percp: double
        percipation in mm.
    """
    keys=['AA1','AA2','AA3','AA4'] # upto 4 repeating field
    values=[]
    for k in range(len(keys)):
        pos_init = string.find(keys[k]) # AA1~AA4, up to 4 repeatable field
        if (pos_init > -1):  # if the field exists
            pos_init += 5 # ATx+AU, 3 field indicator, 2 data source
            temp=int(string[pos_init:pos_init+4])
            values.append(temp/10.0*QC_value(string[pos_init+5]))
    if (len(values)==0):
        percp=''
    else:
        percp=np.nanmean(np.array(values))
    return percp # percipation amount in mm
