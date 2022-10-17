import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math
import dateutil.parser as parser

mpc = pd.read_excel('GE_haleade_MPC.xlsx', header =1)
mpc = mpc.iloc[:,0:3]
mpc.columns = ['wind speed', 'WT_12MW', 'WT_14MW']


data = pd.read_excel("WLS7-436LidarTimeSeriesDecember2018_dataonly.xlsx",sheet_name="WLS7-436SpeedDirection",usecols="G")
data9 = pd.read_excel("WLS7-436LidarTimeSeriesDecember2018_dataonly.xlsx")


#Question 1

time_index = pd.date_range("2018-12-01 00:00:00", "2018-12-26 23:50:00", freq="10min")
data9['DateTime'] = pd.to_datetime(data9['DateTime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

wind_table = pd.DataFrame({'wind_speed_h100': data9["use 100 m height data"]})
wind_table = wind_table.set_index(data9["DateTime"])


wind_table = wind_table.reindex(time_index, fill_value=np.nan)
wind_table = wind_table.interpolate(method = 'linear')
wind_table.index = pd.DatetimeIndex(wind_table.index)


wind_table = wind_table['wind_speed_h100']
wind_table = np.array(wind_table)
data = wind_table
speed = np.array(data)
speed = np.delete(speed,0)


def find_power(x, mpc, wind_turbine):
    if (x>=3 & (x<=25)):
        row_above = np.max(mpc[mpc['wind speed'] <=x].index)
        row_below = row_above + 1

        section_increment = mpc.loc[row_below, wind_turbine] - mpc.loc[row_above, wind_turbine]
        power = (x - mpc.loc[row_above, 'wind speed']) / (mpc.loc[row_below, 'wind speed'] - mpc.loc[row_above, 'wind speed']) * \
            section_increment + mpc.loc[row_above, wind_turbine]

    else:
        power = 0
        
    return power
print(find_power(10.2, mpc, wind_turbine='WT_12MW'))

Rated_power_12 = 12000 * 600 #kW
Rated_power_14 = 14000 * 600
a = np.arange(.5,26,1)

#find power for each of the "bins"
WT12=[]
WT14=[]

speed_len = len(speed) #Then divide by 6 to find "split number"
split_count = speed_len/6  #to split array in groups of 6; 

speed_arrays = np.array_split(speed,split_count)
avg_speed_hour1 =[]
avg_power_hour1=[]
avg_speed_hour2 =[]
avg_power_hour2=[]
for i in range(0,len(speed_arrays)):
    avg = np.average(speed_arrays[i])
    avg_speed_hour1 = np.append(avg_speed_hour1,avg)
print(len(avg_speed_hour1))

for i in range(0, len(avg_speed_hour1)):
    avg_power_hour1= np.append(avg_power_hour1,find_power(avg_speed_hour1[i], mpc, wind_turbine='WT_12MW'))
    avg_power_hour2= np.append(avg_power_hour2,find_power(avg_speed_hour1[i], mpc, wind_turbine='WT_14MW'))




plt.hist(avg_power_hour1, bins=21)
plt.xlabel("Power in kW")
plt.ylabel("OccurencesS")
plt.title("Average Hourly Based Power")
plt.show()
counts3, bin_edges3 = np.histogram(avg_power_hour1, bins=np.arange(0, 20000, 1000))
fractions3=[]
a = np.arange(.5 * 1000 ,19.5 * 1000, 1 * 1000) #midpoints
sum_counts3 = sum(counts3)
print(bin_edges3)

for i in range(0,len(counts3)):
    fractions3.append(counts3[i]/sum_counts3)
print ("fractions")
print(fractions3)
#print(sum(fractions3)) fractions add up to 1
total_power1 = sum(fractions3 * a)

print(total_power1)
print("The cf value for WTT_12 is:")

print(total_power1 * 600 /Rated_power_12)

# 14MW

plt.hist(avg_power_hour2, bins=21)
plt.xlabel("Power in kW")
plt.ylabel("OccurencesS")
plt.title("Average Hourly Based Power")
plt.show()
counts3, bin_edges3 = np.histogram(avg_power_hour2, bins=np.arange(0, 20000, 1000))
fractions3=[]
a = np.arange(.5 * 1000 ,19.5 * 1000, 1 * 1000) #midpoints
sum_counts3 = sum(counts3)
print(bin_edges3)

for i in range(0,len(counts3)):
    fractions3.append(counts3[i]/sum_counts3)
print ("fractions")
print(fractions3)
#print(sum(fractions3)) fractions add up to 1
total_power2 = sum(fractions3 * a)

print(total_power2)

print("The cf value for WTT_14 is:")
print(total_power2 * 600 /Rated_power_14)