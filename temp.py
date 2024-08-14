import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def analyze_csv(filepath, column):
    # Read CSV file
    df = pd.read_csv(filepath)
    pressure_data = df.iloc[:, column].tolist()
    time_data = df.iloc[:, 0].tolist()
    
    # Print the first 5 rows of the dataset
    threshold = pressure_data[0]
    print("1. Threshold is set to the first value of the pressure data: ", threshold)
    print("2. The time data is in seconds")
    print("3. The pressure data is in mmHg")
    print("4. The difference between two inhales is 1500 seconds for MIP")
    print("5. The difference between two exhales is 2250 seconds for MEP")
    # print line seperation
    print("-"*50)
    
    
    
    # Invert the dataset
    for i in range(len(pressure_data)):
        if pressure_data[i] < 1000:
            pressure_data[i] = 1000 + 1000 - pressure_data[i]
        else:
            pressure_data[i] = 1000 - (pressure_data[i] - 1000)
    
    # Find peaks and valleys
    MIP = []   # List of maximum Inhale pressure
    MEP = []   # List of maximum Exhale pressure
    previous_peaks = set()
    previous_valleys = set()
    peaks = []
    valleys = []
    
    for i in range(1, len(pressure_data) - 1):
        if pressure_data[i-1] < pressure_data[i] > pressure_data[i+1]:
            peaks.append((i, round(pressure_data[i],3)))
        elif pressure_data[i-1] > pressure_data[i] < pressure_data[i+1]:
            valleys.append((i, round(pressure_data[i],3)))
            
    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
    valleys = sorted(valleys, key=lambda x: x[1])

    def is_valid(candidate, existing, min_distance, time_data):
        if len(existing) > 0:
            for item in existing:
                if abs(time_data[candidate[0]] - time_data[item[0]]) < min_distance:
                    return False
        return True

    for peak in peaks:
        if is_valid(peak, MIP, 2250, time_data) and peak[1] not in previous_peaks and time_data[peak[0]] < 60000:
            MIP.append(peak)
            previous_peaks.add(peak[1])
        if len(MIP) == 3:
            break

    for valley in valleys:
        if is_valid(valley, MEP, 1500, time_data) and valley[1] not in previous_valleys:
            MEP.append(valley)
            previous_valleys.add(valley[1])
        if len(MEP) == 3:
            break
    
    print("MIP: ", MIP)
    print("MEP: ", MEP)
    
    threshold = pressure_data[0]
    # print("Threshold: ", threshold)
    
    peaks_ends=[]
    peaks_starts=[]
    valleys_ends=[]
    valleys_starts=[]
    # find ending index of for each MIP
    for peak in MIP:
        for i in range(peak[0], len(pressure_data)-1):
            if pressure_data[i] <= threshold:
                peaks_ends.append({i: {round(pressure_data[i],3): time_data[i]}})
                break
            
            
    # find starting index for each MIP
    for peaks in MIP:
        i=peaks[0]
        while i>0 and pressure_data[i] > threshold:
            i-=1
        peaks_starts.append({i: {round(pressure_data[i],3): time_data[i]}})
    
    print("Peaks Starts: ", peaks_starts)
    print("Peaks Ends: ", peaks_ends)
    # find ending index of for each MEP
    for valley in MEP:
        for i in range(valley[0], len(pressure_data)-1):
            if pressure_data[i] >= threshold:
                valleys_ends.append({i: {round(pressure_data[i],3): time_data[i]}})
                break
            
    # find starting index for each MEP
    for valley in MEP:
        i=valley[0]
        while i>0 and pressure_data[i] < threshold:
            i-=1
        valleys_starts.append({i: {round(pressure_data[i],3): time_data[i]}})
        
        
    # 

    def calculate_shaded_area(time_data, pressure_data, threshold, start_index, end_index):
        # Adjust pressure data by subtracting the threshold to get the difference from the threshold
        adjusted_pressure_data = [pressure - threshold for pressure in pressure_data[start_index:end_index]]
        
        # Calculate the area under the curve using the trapezoidal rule
        area = integrate.trapz(adjusted_pressure_data, time_data[start_index:end_index])
        return round(abs(area),3)  # Use abs to ensure the area is positive

        
    # calculate the area under the curve for each MIP and MEP (Total area under the curve for each MIP and MEP)
    total_peak_area = []
    total_valley_area = []
    SMIP = []
    for i in range(len(MIP)):
        start = list(peaks_starts[i].keys())[0]
        end = list(peaks_ends[i].keys())[0]
        # area = integrate.trapz(pressure_data[start:end], time_data[start:end])
        area = calculate_shaded_area(time_data, pressure_data, threshold, start, end)        
        total_peak_area.append(area)
        
        area = calculate_shaded_area(time_data, pressure_data, threshold, start, MIP[i][0])   
        SMIP.append(area)     
        
        
    for i in range(len(MEP)):
        start = list(valleys_starts[i].keys())[0]
        end = list(valleys_ends[i].keys())[0]
        # area = integrate.trapz(pressure_data[start:end], time_data[start:end])
        area=calculate_shaded_area(time_data, pressure_data, threshold, start, end)
        total_valley_area.append(area)
        
    
    
    
    # Find time between starting and peaks of each MIP and MEP
    inspiration_time = []
    expiration_time = []
    for i in range(len(MIP)):
        start = list(peaks_starts[i].keys())[0]
        end = MIP[i][0]
        inspiration_time.append(time_data[end]-time_data[start])
        
    for i in range(len(MEP)):
        start = MEP[i][0]
        end = list(valleys_ends[i].keys())[0]
        expiration_time.append(time_data[end]-time_data[start])
        
        
        
    print("Inspiration Time: ", inspiration_time)
    print("Expiration Time: ", expiration_time)
    print("SMIP: ", SMIP)
    print("Power or Total Peak Area: ", total_peak_area)
    # print("Toatl Valley Area: ", total_valley_area)
         
    
    
    
    
    
    
    
    
    
    
    
    
    # Code for Plottting
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_data, pressure_data)
    
    
    # fill the area under the curve for each MIP and MEP
    for i in range(len(MIP)):
        start = list(peaks_starts[i].keys())[0]
        end = list(peaks_ends[i].keys())[0]
        plt.fill_between(time_data[start:end], pressure_data[start:end],threshold, color='red', alpha=0.5)
        
    for i in range(len(MEP)):
        start = list(valleys_starts[i].keys())[0]
        end = list(valleys_ends[i].keys())[0]
        plt.fill_between(time_data[start:end], pressure_data[start:end],threshold, color='blue', alpha=0.5)  
    

    # Highlight MIP points
    mip_x = [time_data[point[0]] for point in MIP]
    mip_y = [point[1] for point in MIP]
    plt.scatter(mip_x, mip_y, color='red', s=100, label='MIP')

    # Highlight MEP points
    mep_x = [time_data[point[0]] for point in MEP]
    mep_y = [point[1] for point in MEP]
    plt.scatter(mep_x, mep_y, color='green', s=100, label='MEP')

    # Label individual MIP points
    for i, (x, y) in enumerate(zip(mip_x, mip_y)):
        plt.annotate(f'MIP {i+1}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    # Label individual MEP points
    for i, (x, y) in enumerate(zip(mep_x, mep_y)):
        plt.annotate(f'MEP {i+1}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center')
        
    for i, (start, end) in enumerate(zip(peaks_starts, peaks_ends)):
        start_index = list(start.keys())[0]
        end_index = list(end.keys())[0]
        
        # Scatter plot for peak starts and ends
        plt.scatter(time_data[start_index], pressure_data[start_index], color='blue', s=100)
        plt.scatter(time_data[end_index], pressure_data[end_index], color='blue', s=100)
        
        
    for i, (start, end) in enumerate(zip(valleys_starts, valleys_ends)):
        start_index = list(start.keys())[0]
        end_index = list(end.keys())[0]
        
        # Scatter plot for peak starts and ends
        plt.scatter(time_data[start_index], pressure_data[start_index], color='blue', s=100)
        plt.scatter(time_data[end_index], pressure_data[end_index], color='blue', s=100)
        
        
    plt.legend()
    plt.title('Pressure Data with MIP and MEP Highlighted')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pressure')
    plt.show()

analyze_csv('./Pressure_data_2024_07_31_13_41_02.csv', 1)