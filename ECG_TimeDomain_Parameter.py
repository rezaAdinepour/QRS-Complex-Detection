import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os


clear = lambda: os.system('cls')
clear()
print('-'*35)


# Get the current directory
current_dir = os.getcwd() + '\\Dataset'
methods = ['R-R', 'QT', 'T']
methods_idx = np.arange(1, len(methods) + 1)

# List all the files in the current directory
files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]
data_idx = np.arange(1, len(files) + 1)

print('All of Available ECG Signal Files: \n')
for i in range(len(files)):
    print(str(i+1) + ') ' + files[i])

print('-'*35)

menu_data = int(input('enter number of dataset: '))
if (menu_data in data_idx):
    os.chdir(current_dir)
    if (files[menu_data - 1][0] == 'W'):
        # Load ECG signal from txt file 
        ecg_signal = np.loadtxt(files[menu_data - 1])[:, -1]
    else:
        # Load ECG signal from txt file 
        ecg_signal = np.loadtxt(files[menu_data - 1])

    print('-'*35)
    print('Available Methods: ')
    for i in range(len(methods)):
        print(str(i+1) + ') ' + methods[i])
    print('-'*35)
    menu_method = int(input('enter number of methods: '))
    if (menu_method in methods_idx):
        # Generate time axis based on the sampling rate and length of the signal
        sampling_rate = 1000  # Specify the sampling rate (samples per second)
        duration = len(ecg_signal) / sampling_rate  # Calculate the duration of the signal
        time = np.linspace(0, duration, len(ecg_signal))

        # Define the bandpass filter parameters
        lowcut = 0.5  # Lower cutoff frequency (Hz)
        highcut = 30.0  # Upper cutoff frequency (Hz)
        order = 4  # Filter order

        # Apply bandpass filter to the ECG signal
        nyquist_freq = 0.5 * sampling_rate
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(order, [low, high], btype='band')
        filtered_ecg_signal = filtfilt(b, a, ecg_signal)

else:
    print('Invalid input. Please enter a valid Number')

print('-'*35)

# R-R Detection
if menu_method is 1:
    # Apply peak detection to detect R-peaks in the filtered signal
    peaks, _ = find_peaks(filtered_ecg_signal, distance=sampling_rate/2)

    # Define the amplitude threshold for R-peaks
    amplitude_threshold = 50  # Adjust this threshold as needed

    # Remove mistaken R-points based on the R-peak amplitude threshold
    corrected_peaks = []
    for peak in peaks:
        if filtered_ecg_signal[peak] > amplitude_threshold:
            corrected_peaks.append(peak)

    # Calculate the distances between the points on the black dashed line
    line_distances = np.diff(time[corrected_peaks])

    # Plot the original ECG signal, filtered signal, and corrected R-peaks
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time, ecg_signal, color='blue', label='Original ECG Signal')
    plt.title('ECG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, filtered_ecg_signal, color='red', label='Filtered ECG Signal')
    plt.scatter(time[corrected_peaks], filtered_ecg_signal[corrected_peaks], color='black', marker='o', label='Corrected R-Peaks')
    plt.plot(time[corrected_peaks[:-1]], filtered_ecg_signal[corrected_peaks[:-1]], '--', color='black', label='R-R Interval')
    plt.axhline(amplitude_threshold, color='green', linestyle='--', label='Amplitude Threshold')
    plt.title('Filtered ECG Signal with Corrected R-Peak Detection')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Add length annotations horizontally above each line segment
    for i, distance in enumerate(line_distances):
        x_start = time[corrected_peaks[i]]
        x_end = time[corrected_peaks[i+1]]
        y = np.max(filtered_ecg_signal) + 0.05 * (np.max(filtered_ecg_signal) - np.min(filtered_ecg_signal))
        x = (x_start + x_end) / 2
        plt.text(x, y, f"{distance:.2f}s", ha='center', va='center', color='black')

    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.scatter(time[corrected_peaks[:-1]], line_distances, color='blue', s=50, edgecolors='black', facecolors='none')
    plt.plot(time[corrected_peaks[:-1]], line_distances, color='blue', linestyle='-', linewidth=1)
    plt.title('Distances Between Points on the Black Dashed Line')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (s)')
    plt.grid(True)

    # Print the calculated distances
    print('Line Distances:', line_distances)
    print('Number of Line Distances:', len(line_distances))

    plt.show()

# Q-T Interval
if menu_method is 2:
    # Find the maximum peak amplitude in the filtered ECG signal
    max_amplitude = np.max(filtered_ecg_signal)

    # Set the dynamic threshold range for Q peak detection
    threshold_low = 40
    threshold_high = 60

    # Find Q peaks in the filtered ECG signal within the threshold range
    q_peaks, _ = find_peaks(-filtered_ecg_signal, height=(threshold_low, threshold_high),
                            distance=int(0.2 * sampling_rate))

    # Calculate the distances between consecutive Q peaks
    q_intervals = np.diff(time[q_peaks])

    # Create a new figure for the plot
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(time, ecg_signal, color='blue', label='Original ECG Signal')
    plt.title('ECG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, filtered_ecg_signal, color='red', label='Filtered ECG Signal')
    plt.scatter(time[q_peaks], filtered_ecg_signal[q_peaks], color='black', marker='o', label='Q Peaks')

    # Connect even indices to odd indices with a horizontal line and write the distance
    if len(q_peaks) >= 2:
        even_indices = q_peaks[::2]  # Get even indices
        odd_indices = q_peaks[1::2]  # Get odd indices

        for i in range(min(len(even_indices), len(odd_indices))):
            start_index = even_indices[i]
            end_index = odd_indices[i]
            x_values = [time[start_index], time[end_index]]
            y_values = [filtered_ecg_signal[start_index], filtered_ecg_signal[start_index]]
            plt.plot(x_values, y_values, '--', color='black')
            distance = q_intervals[i]
            x_text = (time[start_index] + time[end_index]) / 2
            y_text = filtered_ecg_signal[start_index]
            plt.text(x_text, y_text, f'{distance:.2f}', horizontalalignment='center', verticalalignment='bottom')

    plt.title('Filtered ECG Signal with QT Intervals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    even_odd_distances = q_intervals[1::2]
    plt.plot(range(len(even_odd_distances)), even_odd_distances, color='blue', linestyle='-', label='Connecting Lines')
    plt.scatter(range(len(even_odd_distances)), even_odd_distances, color='black', marker='o', facecolors='none', edgecolors='black', label='Even-Odd Distances')
    plt.title('Distances between Even and Odd Indices')
    plt.xlabel('Point Index')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True)

    # Print the Q-Q intervals
    print('Q-Q Intervals:', q_intervals)
    print('Number of Q-Q Peaks:', len(q_intervals))

    plt.show()

# T-T Detection
if menu_method is 3:
    # Set the threshold for T peak detection
    threshold_low = 20  # Lower amplitude threshold
    threshold_high = 80  # Upper amplitude threshold

    # Find T peaks in the filtered ECG signal within the amplitude range
    t_peaks, _ = find_peaks(filtered_ecg_signal, distance=int(0.2 * sampling_rate), prominence=0.05,
                            height=(threshold_low, threshold_high))

    # Calculate the distances between consecutive T peaks
    peak_distances = np.diff(time[t_peaks])

    # Create a new figure for the scatter plot
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time, ecg_signal, color='blue', label='Original ECG Signal')
    plt.title('ECG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, filtered_ecg_signal, color='red', label='Filtered ECG Signal')
    plt.scatter(time[t_peaks], filtered_ecg_signal[t_peaks], color='black', marker='o', label='T Peaks')
    plt.title('Filtered ECG Signal with T Intervals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Create the scatter plot of peak distances
    plt.subplot(3, 1, 3)
    plt.scatter(range(len(peak_distances)), peak_distances, s=60, edgecolors='black', facecolors='none')
    plt.plot(range(len(peak_distances)), peak_distances, 'blue', linestyle='-', linewidth=1)
    plt.title('Distance between Consecutive T Peaks')
    plt.xlabel('T Peak Index')
    plt.ylabel('T-T Peak Distance (s)')
    plt.grid(True)
    plt.tight_layout()


    # Print the corrected R-R intervals
    print('Corrected T-T Intervals:', peak_distances)
    print('Number of T-T Peaks:', len(peak_distances))

    plt.show()