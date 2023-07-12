from scipy import signal

def bandpass_filter(lowcut,highcut,fs,order = 4):
    """
    defines bandpass Butterworth filter 

    Parameters:
    lowcut (float): lower filter limit [Hz]
    highcut (float): upper filter limit [Hz]
    fs (float): sampling rate [Hz]
    order (int): filter order

    Returns:
    b(ndarray), a(ndarray): filter coefficients 
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return b, a

def apply_filter(data,lowcut,highcut,fs=250):
    """
    applies Butterworth filter to signal

    Parameters:
    data (ndarray): array with eeg data to filter.
    lowcut (float): lower filter limit[Hz]
    highcut (float): upper filter limit[Hz]
    fs (float): sampling rate [Hz]

    Returns:
    filtered_data (ndarray): filtered signal
    
    """
    b,a = bandpass_filter(lowcut,highcut,fs)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i,j] = signal.filtfilt(b,a,data[i,j])
    return data
