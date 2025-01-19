import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.signal.windows import kaiser
from scipy.signal import periodogram
from scipy.io import loadmat


def medfreq(signal, fs):
    # Compute the power spectral density (PSD) estimate
    f, Pxx = periodogram(signal, fs)

    # Compute the cumulative power
    cumPwr = np.cumsum(Pxx)

    # Compute the frequency widths of each frequency bin
    width = np.diff(f, prepend=0)

    # Multiply the PSD by the width to get the power within each bin
    P = width * Pxx

    # Cumulative rectangular integration
    cumPwr = np.concatenate(([0], np.cumsum(P)))

    # Place borders halfway between each estimate
    cumF = np.concatenate(([f[0]], (f[:-1] + f[1:]) / 2, [f[-1]]))

    # Find the integrated power for the entire frequency range
    Plo = np.interp(f[0], cumF, cumPwr)
    Phi = np.interp(f[-1], cumF, cumPwr)

    # Return the power between the frequency range
    pwr = Phi - Plo

    # Return the frequency that divides the power equally
    Plimit = (Plo + Phi) / 2
    f_med = np.interp(Plimit, cumPwr, cumF)

    return f_med



def peak2rms(signal, axis=None):
    """
    Ratio of largest absolute to root mean squared value.
    
    Parameters:
    signal : array_like
        Input signal.
    axis : int, optional
        Axis along which to compute the values. Default is None, which computes the value for the whole array.
    
    Returns:
    peak_to_rms_ratio : ndarray
        Output array containing the peak-to-RMS ratio.
    """
    # Calculate the peak value (maximum absolute value)
    peak_value = np.max(np.abs(signal), axis=axis)
    
    # Calculate the RMS value
    rms_value = np.sqrt(np.mean(signal**2, axis=axis))
    
    # Handle RMS value being zero
    if rms_value == 0:
        # If axis is not None, we might need to return a NaN array with the same shape as peak_value
        if axis is not None:
            return np.full_like(peak_value, np.nan)
        else:
            return np.nan
    
    # Calculate the peak-to-RMS ratio
    peak_to_rms_ratio = peak_value / rms_value
    
    return peak_to_rms_ratio

def peak_value(signal):
    """
    Find the maximum peak value in a signal.
    
    Args:
        signal (np.ndarray): Input signal as a numpy array.
    
    Returns:
        float: The maximum peak value in the signal.
    
    """

    if signal.ndim > 1:
        signal = signal.flatten()
    # Find indices of all peaks
    peaks, _ = find_peaks(signal)
    
    # Find the peak with the maximum amplitude
    max_peak_value = np.max(signal[peaks])
    
    return max_peak_value


def impulse_factor(signal):

    peak_value = np.max(np.abs(signal))
    

    mean_absolute_amplitude = np.mean(np.abs(signal))
    

    impulse_factor = peak_value / mean_absolute_amplitude
    
    return impulse_factor





def clearance_factor(signal):

    peak_value = np.max(np.abs(signal))
    squared_mean_sqrt_amplitude = np.mean(np.sqrt(np.abs(signal)))**2
    clearance_factor = peak_value / squared_mean_sqrt_amplitude
    return clearance_factor





def sinad(signal, fs):


    f, Pxx = periodogram(signal)


    fundamental_freq = f[np.argmax(Pxx)]
    dc_component = Pxx[0]


    Pxx_clean = Pxx.copy()
    Pxx_clean[np.argmax(Pxx)] = 0
    Pxx_clean[0] = 0


    noise_power = np.median(Pxx_clean)

    total_noise_distortion_power = noise_power + dc_component

    sinad = 10 * np.log10(Pxx_clean.max() / total_noise_distortion_power)
    return sinad




def crest_factor(signal):

    peak_value = np.max(np.abs(signal))
    rms_value = np.sqrt(np.mean(signal**2))
    crest_factor = peak_value / rms_value
    return crest_factor


if __name__ == '__main__':

    fs = 60

    data = loadmat('test1.mat')
    x = data['part_smd']
    x_np = np.array(x,dtype=float)
    x_np = x_np.flatten()

    median_freq = medfreq(x_np, fs)
    print(f"Median frequency: {median_freq:.2f} Hz")


    peak_to_rms = peak2rms(x)
    print(f"Peak-to-RMS ratio: {peak_to_rms:.2f}")


    crest_factor = crest_factor(x)
    print(f"crest_factor: {crest_factor:.2f}")


    impulse_factor_value = impulse_factor(x)
    print(f"Impulse factor: {impulse_factor_value:.2f}")


    clearance_factor_value = clearance_factor(x)
    print(f"Clearance factor: {clearance_factor_value:.2f}")
