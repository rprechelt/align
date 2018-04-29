import scipy
import numpy as np
import scipy.signal as signal
import scipy.optimize as optimize

__all__ = ['xcorr_delay', 'apply_delay']


def xcorr_delay(x: np.ndarray, y: np.ndarray, factor: int = 1, method='sample', **kwargs) -> float:
    """
    Use the standard cross-correlation method to compute the relative
    delay in samples between two signals `x` and `y`.

    If method='sample', find the maximum value of the cross correlation
    on a sample-by-sample basis. This method has an error of +/- 0.5 samples.

    If method='gauss', fit a Gaussian to the cross correlation and use
    the maximum location of the Gaussian to get subsample accuracy. Accurate
    if the fit is successful, but may fail on certain signal shapes.

    If method='poly', fit a Quadratic to the cross correlation and use
    the maximum location of the fit to get subsample accuracy. Accurate
    if the fit is successful, but may fail on certain signal shapes.

    In order to improve accuracy, the signals can be upsampled before
    cross correlating in order to provide subsample accuracy. `factor` specifies
    the integer amount by which the signals should be upsampled.
    The larger `factor`, the more accurate the delay estimation but
    the more computationally intensive the delay estimation.

    Any additional keyword arguments are passed to `numpy.correlate`

    Parameters
    ----------
    x: np.ndarray
        The reference signal to align to
    y: np.ndarray
        The signal to find the delay of with respective to `x`
    factor: int
        The factor by which the two signals should be upsampled
             before cross correlating
    method: string
        The method to use to find the delay

    Returns
    -------
    float:
        The delay in samples between `x` and `y`
    """
    # check that factor is an integer
    factor = int(factor)

    # we resample x and y by the desired upsampling factor
    x = signal.resample(x, factor*len(x))
    y = signal.resample(y, factor*len(y))

    # the maximum of the cross correlation is a coarse estimate for the delay
    sample_delay = np.argmax(np.correlate(x, y, mode='full', **kwargs))

    if method == 'sample':
        # return the raw delay and backout the upsampling factor
        return sample_delay / float(factor)
    elif method == 'poly':
        pass
    elif method == 'gauss':
        pass



def apply_delay(x: np.ndarray, delay: float, **kwargs) -> np.ndarray:
    """
    Given a signal and a delay (in samples), use the Fourier shift
    method to apply the delay to the given signal.

    If `x` is real, this will return the real part of the aligned
    signal. If `x` is complex, this will return a complex signal.

    Any additional keyword arguments are passed to the `numpy.fft.fft`.

    Parameters
    ----------
    x: np.ndarray
        The signal to delay
    delay: float
        The delay (in samples) to apply to the signal

    Returns
    -------
    float:
        The delayed version of `x`
    """
    # get the len of the signal as we use it often
    N = len(x)

    # in order for the real part of the IFFT to be real, the frequency
    # indexing (k) has to be conjugate symmetric.
    k = np.roll(np.arange(-N//2, N//2), -N//2)

    # and we compute the delayed signal in frequency space
    phase = np.exp(-2j*np.pi*delay*k/N)
    delayed = np.fft.ifft(phase*np.fft.fft(x, **kwargs))

    # if the user provided a real signal, make sure to return a real signal
    if np.all(np.isreal(x)):
        return np.real(delayed)
    else: # otherwise return the complex signal
        return delayed

