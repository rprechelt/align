from typing import Any

import numpy as np
import scipy.signal as signal

from .fit import fit_gauss, fit_poly

__all__ = [
    "align",
    "get_delay",
    "xcorr_delay",
    "apply_delay",
    "fft_delay",
    "absmax_delay",
    "argmax_delay",
]


def align(
    x: np.ndarray, y: np.ndarray, method: str = "xcorr", **kwargs: Any
) -> np.ndarray:
    """
    Align the `y` to `x`.

    All other arguments are passed along to the alignment method.

    Parameters
    ----------
    x: np.ndarray
        The 1D reference signal to align to.
    y: np.ndarray
        The 1D signal to find the delay of with respective to `x`.
    method: str
        The method to use to find the delay.
    """

    # apply the delay and return
    return apply_delay(y, get_delay(x, y, method, **kwargs))


def get_delay(x: np.ndarray, y: np.ndarray, method: str, **kwargs: Any) -> float:
    """
    Get the relative delay between `x` and `y`.

    All other arguments are passed along to the alignment method.

    Parameters
    ----------
    x: np.ndarray
        The 1D reference signal to align to.
    y: np.ndarray
        The 1D signal to find the delay of with respective to `x`.
    method: str
        The method to use to find the delay.
    """

    # match on the method
    if method == "xcorr":
        delay = xcorr_delay(x, y, **kwargs)
    elif method == "fft":
        delay = fft_delay(x, y, **kwargs)
    elif method == "absmax":
        delay = absmax_delay(x, y)
    elif method == "argmax":
        delay = argmax_delay(x, y)
    else:
        raise ValueError(f"Unknown alignment method: {method}")

    return delay


def absmax_delay(x: np.ndarray, y: np.ndarray) -> float:
    """
    Get the delay between `x` and `y` using the
    index of the maximum absolute value in 'y'
    and 'x'.

    This is the fastest method but has the lowest accuracy
    unless the signals are extremely similar.

    Parameters
    ----------
    x: np.ndarray
        The 1D reference signal to align to.
    y: np.ndarray
        The 1D signal to find the delay of with respective to `x`.
    """

    return float(np.abs(x).argmax() - np.abs(y).argmax())


def argmax_delay(x: np.ndarray, y: np.ndarray) -> float:
    """
    Get the delay between `x` and `y` using the
    index of the maximum absolute value in 'y'
    and 'x'.

    This is the fastest method but has the lowest accuracy
    unless the signals are extremely similar.

    Parameters
    ----------
    x: np.ndarray
        The 1D reference signal to align to.
    y: np.ndarray
        The 1D signal to find the delay of with respective to `x`.
    """

    return float(x.argmax() - y.argmax())


def xcorr_delay(
    x: np.ndarray,
    y: np.ndarray,
    factor: int = 1,
    fit: str = "sample",
    **kwargs: Any,
) -> float:
    """
    Use the standard cross-correlation method to compute the relative
    delay in samples between two signals `x` and `y`.

    If fit='sample', find the maximum value of the cross correlation
    on a sample-by-sample basis. This method has an error of +/- 0.5 samples.

    If fit='gauss', fit a Gaussian to the cross correlation and use
    the maximum location of the Gaussian to get subsample accuracy. Accurate
    if the fit is successful, but may fail on certain signal shapes.

    If fit='poly', fit a Quadratic to the cross correlation and use
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
        The 1D reference signal to align to.
    y: np.ndarray
        The 1D signal to find the delay of with respective to `x`.
    factor: int
        The factor by which the two signals should be upsampled
        before cross correlating
    fit: str
        The fit method to use to find the delay.

    Returns
    -------
    float:
        The delay in samples between `x` and `y`
    """
    # check that factor is an integer
    factor = int(factor)

    # we resample x and y by the desired upsampling factor
    xs = signal.resample(x, factor * x.shape[-1])
    ys = signal.resample(y, factor * y.shape[-1])

    # compute the cross-correlation
    xcorr = np.correlate(xs, ys, mode="same")

    # the maximum of the cross correlation is a coarse estimate for the delay
    sample_delay = int(np.argmax(xcorr))

    # modify the estimated delay based on desired fit
    if fit == "sample" or fit is None:
        pass
    elif fit == "poly":
        sample_delay += fit_poly(xcorr[sample_delay - 1 : sample_delay + 2])
    elif fit == "gauss":
        sample_delay += fit_gauss(xcorr[sample_delay - 1 : sample_delay + 2])
    else:
        raise ValueError(f"Unknown fit: {fit}")

    # a correction for odd/even upsampling effects
    odd_even_correction = -0.5 if (x.shape[-1] % 2 == 1) and (factor > 1) else 0

    # and back out the upsampling factor
    return (sample_delay / factor) - (x.shape[-1] // 2) + odd_even_correction


def fft_delay(x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
    """
    Use linear phase shift induced by a constant time delay to estimate
    the delay between signals by fitting a line to this linear phase.

    Parameters
    ----------
    x: np.ndarray
        The reference signal to align to
    y: np.ndarray
        The signal to compute the delay of w.r.t to x

    Returns
    -------
    float:
        The estimated delay in samples
    """

    # the length of the reference signal
    N = x.shape[-1]

    # compute the FFT of both signals
    X = np.fft.fft(x)
    Y = np.fft.fft(y)

    # and the cross correlation in frequency and time
    XC = X * np.conj(Y)
    xcorr = np.fft.ifft(XC)

    # and the estimate of the sample-level delay
    sample_delay = np.argmax(xcorr)

    # we weight by the absolute value of the cross-correlation
    W = np.abs(XC[1 : N // 2])

    # the phase shift corresponding to the coarse sample_delay
    phi_int = -2 * np.pi * sample_delay / N * np.arange(1, N // 2)

    # and compute the new phase after coarse correction - unwrapped
    phi = np.mod(np.angle(XC[1 : N // 2]) - phi_int + np.pi, 2 * np.pi) - np.pi

    # and the fractional sample delay using weighted linear regression
    frac_delay = (
        -(
            np.sum(np.arange(1, N // 2) * phi * W * W)
            / np.sum((np.arange(1, N // 2) * W) ** 2)
        )
        * (N // 2)
        / np.pi
    )

    # the total delay is the sum of both
    return sample_delay + frac_delay


def apply_delay(x: np.ndarray, delay: float, **kwargs: Any) -> np.ndarray:
    """
    Given a signal and a delay (in samples), use the Fourier shift
    method to apply the delay to the given signal.

    If `x` is real, this will return the real part of the aligned
    signal. If `x` is complex, this will return a complex signal.

    Any additional keyword arguments are passed to the `numpy.fft.fft`.

    Parameters
    ----------
    x: np.ndarray
        The 1D signal to delay
    delay: float
        The delay (in fractional samples) to apply to the signal.

    Returns
    -------
    float:
        The delayed version of `x`
    """
    # get the len of the signal as we use it often
    N = x.shape[-1]

    # in order for the real part of the IFFT to be real, the frequency
    # indexing (k) has to be conjugate symmetric.
    k = np.roll(np.arange(-N // 2, N // 2), -N // 2)

    # and we compute the delayed signal in frequency space
    phase = np.exp(-2j * np.pi * delay * k / N)
    delayed = np.fft.ifft(phase * np.fft.fft(x, **kwargs))

    # if the user provided a real signal, make sure to return a real signal
    if np.all(np.isreal(x)):
        return np.real(delayed)
    else:  # otherwise return the complex signal
        return delayed
