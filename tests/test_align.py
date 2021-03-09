import numpy as np
import scipy.signal as signal

import align


def test_align_version() -> None:
    """
    Check the align version.
    """
    assert align.__version__ == "0.0.5"


def test_simple_align() -> None:
    """
    Test the basic cross-correlation method with two gaussian pulses.
    """
    # make some time arrays and define a time delta
    t = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    dt = 3.675 * (t[1] - t[0])
    t = np.linspace(-1, 1, 2 * 100, endpoint=False)

    # construct two identical gaussian pulses
    x = np.real(signal.gausspulse(t, fc=5))
    y = np.real(signal.gausspulse(t + dt, fc=5))

    # compute the delay without any upsampling
    d = align.xcorr_delay(x, y)

    # and apply the delay
    y = align.apply_delay(y, d)

    # and check that the alignment is within a certain accuracy
    assert np.std(x - y) < 0.06


def test_padded_align_even() -> None:
    """
    Test the padded cross-correlation method with two gaussian pulses of even length.
    """
    # make some time arrays and define a time delta
    t = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    dt = 3.675 * (t[1] - t[0])
    t = np.linspace(-1, 1, 200, endpoint=False)

    # construct two identical gaussian pulses
    x = np.real(signal.gausspulse(t, fc=5))
    y = np.real(signal.gausspulse(t + dt, fc=5))

    # compute the delay without any upsampling
    d = align.xcorr_delay(x, y, 30)

    # and apply the delay
    y = align.apply_delay(y, d)

    # and check that the alignment is within a certain known accuracy
    assert np.std(x - y) < 0.005


def test_padded_align_odd() -> None:
    """
    Test the padded cross-correlation method with two gaussian pulses of odd length.
    """
    # make some time arrays and define a time delta
    t = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    dt = 3.675 * (t[1] - t[0])
    t = np.linspace(-1, 1, 199, endpoint=False)

    # construct two identical gaussian pulses
    x = np.real(signal.gausspulse(t, fc=5))
    y = np.real(signal.gausspulse(t + dt, fc=5))

    # compute the delay without any upsampling
    d = align.xcorr_delay(x, y, 30)

    # and apply the delay
    y = align.apply_delay(y, d)

    # and test the top-level call
    ynew = align.align(x, y, method="xcorr", factor=30)

    # and check that the alignment is within a certain known accuracy
    assert np.std(np.abs(x - y)) < 0.005
    assert np.std(np.abs(y - ynew)) < 5e-5


def test_poly_align() -> None:
    """
    Test polynomial interpolation of unpadded cross correlation.
    """
    # make some time arrays and define a time delta
    t = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    dt = 3.675 * (t[1] - t[0])
    t = np.linspace(-1, 1, 200, endpoint=False)

    # construct two identical gaussian pulses
    x = np.real(signal.gausspulse(t, fc=5))
    y = np.real(signal.gausspulse(t + dt, fc=5))

    # compute the delay with upsampling
    d_padded = align.xcorr_delay(x, y, 30)

    # compute the delay without any upsampling
    d = align.xcorr_delay(x, y, fit="poly")
    # and check that the alignment is within a certain known accuracy
    assert np.abs(d - d_padded) < 0.05


def test_gauss_align() -> None:
    """
    Test Gaussian interpolation of unpadded cross correlation.
    """
    # make some time arrays and define a time delta
    t = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    dt = 3.675 * (t[1] - t[0])
    t = np.linspace(-1, 1, 200, endpoint=False)

    # construct two identical gaussian pulses
    x = np.real(signal.gausspulse(t, fc=5))
    y = np.real(signal.gausspulse(t + dt, fc=5))

    # compute the delay with upsampling
    d_padded = align.xcorr_delay(x, y, 30)

    # compute the delay without any upsampling
    d = align.xcorr_delay(x, y, fit="gauss")

    # compute the new signal
    y = align.apply_delay(y, d)

    # and test the top-level call
    ynew = align.align(x, y, method="xcorr", factor=30)

    # and check that the alignment is within a certain known accuracy
    assert np.abs(d - d_padded) < 0.05
    assert np.mean(np.abs(x - y)) < 7e-4
    assert np.mean(np.abs(x - ynew)) < 7e-4
    assert np.mean(np.abs(y - ynew)) < 5e-5


def test_fft_align() -> None:
    """
    Test FFT phase shift alignment
    """
    for N in [178, 179, 200, 201]:
        # make some time arrays and define a time delta
        t = np.linspace(-2 * np.pi, 2 * np.pi, N // 2)
        dt = 3.675 * (t[1] - t[0])
        t = np.linspace(-1, 1, N, endpoint=False)
        true = dt / (t[1] - t[0])

        # construct two identical gaussian pulses
        x = np.real(signal.gausspulse(t, fc=5))
        y = np.real(signal.gausspulse(t + dt, fc=5))

        # compute the delay with upsampling
        delay = align.fft_delay(x, y)

        # and apply the delay
        y = align.apply_delay(y, delay)

        # use the top-level call
        ynew = align.align(x, y, method="fft")

        # and check that the alignment is within a certain known accuracy
        assert np.abs(true - delay) < 1e-3
        assert np.mean(np.abs(x - y)) < 1e-3
        assert np.mean(np.abs(x - ynew)) < 2e-4
        assert np.mean(np.abs(y - ynew)) < 3e-5
