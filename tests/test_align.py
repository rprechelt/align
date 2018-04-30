import align
import numpy as np
import scipy.signal as signal


def test_simple_align():
    """
    Test the basic cross-correlation method with two gaussian pulses.
    """
    # make some time arrays and define a time delta
    t = np.linspace(-2*np.pi, 2*np.pi, 100)
    dt = 3.675*(t[1]-t[0])
    t = np.linspace(-1, 1, 2 * 100, endpoint=False)

    # construct two identical gaussian pulses
    x = np.real(signal.gausspulse(t, fc=5))
    y = np.real(signal.gausspulse(t+dt, fc=5))

    # compute the delay without any upsampling
    d = align.xcorr_delay(x, y)

    # and apply the delay
    y = align.apply_delay(y, d)

    # and check that the alignment is within a certain accuracy
    assert(np.std(x-y) < 0.06)


def test_padded_align_even():
    """
    Test the padded cross-correlation method with two gaussian pulses of even length.
    """
    # make some time arrays and define a time delta
    t = np.linspace(-2*np.pi, 2*np.pi, 100)
    dt = 3.675*(t[1]-t[0])
    t = np.linspace(-1, 1, 200, endpoint=False)

    # construct two identical gaussian pulses
    x = np.real(signal.gausspulse(t, fc=5))
    y = np.real(signal.gausspulse(t+dt, fc=5))

    # compute the delay without any upsampling
    d = align.xcorr_delay(x, y, 30)

    # and apply the delay
    y = align.apply_delay(y, d)

    # and check that the alignment is within a certain known accuracy
    assert(np.std(x-y) < 0.005)


def test_padded_align_odd():
    """
    Test the padded cross-correlation method with two gaussian pulses of odd length.
    """
    # make some time arrays and define a time delta
    t = np.linspace(-2*np.pi, 2*np.pi, 100)
    dt = 3.675*(t[1]-t[0])
    t = np.linspace(-1, 1, 199, endpoint=False)

    # construct two identical gaussian pulses
    x = np.real(signal.gausspulse(t, fc=5))
    y = np.real(signal.gausspulse(t+dt, fc=5))

    # compute the delay without any upsampling
    d = align.xcorr_delay(x, y, 30)

    # and apply the delay
    y = align.apply_delay(y, d)

    # and check that the alignment is within a certain known accuracy
    assert(np.std(x-y) < 0.005)


def test_poly_align():
    """
    Test polynomial interpolation of unpadded cross correlation.
    """
    # make some time arrays and define a time delta
    t = np.linspace(-2*np.pi, 2*np.pi, 100)
    dt = 3.675*(t[1]-t[0])
    t = np.linspace(-1, 1, 200, endpoint=False)

    # construct two identical gaussian pulses
    x = np.real(signal.gausspulse(t, fc=5))
    y = np.real(signal.gausspulse(t+dt, fc=5))

    # compute the delay with upsampling
    d_padded = align.xcorr_delay(x, y, 30)

    # compute the delay without any upsampling
    d = align.xcorr_delay(x, y, method='poly')
    # and check that the alignment is within a certain known accuracy
    assert(np.abs(d-d_padded) < 0.05)


def test_gauss_align():
    """
    Test Gaussian interpolation of unpadded cross correlation.
    """
    # make some time arrays and define a time delta
    t = np.linspace(-2*np.pi, 2*np.pi, 100)
    dt = 3.675*(t[1]-t[0])
    t = np.linspace(-1, 1, 200, endpoint=False)

    # construct two identical gaussian pulses
    x = np.real(signal.gausspulse(t, fc=5))
    y = np.real(signal.gausspulse(t+dt, fc=5))

    # compute the delay with upsampling
    d_padded = align.xcorr_delay(x, y, 30)

    # compute the delay without any upsampling
    d = align.xcorr_delay(x, y, method='gauss')
    # and check that the alignment is within a certain known accuracy
    assert(np.abs(d-d_padded) < 0.05)
