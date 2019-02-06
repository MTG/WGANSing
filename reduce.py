from __future__ import division

import numpy as np
import scipy.interpolate

import pysptk


def sp_to_mgc(sp, ndim, fw, noise_floor_db=-120.0):
    # HTS uses -80, but we shift WORLD/STRAIGHT by -20 dB (so would be -100); use a little more headroom (SPTK uses doubles internally, so eps 1e-12 should still be OK)
    dtype = sp.dtype
    sp = sp.astype(np.float64)  # required for pysptk
    mgc = np.apply_along_axis(pysptk.mcep, 1, np.atleast_2d(sp), order=ndim-1, alpha=fw, maxiter=0, etype=1, eps=10**(noise_floor_db/10), min_det=0.0, itype=1)
    if sp.ndim == 1:
        mgc = mgc.flatten()
    mgc = mgc.astype(dtype)
    return mgc

def mgc_to_sp(mgc, spec_size, fw):
    dtype = mgc.dtype
    mgc = mgc.astype(np.float64)  # required for pysptk
    fftlen = 2*(spec_size - 1)
    sp = np.apply_along_axis(pysptk.mgc2sp, 1, np.atleast_2d(mgc), alpha=fw, gamma=0.0, fftlen=fftlen)
    sp = 20*np.real(sp)/np.log(10)
    if mgc.ndim == 1:
        sp = sp.flatten()
    sp = sp.astype(dtype)
    return sp

def mgc_to_mfsc(mgc):
    is_1d = mgc.ndim == 1
    mgc = np.atleast_2d(mgc)
    ndim = mgc.shape[1]

    # mirror cepstrum
    mgc1 = np.concatenate([mgc[:, :], mgc[:, -2:0:-1]], axis=-1)

    # re-scale 'dc' and 'nyquist' cepstral bins (see mcep())
    mgc1[:, 0] *= 2
    mgc1[:, ndim-1] *= 2
    
    # fft, truncate, to decibels
    mfsc = np.real(np.fft.fft(mgc1))
    mfsc = mfsc[:, :ndim]
    mfsc = 10*mfsc/np.log(10)

    if is_1d:
        mfsc = mfsc.flatten()

    return mfsc

def sp_to_mfsc(sp, ndim, fw, noise_floor_db=-120.0):
    # helper function, sp->mgc->mfsc in a single step
    mgc = sp_to_mgc(sp, ndim, fw, noise_floor_db)
    mfsc = mgc_to_mfsc(mgc)
    return mfsc
   
def get_warped_freqs(ndim, sr, fw):
    # warped frequencies
    f = np.linspace(0.0, sr/2, ndim)
    w = 2*np.pi*f/sr
    a = -fw  # XXX: why negative?
    w_warped = np.arctan2((1 - a**2)*np.sin(w), (1 + a**2)*np.cos(w) - 2*a)
    f_warped = w_warped/(2*np.pi)*sr       
    return f_warped

def mfsc_to_mgc(mfsc):
    # mfsc -> mgc -> sp is a much slower alternative to mfsc_to_sp()
    is_1d = mfsc.ndim == 1
    mfsc = np.atleast_2d(mfsc)
    ndim = mfsc.shape[1]

    mfsc = mfsc/10*np.log(10)
    mfsc1 = np.concatenate([mfsc[:, :], mfsc[:, -2:0:-1]], axis=-1)
    mgc = np.real(np.fft.ifft(mfsc1))
    mgc[:, 0] /= 2
    mgc[:, ndim-1] /= 2
    mgc = mgc[:, :ndim]

    if is_1d:
        mgc = mgc.flatten()
    
    return mgc

def mfsc_to_sp(mfsc, f_warped, spec_size, sr):
    # reconstruct sp by interpolation
    # alternative is mfsc_to_mgc(), mgc_to_sp() which uses SPTK's mgc2sp
    # interpolation is quite accurate if high mfsc dimensionality is used (e.g. 60)
    
    is_1d = mfsc.ndim == 1
    mfsc = np.atleast_2d(mfsc)
    ndim = mfsc.shape[1]

    f_sp = np.linspace(0, sr/2, spec_size)
    interp_f = scipy.interpolate.CubicSpline(f_warped, mfsc, axis=-1, bc_type='clamped', extrapolate=None)
    sp = interp_f(f_sp)

    if is_1d:
        sp = sp.flatten()
    
    return sp




# world band aperiodicities
wbap_interval = 3000.0
wbap_fmax = 15000.0
wbap_eps = 1e-12

def get_num_wbap(sr):
     return int(np.floor(min(wbap_fmax, sr/2 - wbap_interval)/wbap_interval))

def get_wbap_freqs(n_wbap, with_edges=False, sr=None):
    if with_edges:
        return np.hstack([0, np.arange(1, 1 + n_wbap)*wbap_interval, sr/2])
    else:
        return np.arange(1, 1 + n_wbap)*wbap_interval

def ap_to_wbap(ap, n_wbap, sr):
    # ap (in dB) is linear bpf at regular intervals, e.g. 3k, 6k, 9k, 12k,
    # with points at DC and Nyquist fixed to -60 and 0 dB respectively

    # interpolate bins of ap to get value at exact frequencies
    spec_size = ap.shape[1]
    f_spec = np.linspace(0, sr/2, spec_size)
    interp_f = scipy.interpolate.interp1d(f_spec, ap, kind='linear', axis=-1, assume_sorted=True)

    f_wbap = get_wbap_freqs(n_wbap, with_edges=False)
    wbap = interp_f(f_wbap)

    return wbap

def wbap_to_ap(wbap, spec_size, sr):
    # reconstruct ap by linear interpolation
    n_wbap = wbap.shape[1]
    f_wbap = get_wbap_freqs(n_wbap, with_edges=True, sr=sr)
    a_wbap = np.empty((wbap.shape[0], n_wbap+2))
    a_wbap[:, 0] = -60.0
    a_wbap[:, 1:-1] = wbap[:, :]
    a_wbap[:, -1] = -wbap_eps  # 1 - 10**((-1e-12)/20) in linear
    interp_f = scipy.interpolate.interp1d(f_wbap, a_wbap, kind='linear', axis=-1, assume_sorted=True)

    f_spec = np.linspace(0, sr/2, spec_size)
    ap = interp_f(f_spec)

    return ap
