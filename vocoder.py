from __future__ import division

import warnings

import numpy as np

import pyworld


def extract_sp_world(x, f0, sr, hoptime, fft_size=None):
    # NOTE:
    # F0 values too low for used FFT size are handled as unvoiced in CheapTrick.
    # If this happens, we warn the user.

    f0 = f0.squeeze()
    f0 = np.ascontiguousarray(f0, dtype=np.float64)  # pyworld requires C-contiguous float64 array

    # warn for very low f0
    # fft_size = pyworld.get_cheaptrick_fft_size(sr) if fft_size is None else fft_size
    # f0_floor = pyworld.get_cheaptrick_f0_floor(sr, fft_size)
    # n_f0_too_low = int(np.sum(np.logical_and(f0 > 0, f0 <= f0_floor)))
    # if n_f0_too_low > 0:
    #     warnings.warn('F0 too low (<= {:.2f}) for FFT size ({:d}) for {:d} samples'.format(f0_floor, fft_size, n_f0_too_low))

    n_frames = f0.shape[0]
    t = np.arange(n_frames)*hoptime
    
    sp = pyworld.cheaptrick(x, f0, t, sr)

    if not np.all(np.isfinite(sp)):
        raise ValueError('Configuration or input signal caused NaNs in WORLD CheapTrick analysis')
        # NOTE: This seems to happen occassionally, e.g. taking 16kHz TIMIT audio, upsampling
        # it to 32kHz and performing WORLD analysis
    
    sp = 10*np.log10(sp)  # power to decibels

    return sp

def extract_ap_world(x, f0, sr, hoptime, fft_size=None, fill_unvoiced=True):
    # NOTE:
    # F0 is used in D4C algorithm, but simply clipped to internal `lowest_f0` (hardcoded to 40.0 Hz);
    # `lowest_f0` is also used to determine D4C's internal FFT size. For unvoiced frames, D4C just 
    # results in ap[:] = 1. So no need to handle low `f0` values specially here.

    f0 = f0.squeeze()
    f0 = np.ascontiguousarray(f0, dtype=np.float64)  # pyworld requires C-contiguous float64 array

    n_frames = f0.shape[0]
    t = np.arange(n_frames)*hoptime

    ap = pyworld.d4c(x, f0, t, sr)
    
    if not np.all(np.isfinite(ap)):
        raise ValueError('Configuration or input signal caused NaNs in WORLD D4C analysis')
    
    ap = 10.*np.log10(ap**2)  # linear to decibels
    
    if fill_unvoiced:
        # d4c sets aperodicity to 1 for unvoiced regions, 
        # to avoid discontinuities and allow changing f0 we 
        # fill these regions by extrapolating
        is_voiced = f0 > 0.0
        if not np.any(is_voiced):
            pass  # all unvoiced, do nothing
        else:
            for k in range(ap.shape[1]):
                ap[~is_voiced, k] = np.interp(np.where(~is_voiced)[0], np.where(is_voiced)[0], ap[is_voiced, k])
        #interp_f = scipy.interpolate.interp1d(np.where(is_voiced)[0], ap[is_voiced, :], kind='linear', axis=0, assume_sorted=True, bounds_error=False, fill_value='extrapolate')
        #ap[~is_voiced, :] = interp_f(np.where(~is_voiced)[0])

    return ap

def gen_wave_world(f0, sp, ap, sr, hoptime):
    # NOTE:
    # No need to handle very low or high `f0` values specially for 
    # synthesis.

    f0 = f0.squeeze()
    sp = 10**(sp/10)
    ap = 10**(ap/20)
    ap[f0 == 0, :] = 1.0  # probably has no effect
    ap = np.clip(ap, 0.0, 1.0)  # WORLD fails catastrophically for out of range aperiodicity
    f0 = np.ascontiguousarray(f0, dtype=np.float64)
    sp = np.ascontiguousarray(sp, dtype=np.float64)
    ap = np.ascontiguousarray(ap, dtype=np.float64)
    x = pyworld.synthesize(f0, sp, ap, sr, hoptime*1000.0)

    return x