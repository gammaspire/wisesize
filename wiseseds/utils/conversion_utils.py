import numpy as np

#######################
# Vcosmic to redshift #
#######################
def get_redshift(Vcosmic_array):    
    #for low-z objects, v = cz
    try:
        z=Vcosmic_array/3e5
    except:
        z=np.asarray(Vcosmic_array)/3e5
    return z


####################################
# Extinction correction for fluxes #
####################################
def apply_extinction(flux, ext_values, transmission_to_extinction=True):
    """
    Apply Milky Way extinction correction.
    - flux: linear flux array
    - ext_values: magnitude-like or transmission array
    - transmission_to_extinction: bool flag, defined in params.txt
    """
    if transmission_to_extinction:
        ext_values = -2.5 * np.log10(ext_values)
    corrections = 10 ** (ext_values / 2.5)
    return flux * corrections


##############################
# Error floor for fluxes #
##############################
def apply_error_floor(fluxes, errs, band):
    if band in ['FUV','NUV','W1','W2','W3','W4']:
        frac = 0.13
    else:
        frac = 0.10
    mask = (errs / np.abs(fluxes)) < frac
    errs[mask] = np.abs(frac * fluxes[mask])
    return errs


def clip_negative_outliers(fluxes, errs):
    mask = (fluxes<0.) & ~((0.<(fluxes+4*errs)) & (0.>(fluxes-4*errs)))
    fluxes[mask] = np.nan
    errs[mask] = np.nan
    return fluxes, errs