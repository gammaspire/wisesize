from astropy.io import fits

def load_fits(filepath):
    """
    AIM: Load user-selected FITS image and header.
    """
    
    data, header = fits.getdata(filepath, header=True)
    
    return data, header


def get_galaxy_info(filepath):
    """
    Extract galaxy name and band from the loaded filename.
    """

    filename = filepath.split("/")[-1]

    try:
        split_name = filename.replace(".", "-").split("-")

        galaxy_name = split_name[0]
        band = split_name[3]

    except Exception:
        print('Selected filename is not split with "-" characters with galaxyband; defaulting to generic wavelength.')
        galaxy_name = filename
        band = " "

    return galaxy_name, band


def load_overlay_image(image_path, band, mask_bool):
    '''
    AIM: Load the companion W1/W3 image.
    '''

    band_alt = ('W3' if (band == 'W1') or ('W1' in image_path) else 'W1')

    alt_path = image_path.replace(band, band_alt)

    dat_alt = fits.getdata(alt_path)

    dat_alt *= mask_bool

    return {'band_alt': band_alt, 'dat_alt': dat_alt}