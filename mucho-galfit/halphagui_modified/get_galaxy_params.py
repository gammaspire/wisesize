def get_galaxy_params(OBJID,main_catalog,ephot_catalog):
    ##
    # get sizes for galaxies - will use this to unmask central region
    # need to cut this catalog based on keepflag
    ##
    from astropy.table import Table
    
    # get galaxy id

    galid = np.arange(len(main_catalog))[main_catalog['OBJID']==OBJID][0]
    #self.radius_arcsec = ephot_catalog['SMA_SB24']
    
    bad_sb25 = ephot_catalog['SMA_SB25'] == 0
    
    radius_arcsec = ephot_catalog['SMA_SB25']*(~bad_sb25) + 1.35*ephot_catalog['SMA_SB24']*bad_sb25    
    # use SMA_SB25 instead of SB24 - this should work better for both high and low SB galaxies
    # if SMA_SB25 is not available use 1.35*SMA_SB24
    
    # for galaxies with SMA_SB24=0, set radius to value in main table 
    noradius_flag = radius_arcsec == 0
    radius_arcsec[noradius_flag] = main_catalog['radius'][noradius_flag]
    
    # also save BA and PA from John's catalog
    # use the self.radius_arcsec for the sma
    BA = np.ones(len(radius_arcsec))
    PA = np.zeros(len(radius_arcsec))
    
    BA[~noradius_flag] = ephot_catalog['BA_MOMENT'][~noradius_flag]
    PA[~noradius_flag] = ephot_catalog['PA_MOMENT'][~noradius_flag]

    gRAD = radius_arcsec[galid]

    gBA = BA[galid]
    gPA = PA[galid]    
    gRA = main_catalog['RA'][galid]
    gDEC = main_catalog['DEC'][galid]
    return gRA,gDEC,gRAD,gBA,gPA