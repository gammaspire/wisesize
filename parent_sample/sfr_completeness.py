#GOAL: determine SFR completeness limit for NED-LVS parent sample 
    #CUT BY SNR FIRST!
    
#sfr_completeness(nedlvs_table['Z'][snr_combined_flag],nedlvs_table['SFR_hybrid'][snr_combined_flag], percentile, plot=False)

def sfr_completeness(z, sfr, percentile, plot=False):
    return -10