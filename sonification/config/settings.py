def read_params(param_file):
    '''
    AIM: read the parameters txtfile and extract a dictionary of keys and values.
    '''
    param_dict = {}

    with open(param_file) as f:
        for line in f:
            try:
                key = line.split()[0]
                val = line.split()[1]
                param_dict[key] = val
            except:
                continue

    return param_dict