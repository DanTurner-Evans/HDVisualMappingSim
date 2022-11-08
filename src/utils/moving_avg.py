import numpy as np

def moving_avg(a, span, selection = []):
    '''
    'span' should be odd number
    '''

    if (span % 2) == 0:
        raise ValueError('use odd number for span')
        
    if span>np.size(a):
        raise ValueError('span cannot be larger than the number of elements')
        
    ns = np.floor(span/2).astype(int)
    r = np.zeros(len(a))
    
    if len(selection) == 0:
        idx = np.arange(len(a)-1,-1,-1)
    else:
        idx = np.fliplr(np.sort(selection))

    for ri in idx:
        if np.isnan(a[ri]):
            r[ri] = np.nan
        else:
            r[ri] = np.mean(a[max(0,(ri-ns)):min(len(a),(ri+ns+1))])


    if len(selection)>0:
        r = r[selection]

    return r