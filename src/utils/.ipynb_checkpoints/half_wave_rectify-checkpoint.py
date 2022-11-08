def half_wave_rectify(X):
    P=X.copy()
    N=X.copy()
    P[P<0]=0
    N[N>0]=0
    N=-N

    return [P,N]