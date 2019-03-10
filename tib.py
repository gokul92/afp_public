def TIB_(df, win, bkw):
    Tvec = []
    i_prev = 0

    for i in range(win, df.shape[0]):
        if i%2000 == 0:
            print(i)

        Tvec_len = len(Tvec)

        Tsum = sum(Tvec)

        if Tvec_len <= bkw:
            theta = np.sum(df['b'][:i+1])
            Eb =  df['b'][:i+1].ewm(span = win).mean().tail(1).item()
        else:
            theta = np.sum(df['b'][i-bkw+1:i+1])
            Eb =  df['b'][i-bkw+1:i+1].ewm(span = win).mean().tail(1).item()

        if Tvec_len <= 5 and Tvec_len != 0:
            E0T = np.mean(Tvec)
        elif len(Tvec) == 0:
            E0T = 0.0
        else:
            if Tvec_len <= bkw:
                E0T = pd.Series(Tvec).ewm(span = win).mean().tail(1).item()
            else:
                E0T = pd.Series(Tvec)[Tvec_len - 1 - bkw:].ewm(span = win).mean().tail(1).item()

        if abs(theta) >= abs(Eb)*E0T:

#             if Tvec_len != 0 and Tsum + i - Tvec[-1] > df.shape[0]:
#                 break
#             print(i, abs(theta), abs(Eb)*E0T, abs(Eb), E0T)

            if len(Tvec) == 0:
                Tvec.append(i)
            else:
                Tvec.append(i - i_prev)

            i_prev = i
    return Tvec
