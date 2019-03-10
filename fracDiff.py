class fracDiff:
    def __init__(self, data, threshold):
        self.ts = data
        self.length = len(data)
        self.threshold = threshold

    def getThresholdIndex(self, d):
        i = 2
        w = 1.0
        while abs(w) >= self.threshold:
            w_prev = w
            w = self.getDiffCoef(d, i)
            if abs(w) <= self.threshold:
                return (i-1)
            else:
                i += 1

    def setThreshold(self, thresh):
        self.threshold = thresh

    def getDiffCoef(self, d, k):
        w_arr = self.CoeffArray(d, k)
        w = w_arr[-1]
        return w

    def CoeffArray(self, d, limit):
        w = [0.0]*limit
        w[0] = 1.0
        for i in range(1, limit):
            w[i] = -w[i-1] * (d - i + 1) / i
        return w

    def weighted_avg(self, x, y):
        return sum([a*b for a, b in zip(x, y)])

    def getDiff(self, d, l_star):
        w = self.CoeffArray(d, l_star)
        ts_diff = self.ts.rolling(len(w)).apply(func = self.weighted_avg, kwargs = {'y' : w[::-1]})
        return ts_diff

    # Checking unit root presence only at first lag
    # 1.1 times critical stat for robustness
    def getadf(self, CI = 0.05):
        for d in np.linspace(0, 1, 21):
            l_star = self.getThresholdIndex(d)
            ts_diff = self.getDiff(d, l_star)
            stats = adfuller(ts_diff.dropna(), maxlag = 1, autolag = None)
            if stats[4]['5%']*1.10 > stats[0]:
                return stats[0], stats[1], d
