class sadf():
    def __init__(self, data):
        self.ts = data

    # setter function for self.ts
    def set_ts(self, data):
        self.ts = data

    def apply_lags(self, lags):
        ts_diff = self.ts.diff()
        df = pd.DataFrame(np.array(ts_diff), index = ts_diff.index, columns = ['dy'])
        for lag in range(1, lags):
            df['dy' + '_lag' + str(lag)] = ts_diff.shift(lag)

        df['y_lag1'] = self.ts.shift()

        return df

    def regress(self, df):
        df['intercept'] = np.ones(df.shape[0])
        model = sm.OLS(df.dropna()['dy'], df.loc[:, df.columns != 'dy'].dropna())
        results = model.fit()

        return results.params['y_lag1']/results.bse['y_lag1']

    def get_sadf(self, lags):
        df = self.apply_lags(lags)
        sadf = self.regress(df)
        return sadf
