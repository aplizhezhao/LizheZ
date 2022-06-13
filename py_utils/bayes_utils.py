import numpy as np
import pandas as pd
import pymc3 as pm
import sys
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import ttest_ind, ttest_ind_from_stats
import warnings

warnings.filterwarnings('ignore')


class BayesSampler:

    def __init__(self, data_path, hist_start_date, hist_end_date, campaign_start_date, campaign_end_date, seg,
                 group_col, target_col, agg_func,
                 q1=5, q2=95, index_col='Date', period=4, useLog=False, k=0):
        self.flags = None
        self.data_path = data_path
        self.hist_start_date = hist_start_date
        self.hist_end_date = hist_end_date
        self.campaign_start_date = campaign_start_date
        self.campaign_end_date = campaign_end_date
        self.period = period
        self.seg = seg
        self.q1 = q1
        self.q2 = q2
        self.k = k
        self.index_col = index_col
        self.useLog = useLog
        self.group_col = group_col
        self.target_col = target_col
        self.agg_func = agg_func

    def read_data(self, level='store_id', filter_col='week'):
        all_hist = pd.read_csv(self.data_path)
        all_hist['Date'] = pd.to_datetime(all_hist['Date'])
        all_hist = all_hist.sort_values(by=['Date'], ascending=True).reset_index(drop=True)
        hist_20_21 = all_hist[(all_hist.Date >= self.hist_start_date) & (all_hist.Date <= self.hist_end_date)]
        test_period = all_hist[(all_hist.Date >= self.campaign_start_date) & (all_hist.Date <= self.campaign_end_date)]

        valid_list = (test_period.groupby([level])[filter_col].count() == 2 * self.period).index
        test_period = test_period[test_period.store_id.isin(valid_list)]
        hist_20_21 = hist_20_21[hist_20_21.store_id.isin(valid_list)]
        return hist_20_21, test_period

    def filter_data(self, df, hist2filter, test2filter):
        df_agg = df.groupby(self.group_col, as_index=False)[self.target_col].agg(self.agg_func)
        _q1, _q2 = np.percentile(df_agg[self.target_col], [self.q1, self.q2])
        iqr = _q2 - _q1
        lower, upper = _q1 - self.k * iqr, _q2 + self.k * iqr
        filter_id = df_agg[(df_agg[self.target_col] >= lower) & (df_agg[self.target_col] <= upper)][self.group_col]
        hist = hist2filter[hist2filter[self.group_col].isin(filter_id)]
        test = test2filter[test2filter[self.group_col].isin(filter_id)]
        return hist, test

    def preprocess(self, df, file):
        if self.seg:
            self.flags = df.groupby(self.seg).groups.keys()
        total_index = df[self.index_col].unique()
        group_col = self.seg + ['test_ctrl']
        res_list = []

        for i in range(len(total_index) - self.period * 2 + 1):
            pre_start_time, pre_end_time = total_index[i], total_index[i + self.period - 1]
            post_start_time, post_end_time = total_index[i + self.period], total_index[i + 2 * self.period - 1]
            sliced_pre = df[(df[self.index_col] >= pre_start_time) & (df[self.index_col] <= pre_end_time)].groupby(
                group_col).agg({'pv': 'sum', 'year': 'first', 'week': 'first'}).rename(
                columns={'pv': 'pv sum'})
            sliced_post = df[(df[self.index_col] >= post_start_time) & (df[self.index_col] <= post_end_time)].groupby(
                group_col).agg({'pv': 'sum', 'year': 'last', 'week': 'last'}).rename(
                columns={'pv': 'pv sum'})
            new_sliced = sliced_post[['pv sum']].merge(sliced_pre[['pv sum']], left_index=True, right_index=True,
                                                       suffixes=(' post', ' pre'))
            new_sliced['lift'] = new_sliced['pv sum post'] / new_sliced['pv sum pre']
            new_sliced['start_year'] = sliced_pre['year'].values[0]
            new_sliced['start_week'] = sliced_pre['week'].values[0]
            new_sliced['end_year'] = sliced_post['year'].values[-1]
            new_sliced['end_week'] = sliced_post['week'].values[-1]
            new_sliced['start_date'] = pre_start_time
            new_sliced['end_date'] = post_end_time
            res_list.append(new_sliced)
        res = pd.concat(res_list)
        if file:
            res.to_csv(file)
        return res

    def calc_diff(self, hist):
        hist_by_time = hist.groupby(['test_ctrl'] + self.seg + ['start_year', 'start_week', 'end_year', 'end_week']).agg(
            {'lift': 'mean'})
        hist_by_time_test = hist_by_time.loc['test']
        hist_by_time_ctrl = hist_by_time.loc['ctrl']
        hist_by_time_diff = hist_by_time_test.merge(hist_by_time_ctrl, right_index=True, left_index=True,
                                                    suffixes=('_test', '_ctrl'))
        hist_by_time_diff['diff_lift'] = hist_by_time_diff['lift_test'] / hist_by_time_diff['lift_ctrl']

        if self.useLog:
            hist_by_time_diff['diff_log_lift'] = np.log(hist_by_time_diff['lift_test']) - np.log(hist_by_time_diff['lift_ctrl'])
        return hist_by_time_diff

    def result_calc_raw(self, hist_by_time_diff, test_by_time_diff, file):
        if self.useLog:
            lift_col = 'diff_log_lift'
        else:
            lift_col = 'diff_lift'
        flags = self.flags
        res = {'seg': [], 'hist_mean': [], 'hist_sd': [], 'test_mean': []}
        for flag in flags:
            city_hml_hist = hist_by_time_diff.loc[flag]
            city_hml_hist_posterior = self.find_model_posterior(city_hml_hist[lift_col])
            city_hml_hist_mean, city_hml_hist_sd = city_hml_hist_posterior.loc['mu', 'mean'], \
                                                   city_hml_hist_posterior.loc['sd', 'mean']
            city_hml_test_mean = test_by_time_diff.loc[flag]['diff_lift'].values[0]
            res['seg'].append(flag)
            res['hist_mean'].append(city_hml_hist_mean)
            res['hist_sd'].append(city_hml_hist_sd)
            res['test_mean'].append(city_hml_test_mean)
        res = pd.DataFrame(res)
        if file:
            res.to_csv(file)
        return res

    def find_model_posterior(self, data):
        mu_prior = 1
        sd_prior = 0.01
        model = pm.Model()
        with model:
            sd = pm.Gamma('sd', alpha=1, beta=1000)
            #sd = pm.Uniform('sd', lower=0, upper=0.01)
            #sd = pm.Normal("sd", mu=sd_prior, sigma=0.001)
            mu = pm.Normal("mu", mu=mu_prior, sigma=sd)
            obs = pm.Normal("obs", mu=mu, sigma=sd, observed=data)
            trace = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.95)
        return az.summary(trace, kind='stats')

    def plot(self, res):
        if not res.seg[0]:
            x = ['total']
        else:
            x = [x for x in res.seg]
        # fig, ax = plt.subplots()
        plt.scatter(res.index, y=res.hist_mean - 1, marker='o', color='C0')
        plt.scatter(res.index, y=res.hist_mean + 2 * res.hist_sd - 1, marker='_', color='C0')
        plt.scatter(res.index, y=res.hist_mean - 2 * res.hist_sd - 1, marker='_', color='C0')
        plt.scatter(res.index, y=res.test_mean - 1, marker='x', color='C1')
        # plt.ylim(-0.06, 0.06)
        plt.grid()
        title = f'result by {self.seg}'
        plt.title(title)
        plt.xticks(ticks=res.index.values, labels=x)
        # plt.savefig(f'C:/Users/Lizhe.Zhao/Documents/Notes/VISA/pics/results/{title}.png')

    def read_data_and_calc_result(self, hist_file=None, test_file=None, res_file=None):
        # read data
        historical, test_period = self.read_data()

        # filter data
        hist, test = self.filter_data(test_period, hhistorical, test_period, )

        # preprocess
        hist = self.preprocess(hist, hist_file)
        test = self.preprocess(test, test_file)

        # calculate lift
        hist_diff = self.calc_diff(hist)
        test_diff = self.calc_diff(test)

        # simulation to find posterior
        res = result_calc_raw(hist_diff, test_diff, res_file)

        # plot result
        self.plot(res)

        return res
