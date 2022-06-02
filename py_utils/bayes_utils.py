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
    
    def __init__(self):
        self.flags = None
        self.segs = None
        self.filter_id = None
    
    def filter_data(self, df, group_col:str, target_col:str, agg_func:str, q1, q2, k, hist2filter, test2filter):
        df_agg = df.groupby(group_col, as_index=False)[target_col].agg(agg_func)
        q1, q2 = np.percentile(df_agg[target_col], [q1, q2])
        iqr = q2 - q1
        lower, upper = q1 - k * iqr, q2 + k * iqr
        filtered_id = df_agg.loc[(df_agg[target_col] >= lower) & (df_agg[target_col] <= upper)][group_col]
        filtered_df = df[df[group_col].isin(filtered_id)]
        self.filter_id = filtered_id
        hist = hist2filter[hist2filter.zip.isin(filter_id)]
        test = test2filter[test2filter.zip.isin(filter_id)]
        return hist, test
    
    def _preprocess(self, df, seg:list, group_col:list, index_col:str, period:int, file:str):
        agg_df = df.groupby(seg+group_col).agg({'pv': 'sum', 'year':'first', 'week':'first'})
        flags = agg_df.index.to_series().str[:len(seg)-1].unique()
        self.flags = flags
        total_index = agg_df.index.get_level_values((seg+group_col).index(index_col)).unique()
        res = pd.DataFrame()
        level = len(seg) 
        
        for i in range(len(total_index)-period*2+1):
            for flag in flags:
                for group in ['test', 'ctrl']:
                    sliced_pre = agg_df.xs(flag, drop_level=False).xs(slice(total_index[i], total_index[i+period-1]), level=level, drop_level=False).xs(group, level=level+1, drop_level=False).groupby(seg).agg({'pv': 'sum', 'year':'first', 'week':'first'})
                    sliced_post = agg_df.xs(flag, drop_level=False).xs(slice(total_index[i+period], total_index[i+2*period-1]), level=level, drop_level=False).xs(group, level=level+1, drop_level=False).groupby(seg).agg({'pv': 'sum', 'year':'last', 'week':'last'})
                    sliced_post = sliced_post.rename(columns={'pv': 'post pv sum'})
                    sliced_pre = sliced_pre.rename(columns={'pv': 'pre pv sum'})
                    new_sliced = sliced_post.merge(sliced_pre, left_index=True, right_index=True)
                    new_sliced['lift'] = new_sliced['post pv sum']/new_sliced['pre pv sum']
                    new_sliced['log_lift'] = np.log(new_sliced['lift'])
                    if 'zip' in seg:                
                        new_sliced['std_log_lift'] = np.std(new_sliced['log_lift'])
                        new_sliced['count_zip'] = len(new_sliced)
                    new_sliced['test_ctrl'] = group
                    new_sliced['start_year'] = sliced_pre['year'].values[0]
                    new_sliced['start_week'] = sliced_pre['week'].values[0]
                    new_sliced['end_year'] = sliced_post['year'].values[-1]
                    new_sliced['end_week'] = sliced_post['week'].values[-1]
                    new_sliced['start_date'] = total_index[i]
                    new_sliced['end_date'] = total_index[i+2*period-1]
                    res = pd.concat([res, new_sliced])
        res = res.drop(columns=['year_x', 'week_x', 'year_y', 'week_y'])
        res = res.reset_index(drop=False)
        res.to_csv(file)
        return res
    
        
    def _preprocess_no_seg(self, df, group_col:list, index_col:str, period:int, file:str):
        # seg by None
        seg = []
        agg_df = df.groupby(seg+group_col).agg({'pv': 'sum', 'year':'first', 'week':'first'})
        total_index = agg_df.index.get_level_values(0).unique()
        res = pd.DataFrame()
        self.flags = [()]

        for i in range(len(total_index)-period*2+1):
            for group in ['test', 'ctrl']:
                sliced_pre = agg_df.loc[slice(total_index[i], total_index[i+period-1])].xs(group, level=1, drop_level=False).reset_index(drop=False)
                sliced_post = agg_df.loc[slice(total_index[i+period], total_index[i+2*period-1])].xs(group, level=1, drop_level=False).reset_index(drop=False)
                sliced_post = sliced_post.rename(columns={'pv': 'post pv sum'})
                sliced_pre = sliced_pre.rename(columns={'pv': 'pre pv sum'})
                new_sliced = sliced_post.merge(sliced_pre, left_index=True, right_index=True, suffixes=('_x', '_y'))
                new_sliced['lift'] = np.sum(new_sliced['post pv sum'])/np.sum(new_sliced['pre pv sum'])
                new_sliced['log_lift'] = np.log(new_sliced['lift'])
                if 'zip' in group_col:                
                    new_sliced['std_log_lift'] = np.std(new_sliced['log_lift'])
                    new_sliced['count_zip'] = len(new_sliced)
                new_sliced['test_ctrl'] = group
                new_sliced['start_year'] = sliced_pre['year'].values[0]
                new_sliced['start_week'] = sliced_pre['week'].values[0]
                new_sliced['end_year'] = sliced_post['year'].values[-1]
                new_sliced['end_week'] = sliced_post['week'].values[-1]
                new_sliced['start_date'] = total_index[i]
                new_sliced['end_date'] = total_index[i+2*period-1]
                res = pd.concat([res, new_sliced])
        res = res.drop(columns=['year_x', 'week_x', 'year_y', 'week_y', 'test_ctrl_x', 'test_ctrl_y', 'Date_x', 'Date_y'])
        res = res.reset_index(drop=False).iloc[:, 1:]
        res.to_csv(file)
        return res
        
        
    def preprocess(self, df, seg:list, group_col:list, index_col:str, period:int, file:str):
        if len(seg)>0:
            res = self._preprocess(df, seg, group_col, index_col, period, file)
        else:
            res = self._preprocess_no_seg(df, group_col, index_col, period, file)
        return res
        
    def read_data(self, path):
        res = pd.read_csv(path).iloc[: , 1:]
        return res
        
    def calc_diff_log_no_zip(self, hist, test, seg):
        hist_by_time = hist.groupby(['test_ctrl']+seg+['start_year', 'start_week', 'end_year', 'end_week' ]).agg({'log_lift': 'mean'})#, 'std_log_lift': 'first', 'count_zip': 'first'})
        hist_by_time_test = hist_by_time.loc['test']
        hist_by_time_ctrl = hist_by_time.loc['ctrl']
        hist_by_time_diff = hist_by_time_test.merge(hist_by_time_ctrl, right_index=True, left_index=True, suffixes=('_test', '_ctrl'))
        hist_by_time_diff['diff_log_lift'] = hist_by_time_diff['log_lift_test']-hist_by_time_diff['log_lift_ctrl']
        if len(seg)>0:
            hist_by_time_diff['diff_log_lift_std'] = hist_by_time_diff.reset_index(drop=False).groupby(seg)['diff_log_lift'].agg('std')
        else:
            hist_by_time_diff['diff_log_lift_std'] = np.std(hist_by_time_diff['diff_log_lift'])

        test_by_time = test.groupby(['test_ctrl']+seg+['start_year', 'start_week', 'end_year', 'end_week' ]).agg({'log_lift': 'mean'})
        test_by_time_test = test_by_time.loc['test']
        test_by_time_ctrl = test_by_time.loc['ctrl']
        test_by_time_diff = test_by_time_test.merge(test_by_time_ctrl, right_index=True, left_index=True, suffixes=('_test', '_ctrl'))
        test_by_time_diff['diff_log_lift'] =test_by_time_diff['log_lift_test']-test_by_time_diff['log_lift_ctrl']
        return hist_by_time_diff, test_by_time_diff
        
    def calc_diff_log(self, hist, test, seg):
        seg.remove('zip')
        hist_by_time = hist.groupby(['test_ctrl']+seg+['start_year', 'start_week', 'end_year', 'end_week' ]).agg({'log_lift': 'mean', 'std_log_lift': 'first', 'count_zip': 'first'})
        hist_by_time_test = hist_by_time.loc['test']
        hist_by_time_ctrl = hist_by_time.loc['ctrl']
        hist_by_time_diff = hist_by_time_test.merge(hist_by_time_ctrl, right_index=True, left_index=True, suffixes=('_test', '_ctrl'))
        hist_by_time_diff['diff_log_lift'] = hist_by_time_diff['log_lift_test']-hist_by_time_diff['log_lift_ctrl']
        hist_by_time_diff['diff_log_lift_std'] = (((hist_by_time_diff['count_zip_test']*hist_by_time_diff['std_log_lift_test']**2)+(hist_by_time_diff['count_zip_ctrl']*hist_by_time_diff['std_log_lift_ctrl']**2))/ (hist_by_time_diff['count_zip_test']+hist_by_time_diff['count_zip_ctrl']))**0.5
        hist_by_time_diff['dof'] = hist_by_time_diff['count_zip_test']+hist_by_time_diff['count_zip_ctrl']

        test_by_time = test.groupby(['test_ctrl']+seg+['start_year', 'start_week', 'end_year', 'end_week' ]).agg({'log_lift': 'mean', 'std_log_lift': 'first', 'count_zip': 'first'})
        test_by_time_test = test_by_time.loc['test']
        test_by_time_ctrl = test_by_time.loc['ctrl']
        test_by_time_diff = test_by_time_test.merge(test_by_time_ctrl, right_index=True, left_index=True, suffixes=('_test', '_ctrl'))
        test_by_time_diff['diff_log_lift'] =test_by_time_diff['log_lift_test']-test_by_time_diff['log_lift_ctrl']
        test_by_time_diff['diff_log_lift_std'] = (((test_by_time_diff['count_zip_test']*test_by_time_diff['std_log_lift_test']**2)+(test_by_time_diff['count_zip_ctrl']*test_by_time_diff['std_log_lift_ctrl']**2))/ (test_by_time_diff['count_zip_test']+test_by_time_diff['count_zip_ctrl']))**0.5
        test_by_time_diff['dof'] = test_by_time_diff['count_zip_test']+test_by_time_diff['count_zip_ctrl']

        return hist_by_time_diff, test_by_time_diff
        
        
    def result_calc_no_zip(self, hist_by_time_diff, test_by_time_diff):
        flags = self.flags
        res = {'seg':[], 'hist_mean':[], 'hist_sd':[], 'test_mean':[]} #, 'test_sd':[], 't_score':[], 'p_value':[]}
        for flag in flags:
            city_hml_hist = hist_by_time_diff.loc[flag]
            city_hml_test = test_by_time_diff.loc[flag]
            city_hml_hist_posterior = self.find_model_posterior(city_hml_hist['diff_log_lift'])
            city_hml_hist_mean, city_hml_hist_sd = city_hml_hist_posterior.loc['mu', 'mean'], city_hml_hist_posterior.loc['sd', 'mean']
            city_hml_test_mean = test_by_time_diff.loc[flag]['diff_log_lift'].values[0]
            res['seg'].append(flag)
            res['hist_mean'].append(city_hml_hist_mean)
            res['hist_sd'].append(city_hml_hist_sd)
            res['test_mean'].append(city_hml_test_mean)
        res = pd.DataFrame(res)
        return res
            
    def result_calc(self, hist_by_time_diff, test_by_time_diff):
        flags = self.flags
        res = {'seg':[], 'hist_mean':[], 'hist_sd':[], 'test_mean':[], 'test_sd':[], 't_score':[], 'p_value':[]}
        for flag in flags:
            city_hml_hist = hist_by_time_diff.loc[flag]
            city_hml_test = test_by_time_diff.loc[flag]
            city_hml_hist_posterior = self.find_model_posterior(city_hml_hist['diff_log_lift'])
            city_hml_hist_mean, city_hml_hist_sd = city_hml_hist_posterior.loc['mu', 'mean'], city_hml_hist_posterior.loc['sd', 'mean']
            city_hml_test_mean, city_hml_test_sd = test_by_time_diff.loc[flag]['diff_log_lift'].values[0], test_by_time_diff.loc[flag]['diff_log_lift_std'].values[0]
            t_score, p_value = ttest_ind_from_stats(city_hml_test_mean, city_hml_test_sd, city_hml_test['dof'].values[0],         city_hml_hist_mean, city_hml_hist_sd, city_hml_hist['dof'].values[0])
            res['seg'].append(flag)
            res['hist_mean'].append(city_hml_hist_mean)
            res['hist_sd'].append(city_hml_hist_sd)
            res['test_mean'].append(city_hml_test_mean)
            res['test_sd'].append(city_hml_test_sd)
            res['t_score'].append(t_score)
            res['p_value'].append(p_value)
        res = pd.DataFrame(res)
        return res
        
        
    def find_model_posterior(self, data):
        mu_prior = np.mean(data)
        sd_prior = np.std(data)
        model = pm.Model()
        with model:
            mu = pm.Normal("mu", mu=mu_prior, sigma=0.1)
            sd = pm.Normal("sd", mu=sd_prior, sigma=0.1)
            obs = pm.Normal("obs", mu=mu, sigma=sd, observed=data)
            trace = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.95)
        return az.summary(trace, kind='stats')
        
    def plot_res(self, res):
        fig, ax = plt.subplots()
        plt.scatter(x=res.seg, y=res.hist_mean, marker='o', color='C0')
        plt.scatter(x=res.seg, y=res.hist_mean+res.hist_sd, marker='_', color='C0')
        plt.scatter(x=res.seg, y=res.hist_mean-res.hist_sd, marker='_', color='C0')
        plt.scatter(x=res.seg, y=res.test_mean, marker='x', color='C1')
        plt.ylim(-0.06, 0.06)
        plt.grid()
        title = f'result by {self.seg}'
        plt.title(title)
        #plt.savefig(f'C:/Users/Lizhe.Zhao/Documents/Notes/VISA/pics/results/{title}.png')
    
    def read_processed_and_calc_result(self, file_hist, file_test, seg):
        # read data
        hist = self.read_data(file_hist)
        test = self.read_data(file_test)
        if 'zip' in seg:
            hist_by_time_diff, test_by_time_diff = self.calc_diff_log(hist, test, seg)
            res = self.result_calc(hist_by_time_diff, test_by_time_diff)
        else:
            hist_by_time_diff, test_by_time_diff = self.calc_diff_log_no_zip(hist, test, seg)
            res = self.result_calc_no_zip(hist_by_time_diff, test_by_time_diff)
        return res