import numpy as np
import pandas as pd
import pymc3 as pm
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
import arviz as az
from scipy.stats import ttest_ind
import statsmodels.stats.weightstats as ws

def lift_calc(df, col1, col2, default_col1):
    res = []
    for i in range(0, len(df)):
        w, r = df.loc[i, [col1, col2]]
        if w == default_col1:
            dr = r / r
        else:
            dr = r / df.loc[i-1, col2]
        res.append(dr)
    return res, np.log(res)

def cumulative_lift(df, col1, col2, default_col1, col):
    res = []
    cur = 0
    for i in range(0, len(df)):
        w, r = df.loc[i, [col1, col2]]
        if w == default_col1[0]:
            cur = 0
        elif w == default_col1[1]:
            cur = r
        else:
            cur += r
        df.loc[i, col] = cur
    return df

def weekly_lift_calculator(filtered_df, groupby):
    filtered_df = filtered_df.groupby(groupby+['zip', 'test_ctrl', 'week'], as_index=False).agg({'pv': 'mean'}).reset_index(drop=True)
    filtered_df['week lift'], filtered_df['log lift'] = lift_calc(filtered_df, col1='week', col2='pv', default_col1=6) 
    filtered_df['cumulative log lift'] = np.zeros(len(filtered_df))
    filtered_df = cumulative_lift(filtered_df, col1='week', col2='log lift', default_col1=[6, 10], col='cumulative log lift')
    res = filtered_df.groupby(['test_ctrl', 'week']+groupby)['week lift', 'log lift', 'cumulative log lift'].agg('mean')
    weeks = res.loc[('test')].index.get_level_values(0).unique().tolist()
    if groupby:
        flags = res.loc[('test', weeks[0])].index.unique().tolist()
    else:
        flags = []
    res = res.reset_index().groupby(groupby+['test_ctrl', 'week']).agg('sum')
    return res, flags, weeks

def prepost_lift_calculator(filtered_df, groupby, pre_week, post_week):
    filter_week = pre_week+post_week
    filtered_df['pre_post'] = filtered_df['week'].apply(lambda x: 'pre' if x in pre_week else 'post')
    filtered_df = filtered_df.groupby(['pre_post', 'test_ctrl']+groupby+['zip'])['pv'].agg(['sum'])
    pre, post = filtered_df.loc['pre'], filtered_df.loc['post']
    post['lift'] = post['sum'] / pre['sum']
    post['log lift'] = np.log(post['lift'])
    if groupby:
        test, ctrl = post.xs('test', level=0), post.xs('ctrl', level=0)
    else:
        test, ctrl = post.loc['test'], post.loc['ctrl']
    return test, ctrl

def filter_data(df, group_col:str, target_col:str, agg_func:str):
    q1=5
    q2=95
    k=0
    df_agg = df.groupby(group_col, as_index=False)[target_col].agg(agg_func)
    q1, q2 = np.percentile(df_agg[target_col], [q1, q2])
    iqr = q2 - q1
    lower, upper = q1 - k * iqr, q2 + k * iqr
    filtered_id = df_agg.loc[(df_agg[target_col] >= lower) & (df_agg[target_col] <= upper)][group_col]
    filtered_df = df[df[group_col].isin(filtered_id)]
    return df, filtered_df.reset_index(drop=True)

def moving_average(df, window_size, col):
    df = df.reset_index(drop=True)
    i = 0
    res = []
    while i < len(df)-window_size+1:
        window_avg = np.mean(df[col][i:i+window_size])
        res.append(window_avg)
        i += 1
    return res

def plot_cumulative(df, flags, weeks):
    weeks.sort()
    MIN, MAX = np.min(df['cumulative log lift'])-0.1, np.max(df['cumulative log lift'])+0.1
    if len(flags)==0:
        n = 1
        flag1, flag2 = ('test',), ('ctrl',)
        fig, ax = plt.subplots()
        ax.plot(weeks[:4], df.loc[flag1, 'cumulative log lift'][:4], 'o--', label='test', color='C0')
        ax.plot(weeks[:4], df.loc[flag2, 'cumulative log lift'][:4], 'o--', label='ctrl', color='C1')
        ax.hlines(np.mean(df.loc[flag1, 'cumulative log lift'][:4]), weeks[0], weeks[3], color='C0', linestyle='--')
        ax.hlines(np.mean(df.loc[flag2, 'cumulative log lift'][:4]), weeks[0], weeks[3], color='C1', linestyle='--')
        ax.plot(weeks[3:], [0]+df.loc[flag1, 'cumulative log lift'][4:].tolist(), 'o--', color='C0')
        ax.plot(weeks[3:], [0]+df.loc[flag2, 'cumulative log lift'][4:].tolist(), 'o--', color='C1')
        ax.hlines(np.mean(df.loc[flag1, 'cumulative log lift'][4:]), weeks[4], weeks[-1], color='C0', linestyle='--')
        ax.hlines(np.mean(df.loc[flag2, 'cumulative log lift'][4:]), weeks[4], weeks[-1], color='C1', linestyle='--')
        ax.legend(loc='upper left')
        ax.set_ylim(MIN, MAX)
        ax.grid()
        ax.set_title(f'cumulative log week lift')
        plt.savefig(f'C:/Users/Lizhe.Zhao/Documents/Notes/VISA/pics/cumulative log week lift.png')
    else:
        n = len(flags)
        for i in range(n):
            if (len(flags[i])>1) & (type(flags[i])!=str):
                flag1, flag2 = sum((flags[i], ('test',)), ()), sum((flags[i], ('ctrl',)), ())
            else:
                flag1, flag2 = (flags[i],)+('test',), (flags[i],)+('ctrl',)
            fig, ax = plt.subplots()
            plt.plot(weeks[:4], df.loc[flag1, 'cumulative log lift'][:4], 'o--', label='test', color='C0')
            plt.plot(weeks[:4], df.loc[flag2, 'cumulative log lift'][:4], 'o--', label='ctrl', color='C1')
            plt.hlines(np.mean(df.loc[flag1, 'cumulative log lift'][:4]), weeks[0], weeks[3], color='C0', linestyle='--')
            plt.hlines(np.mean(df.loc[flag2, 'cumulative log lift'][:4]), weeks[0], weeks[3], color='C1', linestyle='--')
            plt.plot(weeks[3:], [0]+df.loc[flag1, 'cumulative log lift'][4:].tolist(), 'o--', color='C0')
            plt.plot(weeks[3:], [0]+df.loc[flag2, 'cumulative log lift'][4:].tolist(), 'o--', color='C1')
            plt.hlines(np.mean(df.loc[flag1, 'cumulative log lift'][4:]), weeks[4], weeks[-1], color='C0', linestyle='--')
            plt.hlines(np.mean(df.loc[flag2, 'cumulative log lift'][4:]), weeks[4], weeks[-1], color='C1', linestyle='--')
            plt.legend(loc='upper left')
            plt.ylim(MIN, MAX)
            plt.grid()
            plt.title(f'cumulative {flags[i]} log week lift')
            plt.savefig(f'C:/Users/Lizhe.Zhao/Documents/Notes/VISA/pics/cumulative {flags[i]} log week lift.png')
                        
def plot_weekly(df, flags, weeks):
    weeks.sort()
    MIN, MAX = np.min(df['log lift'])-0.1, np.max(df['log lift'])+0.1
    
    if len(flags)==0:
        n = 1
        fig, ax = plt.subplots()
        flag1, flag2 = ('test',), ('ctrl',)
        ax.plot(weeks, df.loc[flag1, 'log lift'], 'o--', label='test')
        ax.plot(weeks, df.loc[flag2, 'log lift'], 'o--', label='ctrl')
        ax.legend(loc='upper left')
        ax.set_ylim(MIN, MAX)
        ax.grid()
        ax.set_title(f'log week lift')
        plt.savefig(f'C:/Users/Lizhe.Zhao/Documents/Notes/VISA/pics/log week lift.png')
    else:
        n = len(flags)
        for i in range(n):
            fig, ax = plt.subplots()
            if (len(flags[i])>1) & (type(flags[i])!=str):
                flag1, flag2 = sum((flags[i], ('test',)), ()), sum((flags[i], ('ctrl',)), ())
            else:
                flag1, flag2 = (flags[i],)+('test',), (flags[i],)+('ctrl',)
            ax.plot(weeks, df.loc[flag1, 'log lift'], 'o--', label='test')
            ax.plot(weeks, df.loc[flag2, 'log lift'], 'o--', label='ctrl')
            ax.legend(loc='upper left')
            ax.set_ylim(MIN, MAX)
            ax.grid()
            ax.set_title(f'{flags[i]} log week lift')
            plt.savefig(f'C:/Users/Lizhe.Zhao/Documents/Notes/VISA/pics/{flags[i]} log week lift.png')
        
def t_test(test, ctrl):
    t_lift, pval_lift = ttest_ind(test['lift'], ctrl['lift'], alternative='greater', equal_var=False)
    t_log_lift, pval_log_lift = ttest_ind(test['log lift'], ctrl['log lift'], alternative='greater', equal_var=False)
    return t_lift, pval_lift, t_log_lift, pval_log_lift

def z_test(test, ctrl):
    col1 = ws.DescrStatsW(test['lift'])
    col2 = ws.DescrStatsW(ctrl['lift'])
    cm_obj = ws.CompareMeans(col1, col2)    
    zstat, z_pval = cm_obj.ztest_ind(alternative='larger', usevar='unequal', value=0)
    col1 = ws.DescrStatsW(test['log lift'])
    col2 = ws.DescrStatsW(ctrl['log lift'])
    cm_obj = ws.CompareMeans(col1, col2)  
    zstat_log, z_pval_log = cm_obj.ztest_ind(alternative='larger', usevar='unequal', value=0)
    return zstat, z_pval, zstat_log, z_pval_log

def prepost_lift_calc_and_test(df, segby, pre_week, post_week):
    filtered_week = pre_week+post_week
    filtered_df = df[(filtered_df.week.isin(filtered_week))].reset_index(drop=True)
    for seg in segby:
        test, ctrl = prepost_lift_calculator(filtered_df, seg, pre_week, post_week)
        if seg:
            flags = filtered_df.groupby(seg).agg('sum').index.unique().tolist()
        else:
            flags = []
        if len(flags)>1:
            for flag in flags:
                t_lift, pval_lift, t_log_lift, pval_log_lift = t_test(test.xs(flag, level=0), ctrl.xs(flag, level=0))
                #z_lift, zpval_lift, z_log_lift, zpval_log_lift = z_test(test.xs(flag, level=0), ctrl.xs(flag, level=0))
                print(f'{flag} log lift one side t-test: {t_log_lift}, p-value: {pval_log_lift}')
                #print(f'{flag} log lift one side z-test: {z_log_lift}, p-value: {zpval_log_lift}')
                #print(f'{flag} lift one side t-test: {t_lift}, p-value: {pval_lift}')
        else:
            t_lift, pval_lift, t_log_lift, pval_log_lift = t_test(test, ctrl)
            #z_lift, zpval_lift, z_log_lift, zpval_log_lift = z_test(test, ctrl)
            print(f'total log lift one side t-test: {t_log_lift}, p-value: {pval_log_lift}')
            #print(f'total log lift one side z-test: {z_log_lift}, p-value: {zpval_log_lift}')
            #print(f'total lift one side t-test: {t_lift}, p-value: {pval_lift}')

def weekly_lift_calc_and_plot(df, segby, pre_week, post_week):
    filtered_week = pre_week+post_week
    filtered_df = df[(df.week.isin(filtered_week))].reset_index(drop=True)
    for seg in segby:
        res, flags, weeks = weekly_lift_calculator(filtered_df, seg)
        plot_weekly(res, flags, weeks)
        plot_cumulative(res, flags, weeks)
    #return res

def pooled_std(df):
    res = (np.sum((df['count']-1) * df['std']**2) / (np.sum(df['count'])-len(df)))**0.5
    return res

def Bayesian_test_filter(filtered_df, seg, week_to_drop):
    filtered_df = filtered_df[filtered_df.week!=week_to_drop]
    df_agg_zip = filtered_df.groupby(seg+['zip', 'pre_post', 'test_ctrl'])['pv'].agg(['mean', 'std', 'count', 'sum']).reset_index()
    return df_agg_zip

def pm_model(df1, df2, find_prior=True):
    # find prior
    if find_prior:
        res_df = find_model_prior(base_df=df1)
        pv_mean, pv_mean_std = res_df.loc['mu_pri', ['mean', 'sd']]
        pv_std, pv_std_std = res_df.loc['sig_pri', ['mean', 'sd']]
    else:
        pv_mean = np.mean(df1['mean'])
        pv_mean_std = np.std(df1['mean'])
        pv_std = np.mean(df1['std'])
        pv_std_std = np.std(df1['std'])

    with pm.Model() as model:

        # define priors
        mu1 = pm.Normal('mu1', mu=pv_mean, sigma=pv_mean_std)
        sig1 = pm.Normal('sig1', mu=pv_std, sigma=pv_std_std)
        mu2 = pm.Normal('mu2', mu=pv_mean, sigma=pv_mean_std)
        sig2 = pm.Normal('sig2', mu=pv_std, sigma=pv_std_std)

        # define likelihood
        obs_1 = pm.Normal('obs1', mu=mu1, sigma=sig1, observed=df1['mean'])
        obs_2 = pm.Normal('obs2', mu=mu2, sigma=sig2, observed=df2['mean'])

        # define metrics
        start = pm.find_MAP()
        step = pm.Metropolis(vars=[mu1, sig1, mu2, sig2])

        trace = pm.sample(draws=5000, step=step, tune=2000, return_inferencedata=False)
        burned_trace = trace[2000:]
    
    return burned_trace

def plot_pm_model(trace, label1, label2, title):
    fig, ax = plt.subplots()
    mean1 = trace['mu1']
    mean2 = trace['mu2']
    plt.hist(mean1, label=label1)
    plt.hist(mean2, label=label2)
    plt.legend()

    difference = mean2-mean1
    ES = difference / ((np.std(mean2))**2+(np.std(mean1))**2)**0.5
    hdi = az.hdi(ES, hdi_prob=0.95)
    rope = [-0.1, 0.1]

    fig, ax = plt.subplots()
    plt.hist(ES, density=False, label=title)
    plt.vlines(hdi[0], 0, 2500, linestyle='--', color='red', label='95% HDI')
    plt.vlines(hdi[1], 0, 2500, linestyle='--', color='red')
    plt.vlines(rope[0], 0, 2500, linestyle='--', color='black', label='ROPE')
    plt.vlines(rope[1], 0, 2500, linestyle='--', color='black')
    plt.legend(loc='upper right')
    plt.title(title)
    
def find_model_prior(base_df):
    model = pm.Model()
    with model:
        alpha_1 = pm.Normal('alpha_1', mu=np.mean(base_df['mean']), sigma=np.mean(base_df['mean']))
        beta_1 = pm.Normal('beta_1', mu=np.std(base_df['mean']), sigma=np.std(base_df['mean']))
        alpha_2 = pm.Normal('alpha_2', mu=np.mean(base_df['std']), sigma=np.mean(base_df['std']))
        beta_2 = pm.Normal('beta_2', mu=np.std(base_df['std']), sigma=np.std(base_df['std']))

        mu_pri = pm.Normal('mu_pri', mu=alpha_1, sigma=beta_1)
        sig_pri = pm.Normal('sig_pri', mu=alpha_2, sigma=beta_2)

        pv_pri = pm.Normal('pv_pri', mu=mu_pri, sigma=sig_pri, observed=base_df['mean'])

        trace = pm.sample(5000, tune=2000, return_inferencedata=False, target_accept=0.95)

    #with model:
    #    az.plot_trace(trace, compact=False)

    #with model:
    #    display(az.summary(trace, kind='stats', round_to=2))
    return az.summary(trace, kind='stats', round_to=2)

def pm_model_lift(df1, df2, find_prior=False):
    # find prior
    if find_prior:
        res_df = find_model_prior(base_df=df1)
        pv_mean, pv_mean_std = res_df.loc['mu_pri', ['mean', 'sd']]
        pv_std, pv_std_std = res_df.loc['sig_pri', ['mean', 'sd']]
    else:
        # pre_treatment, test and ctrl
        pv_mean = np.mean(df1['mean'])
        pv_mean_std = np.std(df1['mean'])
        #pv_std = np.mean(df1['std'])
        #pv_std_std = np.std(df2['std'])

    with pm.Model() as model:

        # define priors
        param = pm.Gamma('lam', alpha=pv_mean**2/pv_mean_std**2, beta=pv_mean/pv_mean_std**2)

        # define likelihood
        obs_1 = pm.Exponential('obs1', lam=param, observed=df1['mean'])
        obs_2 = pm.Exponential('obs2', lam=param, observed=df2['mean'])

        # define metrics
        start = pm.find_MAP()
        step = pm.Metropolis(vars=[param])

        trace = pm.sample(draws=5000, step=step, tune=2000)
        burned_trace = trace[2000:]
    
    return burned_trace

def plot_pm_model_lift(trace, label1, label2, title):
    fig, ax = plt.subplots()
    mean1 = 1/trace['lam']
    mean2 = 1/trace['lam']
    plt.hist(mean1, bins=20, label=label1)
    plt.hist(mean2, bins=20, label=label2)
    plt.legend()

    difference = mean2-mean1
    ES = difference / ((np.std(mean2))**2+(np.std(mean1))**2)**0.5
    hdi = az.hdi(ES, hdi_prob=0.95)
    rope = [-0.1, 0.1]

    fig, ax = plt.subplots()
    plt.hist(ES, density=False, bins=20, label=title)
    plt.vlines(hdi[0], 0, 2500, linestyle='--', color='red', label='95% HDI')
    plt.vlines(hdi[1], 0, 2500, linestyle='--', color='red')
    plt.vlines(rope[0], 0, 2500, linestyle='--', color='black', label='ROPE')
    plt.vlines(rope[1], 0, 2500, linestyle='--', color='black')
    plt.legend(loc='upper right')
    plt.title(title)
    
def calculate_lift_city_hml(hist, path):
    # seg by city, hml_flag
    seg = ['city', 'hml_flag']
    hist_by_city_hml = hist.groupby(seg+['zip', 'Date', 'test_ctrl']).agg({'pv': 'sum', 'year': 'first', 'week': 'first'})
    flags = hist_by_city_hml.index.to_series().str[:2].unique()
    total_Date = hist_by_city_hml.index.get_level_values(3).unique()
    res = pd.DataFrame()
    for i in range(len(total_Date)-7):
        for flag in flags:
            city, hml = flag[0], flag[1]
            for group in ['test', 'ctrl']:
                sliced_pre = hist_by_city_hml.loc[city, hml, :, slice(total_Date[i], total_Date[i+3]), group].groupby(['city', 'hml_flag', 'zip']).agg({'pv': 'sum', 'year':'first', 'week':'first'})
                sliced_post = hist_by_city_hml.loc[city, hml, :, slice(total_Date[i+4], total_Date[i+7]), group].groupby(['city', 'hml_flag', 'zip']).agg({'pv': 'sum', 'year':'last', 'week':'last'})
                sliced_post = sliced_post.rename(columns={'pv': 'post pv sum'})
                sliced_pre = sliced_pre.rename(columns={'pv': 'pre pv sum'})
                new_sliced = sliced_post.merge(sliced_pre, left_index=True, right_index=True)
                new_sliced['lift'] = new_sliced['post pv sum']/new_sliced['pre pv sum']
                new_sliced['log_lift'] = np.log(new_sliced['lift'])
                new_sliced['std_log_lift'] = np.std(new_sliced['log_lift'])
                new_sliced['count_zip'] = len(new_sliced)
                new_sliced['test_ctrl'] = group
                new_sliced['start_year'] = sliced_pre['year'][0]
                new_sliced['start_week'] = sliced_pre['week'][0]
                new_sliced['end_year'] = sliced_post['year'][-1]
                new_sliced['end_week'] = sliced_post['week'][-1]
                new_sliced['start_date'] = total_Date[i]
                new_sliced['end_date'] = total_Date[i+7]
                res = pd.concat([res, new_sliced])
    res = res.drop(columns=['year_x', 'week_x', 'year_y', 'week_y'])
    res = res.reset_index(drop=False)
    res.to_csv(path)