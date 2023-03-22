
#%%
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import chain

#%%
path = '/Users/donaldhu/Downloads/group_project'
df_price = pd.read_csv(f'{path}/price.csv')
df_ratio = pd.read_excel(f'{path}/ratio.xlsx')
df_cluster_label = pd.read_csv(f'{path}/clustered_3.csv')
# %%
dict_cluster = df_cluster_label[['cluster', 'ticker']].groupby('cluster')['ticker'].apply(list).to_dict()

df_price['date'] = pd.to_datetime(df_price['date']).astype(str)
df_price = df_price[(df_price['date']>'2017-01-01') & (df_price['date']<'2023-01-01')].copy()
df_price = df_price.set_index('date')

stock_list = set(df_price.columns) & set(df_cluster_label['ticker'].values)
df_price = df_price[stock_list].copy()
# %%
df_return = df_price.apply(lambda x: np.log(1 + x.pct_change().dropna()))
df_sharpe_ratio = df_return.mean() / (df_return.std() / np.sqrt(df_return.shape[0]))


# %%
def get_portfolio_info(df_return):
    df_portfolio_return =  df_return.mean(axis=1)
    portfolio_risk = (df_portfolio_return.std() / np.sqrt(df_portfolio_return.shape[0])
    portfolio_sharpe_ratio = df_portfolio_return.mean() / portfolio_risk)
    return df_portfolio_return, portfolio_sharpe_ratio

#%%


# calculate sharpe ratio for clusters
cluster_sharpe_ratio = dict()
for key, values in dict_cluster.items():
    cluster_sharpe_ratio[key] = get_portfolio_info(df_return[values])[1]
cluster_sharpe_ratio = pd.DataFrame.from_dict(cluster_sharpe_ratio, orient='index', columns=['sharpe_ratio'])

# %%
def get_accumulate_return():
    t = pd.DataFrame()
    for i in dict_cluster.keys():
        t[i] = get_portfolio_info(df_return[dict_cluster[i]])[0]
    # t['cluster_portfolio'] = get_portfolio_info(df_return[top_sharpe_ratio_clusters_stocks])[0]
    # t['top_stock_portfolio'] = get_portfolio_info(df_return[top_sharpe_ratio_stocks])[0]
    t.cumsum().plot()
    return t.cumsum()
get_accumulate_return().to_csv(f'{path}/accumulated_return.csv')
# %%
def get_portfolio_ratio(symbols, ratio, years_of_ratios):
    df_t = df_ratio[df_ratio['Ratio'].isin(ratios)]
    df_t = pd.merge(df_t, df_cluster_label.rename(columns={'ticker':'symbol'}), on='symbol')
    df_t['avg_ratio'] = df_t[years_of_ratios].apply(np.mean, axis=1)
    df_t = df_t[['cluster', 'Ratio', 'avg_ratio']].copy()
    df_t = df_t.pivot_table(index='cluster', columns='Ratio', aggfunc=np.mean)
    def convert_pivot_to_dataframe(df):
        df.columns = df.columns.droplevel(0)
        df.columns.name = None
        df = df.reset_index() 
        return df
    df_t = convert_pivot_to_dataframe(df_t)
    print(df_t)
    return df_t
ratios = ['assetTurnover', 'cashConversionCycle', 'capitalExpenditureCoverageRatio', 'cashFlowCoverageRatios', 'dividendYield', 'priceEarningsToGrowthRatio', 'priceBookValueRatio']
get_portfolio_ratio(dict_cluster[0], ratios, ['2020', '2021', '2022']).to_csv(f'{path}/portfolio_ratios.csv', index=False)
# %%
#%%
# application trials

top_sharpe_ratio_clusters = cluster_sharpe_ratio.sort_values(by='sharpe_ratio', ascending=False).reset_index()['index'].values.tolist()[:5]
# Get top 3 sharpe ratio stock from each cluster to form a portfolio
top_sharpe_ratio_clusters_stocks = list()
for c in top_sharpe_ratio_clusters:
    t = df_sharpe_ratio.sort_values(ascending=False).reset_index()
    t = t[t['index'].isin(dict_cluster[c])]
    t = t['index'].values.tolist()[:3]
    top_sharpe_ratio_clusters_stocks += t
# By companation, form a portfolio with a equal amount of stocks
top_sharpe_ratio_stocks = df_sharpe_ratio.sort_values(ascending=False).reset_index()['index'].values.tolist()[:len(top_sharpe_ratio_clusters_stocks)]

# Check the performance
for i in ['2019', '2020', '2021', '2022']:
    df_t = df_return.reset_index().copy()
    df_t = df_t[df_t['date'].str[:4] == i].set_index('date')
    print(f'Sharpe ratio for year {i}')
    print('Constructed portfolio: ', get_portfolio_info(df_t[top_sharpe_ratio_clusters_stocks])[1])
    print('Market best stock: ', get_portfolio_info(df_t[top_sharpe_ratio_stocks])[1])

# %%
