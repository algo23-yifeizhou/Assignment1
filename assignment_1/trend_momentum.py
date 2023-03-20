#%%
import pickle
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd 
import numpy as np
# import talib
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
import numba 
from numba import jit
import pdb
import re
import scipy.stats
# from scipy.stats import ttest_1samp
from sys import getsizeof
import warnings
import datetime as dt
import matplotlib 
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')  #过滤代码运行过程中烦人的警告
matplotlib.rcParams['axes.unicode_minus']=False #解决画图中负数显示问题
#%% 评价指标
def Indicators(value):
    '''
    PARAMETERS
    value:t级别净值序列,含初始净值
    hold_freq:t级别（调仓周期）
    t_days:交易日序列,只用于计算跨月收益集中度t_hhi,len(t_days) = len(value) - 1
    
    RETURN 
    result:策略各项指标,可自行增删
    '''
    from scipy.stats import skew, kurtosis, norm
    #盈亏比
    def PlRatio(value):
        value = value[1:]-value[:-1]
        ratio = -value[value>0].mean()/value[value<0].mean()
        return ratio
    #日胜率,日基准收益为0
    def WinRate(Returns):
        pos = sum(Returns > 0)
        neg = sum(Returns < 0)
        return pos/(pos+neg)
    #最大回撤
    def MaxDrawBack(value):
        i = np.argmax(np.maximum.accumulate(value)-value)  # 结束位置
        if i == 0:
            return 0
        j = np.argmax(value[:i])  # 开始位置
        return (value[j]-value[i])/(value[j])
    #最长回撤时间，水下时间
    def MaxDrawBackDays(value):
        maxValue = 0
        maxValueIndex = []
        for k,v in enumerate(value):
            if v >= maxValue:
                maxValue = v
                maxValueIndex.append(k)
        last = len(value)-maxValueIndex[-1]-1 #回测最后处于最长回撤时期
        if len(maxValueIndex) == 1: #未创新高
            return last
        else:
            maxValueIndex = pd.Series(maxValueIndex)
            maxValueIndex -= maxValueIndex.shift(1) 
            maxValueIndex = maxValueIndex.dropna().values
            return max(maxValueIndex.max(),last) 
    #下行波动率
    def UnderVo(Returns):
        sigma = 0
        num = len(Returns)
        for k,r in enumerate(Returns):
            rMean = np.mean(Returns[:k])
            if r < rMean:
                sigma += (r-rMean)**2
        sigma = np.sqrt(sigma*250/num)
        return sigma
    
    #收益集中度  ret：收益率序列Series，index=date
    def getHHI(ret): 
        if ret.shape[0]<=2:
            return np.nan
        weight=ret/ret.sum()
        hhi=(weight**2).sum()
        hhi=(hhi-ret.shape[0]**-1)/(1.-ret.shape[0]**-1)
        return hhi
    #+/-/跨月收益集中度 
    def ReturnsConcentration(ret):
        pos_ret = ret[ret>0]
        neg_ret = ret[ret<0]
        pos_hhi = getHHI(pos_ret) # concentration of positive returns per bet
        neg_hhi = getHHI(neg_ret) # concentration of negative returns per bet
        t_hhi = getHHI(ret.groupby(pd.TimeGrouper(freq='M')).count()) # concentr. bets/month
        return pos_hhi,neg_hhi,t_hhi
    
    #PSR #ret: 1darray,1d收益率序列threshold:夏普率参照 rf: 年化无风险收益率
    def calcPsr(ret,sharpe,threshold=0,rf=0): 
        skw = skew(ret)
        kur = kurtosis(ret,fisher=False) #fisher=False:正态分布峰度=3
        prob = norm.cdf(((sharpe-threshold)*np.sqrt(ret.shape[0]))/np.sqrt(1-skw*sharpe+0.25*(kur-1)*sharpe**2))
        return prob #夏普率大于基准的概率
    #计算
    value=np.array(value)
    value=value/value[0] #每日净值 1darray
    value1=pd.Series(value)
    Returns=value1.pct_change(1).dropna().values #每日收益率 1darray
    ###
    TotalRetn = round(value[-1]*100-100,2) #总收益
    AnnualRetn= round(pow(value[-1],250/(len(value[1:])))*100-100,2) #年化收益
    Plr = round(PlRatio(value),2) #盈亏比
    Wr  = round(WinRate(Returns)*100,2) #日胜率
    Volatility     = round(np.sqrt(Returns.var()*250)*100,2) #年化波动率
    SharpRatio     = round((AnnualRetn)/Volatility,2) #年化夏普比
    # PSR
    PSRatio        = round(calcPsr(ret=Returns,sharpe=SharpRatio,threshold=1,rf=0),2) #概率夏普比 
    ###
    MaxDrawback    = round(MaxDrawBack(value)*100,2) #最大回撤
    KMRatio        = round(AnnualRetn/MaxDrawback,2) #卡玛比率
    MaxDrawbackDays= int(MaxDrawBackDays(value)) #最长回撤时间
    SortinoRatio   = round((AnnualRetn-4)/UnderVo(Returns)/100,2) #索提诺比率
    # HHI
#     Returns_s      = pd.Series(Returns)
#     Returns_s.index  = pd.to_datetime(t_days) 
#     HHI = getHHI(ret = Returns_s) #收益集中度
#     pos_hhi,neg_hhi,t_hhi = ReturnsConcentration(ret = Returns_s)
    ###
    Returns2 = np.sort(Returns)
    Max2  = round(Returns2[0]*100,2) #最大单日回撤
    var5  = round(Returns2[int(len(Returns2)*0.05)]*100,2) #收益率5%分位数
    '''
    以下输出值按需增删
    '''
    columns=['总收益','年化收益','波动率','夏普比','最大回撤','日胜率','盈亏比','最长回撤日数','calmar比率','单期最大回撤']
    data=[[TotalRetn,AnnualRetn,Volatility,SharpRatio,MaxDrawback,Wr,Plr,MaxDrawbackDays,KMRatio,Max2]]
    result=pd.DataFrame(columns=columns,data=data)
    return result
#%% plot the net value curve
def net_value(returns, benchmark=None,commission=0.0000):
    dates = returns.index
    net_value_se = pd.Series(((1-commission)*(1 + returns.values)).cumprod(),index=dates,name='intraday')
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-white')
    fig = plt.figure(figsize=(16,9))
    ax1 = plt.axes()
    ax1.set_xlabel('date')
    ax1.set_ylabel('net_value')
    ax1.set_title(net_value_se.name + '  Net Value')
    # 净值曲线
    ax1.plot(net_value_se,linestyle='-',label='Net value')
    # jis
    graph_bottom = plt.ylim()[0]
    graph_top = plt.ylim()[1]
    graph_height = graph_top - graph_bottom
    
    indi = Indicators(net_value_se)
    indi =indi.rename(index={0:net_value_se.name})
    if benchmark is not None:

        # benchmark净值曲线
        ax1.plot(benchmark_net_value,linestyle='-.',color='grey',label=bench_code)
        raph_bottom = plt.ylim()[0]
        graph_top = plt.ylim()[1]
        graph_height = graph_top - graph_bottom
    # 超额收益曲线
        excess = net_value_se.pct_change().dropna() - benchmark_net_value.pct_change().dropna()
        ax1.bar(x=excess.index,height=(0.2*graph_height/excess.min())*excess.values,bottom=1,color='orange',label='Excess return rate')
        indi_benchmark = Indicators(benchmark_net_value)
        indi_benchmark =indi_benchmark.rename(index={0:bench_code})
        indi = indi.append(indi_benchmark)
    #日亏损
    rt = net_value_se.pct_change().dropna()
    rt[rt>0]=0
    drawdown_se = rt
    ax1.bar(x=drawdown_se.index,height=(-0.2*graph_height/rt.min())*drawdown_se.values,bottom=graph_top,color='silver',label='Drawdown')
    
    
    plt.legend()
    plt.show()

    return indi

#%%#########获取bar数据############
##########################################期货标的################################################
# 样本内：2010-05-01 至 2021-05-01
# 样本内：2015-05-01 至 2021-05-01
# 样本外：2021-05-01 至 2022-11-10
#######起始日期
start_date = '2015-05-01'
######终止日期
end_date = '2023-02-01'
######标的
# IC 2015-04-16上市,2015-04-17才有主力合约信息
# IH 2015-04-16上市,2015-04-17才有主力合约信息
# IF 2010-04-16上市,2010-04-17才有主力合约信息
code = 'IH'
######采样频率/min
freq = 1
##########################################指数标的################################################
# IC:中证500，'000905.XSHG'
# IH:上证50， '000016.XSHG'
# IF:沪深300，'000300.XSHG'

if code=='IC':    
    index = '000905.XSHG'
elif code == 'IH':
    index = '000016.XSHG'
elif code == 'IF':
    index = '000300.XSHG'
else:
    raise ValueError('The code is wrong!')
    
F_file_name = 'future{}_{}min({}to{}).pkl'.format(code,freq,
                                                  start_date,
                                                  end_date)

I_file_name = 'index{}_{}min({}to{}).pkl'.format(code,freq,
                                                  start_date,
                                                  end_date)
folder_name = 'bar_data'

F_path = folder_name + '/' + F_file_name
I_path = folder_name + '/' + I_file_name
with open(F_path,'rb') as F_f:
    future_bar_list = pickle.load(F_f)
with open(I_path,'rb') as I_f:
    index_bar_list = pickle.load(I_f)  
# bar_list内的结构，[(一天的bar数据:依次为(datetime,open,high,low,close,volume,open_interest)
#                    三天的df数据：最后一天为最新日期，columns为(open,high,low,close,volume,open_interest))]
# 每组，bar的日期与df中间日期相等

# 2016-01-01 以前，期货交易时间为9:15-11:30,13:00-15:15，四个半小时
# 之后交易时间为9:00-11:30,13:00-15:00，四个小时
# bar_list当中第167号位置为2016-01-04，为第一个四小时交易日
# print(bar_list[167])
# future_data_raw = future_bar_list[167:-300]
# index_data_raw = index_bar_list[167:-300]

# future_data_raw = future_bar_list[-500:]
# index_data_raw = index_bar_list[-500:]

future_data_raw = future_bar_list[167:]
index_data_raw = index_bar_list[167:]

# future_data_raw = future_bar_list
# index_data_raw = index_bar_list
#%% 日内检测，注：部分天OLS存在多重共线性，待解决
t = 5
# sig_pre_slicer = 120
future_data_daily = future_data_raw[t]
# index_data_daily = index_data_raw[t]
future_data_yesterday = future_data_raw[t-1]
# index_data_yesterday = index_data_raw[t-1]


future_bars_daily = future_data_daily[0]
future_bars_yesterday = future_data_yesterday[0]
future_arr = np.empty((480))
time_list_y = []
time_list_t = []
for _, bar in enumerate(future_bars_daily):
    time_list_y.append('y_{:d}:{:02d}'.format(bar[0].hour,bar[0].minute))
    time_list_t.append('t_{:d}:{:02d}'.format(bar[0].hour,bar[0].minute))
time_list = time_list_y + time_list_t
for i, (bar_y, bar_t) in enumerate(zip(future_bars_yesterday, future_bars_daily)):
    future_arr[i] = bar_y[4]
    future_arr[i+240] = bar_t[4]

#收益率频率
retn_freq = 5
future_retn = future_arr[retn_freq:] / future_arr[:-retn_freq] - 1
#分别计算不同lag长度的MA，源数据频率不变
import talib
MA_windows = (3,5,10,15)
MA_number = len(MA_windows) #有几种MA 的lag
future_MA_list = []
for w in MA_windows:
    future_MA_list.append(talib.MA(future_arr, timeperiod=w))

future_MA_arr = np.array(future_MA_list).T
raw_arr_f = np.insert(future_MA_arr,[0],np.NaN, axis=1)
raw_arr_f[:-retn_freq,0] = future_retn

#%% 上面那部分是取数据，下面是因子生成模块，对任何级别的数据都能够使用
# 对每个滑窗进行多元OLS回归，返回系数贝塔
def multi_OLS_beta(window_arr):
    Y = window_arr[:,0]
    X = window_arr.copy()
    X[:,0] = 1
    beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)),X.T),Y)
    return beta
# 引入numpy滑窗函数
def arr_rolling_window(arr, window, axis=0):
    '''
    return 2D array of 滑窗array
    对于大数组用pandas的rolling apply快
    对于小数组,用np.array快
    '''
    if axis == 0:
        shape = (arr.shape[0]-window+1, window, arr.shape[-1])
        strides = (arr.strides[0],) + arr.strides
        arr_rolling = np.lib.stride_tricks.as_strided(arr,shape=shape,strides=strides)
    elif axis==1:
        shape = (arr.shape[-1]-window+1, ) + (arr.shape[0], window)
        strides = (arr.strides[-1],) + arr.strides
        arr_rolling = np.lib.stride_tricks.as_strided(arr,shape=shape,strides=strides)
    return arr_rolling

beta_OLS_window = 10
beta_expectation_window = 3
beta_sample_window = beta_OLS_window + beta_expectation_window - 1
sample_lenth = beta_sample_window + MA_number

factor_list = []
for t, MA_vector in enumerate(raw_arr_f[240:, 1:]):
    raw_sample = raw_arr_f[240+t-sample_lenth:240+t+1]
    beta_list = []
    for _, window_arr in enumerate(arr_rolling_window(arr=raw_sample,window=beta_OLS_window,axis=0)):
        beta_list.append(multi_OLS_beta(window_arr)[1:])
    beta_arr = np.array(beta_list)
    factor_value =1/beta_expectation_window *  np.dot(np.dot(np.ones(beta_expectation_window), beta_arr[-beta_expectation_window:]), MA_vector.T)
    factor_list.append(factor_value)

# prepare backtest
sig_and_pre = pd.DataFrame([factor_list,future_retn[240:]],columns=time_list_t).T
def _one_trade(sig, pre, oritation):
    if sig > 0:
        retn = pre
    elif sig < 0:
        retn = -pre
    else:
        retn = 0
    if oritation == 'momentum':
        return retn
    elif oritation == 'reverse':
        return -retn
oritation = 'momentum'  
# oritation = 'reverse'  
retn_list = []
for i, row in sig_and_pre.iterrows():
    retn_list.append(_one_trade(row[0],row[1],oritation))
retn = pd.Series(retn_list)
net_value(retn)
# %%
