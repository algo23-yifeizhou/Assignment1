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
#%%
# 
# df = pd.read_csv('IH_data.csv')
# df.loc[:,df.columns.str.startswith('0_')]

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
#%%
t = 1
# sig_pre_slicer = 120
future_data_daily = future_data_raw[t]
index_data_daily = index_data_raw[t]

future_bars = future_data_daily[0]
index_bars = index_data_daily[0].values

df_future = future_data_daily[1]
df_index = index_data_daily[1]

def _get_start_price(price_df, start_price):       
    if start_price == 'today_open':
        return price_df['open'][1]
    if start_price == 'yesterday_close':
        return price_df['close'][0]

future_list = [_get_start_price(df_future,'yesterday_close'),
               _get_start_price(df_future,'today_open')]
index_list = [_get_start_price(df_index,'yesterday_close'),
               _get_start_price(df_index,'today_open')]
time_list = ['y_cl','9:30']
for _, bar in enumerate(future_bars):
    time_list.append('{:d}:{:02d}'.format(bar[0].hour,bar[0].minute))

for bar_f, bar_i in zip(future_bars, index_bars):
    future_list.append(bar_f[4])
    index_list.append(bar_i[4])



    


# for future_data_daily,index_data_daily in zip(future_data_raw,index_data_raw):
#     future_volume.append(calc_daily_data(future_data_daily,index_data_daily)[0][0])
#     open_interest.append(calc_daily_data(future_data_daily,index_data_daily)[0][1])
#%%按指定频率采样
retn_freq = 3
start_price = 'yesterday_close'

if start_price == 'yesterday_close':
    future_list.pop(1)
    time_list.pop(1)
elif start_price == 'today_open':
    future_list.pop(0)
    time_list.pop(0)
future_arr = np.array(future_list)
# future_freq = np.array(future_list)[::retn_freq]#每间隔retn_freq取close price
# future_retn = future_freq[1:] / future_freq[:-1] - 1 #计算间隔收益率,以结尾价格下标作为收益率下标

future_retn = future_arr[retn_freq:] - future_arr[:-retn_freq]
f_se = pd.Series(future_arr[retn_freq:],index=time_list[retn_freq:]) - pd.Series(future_arr[:-retn_freq],index=time_list[:-retn_freq])
#%% 分别计算不同lag长度的MA，源数据频率不变
import talib
MA_windows = (3,5,10,15)
future_MA_list = []
index_MA_list = []

for w in MA_windows:
    future_MA_list.append(talib.MA(np.array(future_list), timeperiod=w))
    index_MA_list.append(talib.MA(np.array(index_list), timeperiod=w))

future_MA_arr = np.array(future_MA_list).T
index_MA_arr = np.array(index_MA_list).T

#%% 上面那部分是取数据，下面是因子生成模块，对任何级别的数据都能够使用
# sig_slicer = slice(0,122)#slicer和普通slice一样，多了y_close和t_open，所以+2
# pre_slicer = slice(122,242)
# index_MA = pd.DataFrame(index_MA_arr,index=time_list)

# pre_retn = future_MA_arr[pre_slicer,0][-1] / future_MA_arr[pre_slicer,0][0] - 1
factor_window = 5
retn_window = 3
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
roll_windows =  arr_rolling_window(arr=future_MA_arr,window=3,axis=0)






# %%
a = [1,2,3,4,5]
a.pop(1)
print(a)
