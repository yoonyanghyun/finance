import pandas as pd
import numpy as np
from itertools import groupby, chain
import matplotlib.pyplot as plt
import pygsheets  
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
pio.renderers.default = 'browser' 

"""
-------------------------------------------데이터 분석-------------------------------------------
"""
def rebase(ticker_list, dataframe):
    for ticker in ticker_list:
        dataframe[ticker] = (dataframe[ticker]/ dataframe[ticker][0])*100 
    return dataframe

def date_table(dataframe):
    #Create Date Table
    
    df_date = pd.DataFrame(index = dataframe.index)
    df_date['year'] =  dataframe.index.year
    df_date['month'] =  dataframe.index.month
    df_date['quarter'] =  dataframe.index.quarter
    df_date['eom'] =  0
    df_date['eoq'] =  0
    df_date['eoy'] =  0
    
    
    #Get End of Month / Quarter / Year
    for i in range(len(df_date)-1):
        if df_date['quarter'][i] != df_date['quarter'][i+1]:
            df_date['eoq'][i]  = 1
    
    for i in range(len(df_date)-1):
        if df_date['month'][i] != df_date['month'][i+1]:
            df_date['eom'][i]  = 1
            
    for i in range(len(df_date)-1):        
        if df_date['year'][i] != df_date['year'][i+1]:
            df_date['eoy'][i]  = 1
    
    return df_date

def scailed_weight(dataframe, df_date, df_daily_return, ticker_name, ticker_weight, Rebalancing_Period):
    #Create Portfolio Constituents Datafraame
    
    df_ticker = pd.DataFrame(index = dataframe.index, columns = ticker_name)
    df_ticker.iloc[0] = ticker_weight
    
    if Rebalancing_Period == "M":   
        for i in range(len(df_ticker)-1):        
            if df_date['eom'][i+1] == 1:
                df_ticker.iloc[i+1] = df_ticker.iloc[0] * (1+df_daily_return.iloc[i+1])
            else:
                df_ticker.iloc[i+1] = df_ticker.iloc[i] * (1+df_daily_return.iloc[i+1])
    elif Rebalancing_Period == "Q":   
        for i in range(len(df_ticker)-1):        
            if df_date['eoq'][i+1] == 1:
                df_ticker.iloc[i+1] = df_ticker.iloc[0] * (1+df_daily_return.iloc[i+1])
            else:
                df_ticker.iloc[i+1] = df_ticker.iloc[i] * (1+df_daily_return.iloc[i+1])
    elif Rebalancing_Period == "Y":   
        for i in range(len(df_ticker)-1):        
            if df_date['eoy'][i+1] == 1:
                df_ticker.iloc[i+1] = df_ticker.iloc[0] * (1+df_daily_return.iloc[i+1])
            else:
                df_ticker.iloc[i+1] = df_ticker.iloc[i] * (1+df_daily_return.iloc[i+1])
    else:
        print("rebalancing period type error")
    
    df_ticker['sum'] = df_ticker.sum(axis = 1)
    df_scaled_ticker = df_ticker.iloc[:,0:].div(df_ticker['sum'], axis=0)
    del df_scaled_ticker['sum']
    
    return df_scaled_ticker
    
def active_share(dataframe, df_scaled_ticker, ticker_weight):
    #Create Active Share Dataframe
    df_activeshare = pd.DataFrame(index = dataframe.index, columns = ["activeshare"])
    df_activeshare['activeshare'] = abs(df_scaled_ticker-ticker_weight).sum(axis = 1)/2
    
    return df_activeshare

def trading(dataframe, df_scaled_ticker, df_date, ticker_name, ticker_weight, Rebalancing_Period):
    #Create Trading Dataframe
    df_trading = pd.DataFrame(index = dataframe.index, columns = ticker_name)
    df_trading.iloc[0] = ticker_weight
    
    if Rebalancing_Period == "M":   
        for i in range(len(df_scaled_ticker)):        
            if df_date['eom'][i] == 1:
                df_trading.iloc[i] = df_scaled_ticker.iloc[i-1] - ticker_weight
            else:
                df_trading.iloc[i] = 0
        df_trading.iloc[0] = ticker_weight
    elif Rebalancing_Period == "Q":   
        for i in range(len(df_scaled_ticker)):        
            if df_date['eoq'][i] == 1:
                df_trading.iloc[i] = df_scaled_ticker.iloc[i-1] - ticker_weight
            else:
                df_trading.iloc[i] = 0
        df_trading.iloc[0] = ticker_weight
    elif Rebalancing_Period == "Y":   
        for i in range(len(df_scaled_ticker)):        
            if df_date['eoy'][i] == 1:
                df_trading.iloc[i] = df_scaled_ticker.iloc[i-1] - ticker_weight
            else:
                df_trading.iloc[i] = 0
        df_trading.iloc[0] = ticker_weight
    else:
        print("trading error")
    
    return df_trading

def turnover(dataframe, df_trading):
    #Create Turnover Dataframe    
    df_turnover = pd.DataFrame(index = dataframe.index, columns = ["buy","sell","turnover"])
    df_turnover['buy'] = df_trading[df_trading >= 0].sum(axis =1)
    df_turnover['sell'] = abs(df_trading[df_trading < 0].sum(axis =1))
    df_turnover['turnover'] = (df_turnover['buy'] + df_turnover['sell'])/2
    return df_turnover

def portfolio(dataframe, df_scaled_ticker, hedged_tickers, df_daily_return, df_fx_daily_return, df_turnover, Fee):
    #Create Daily PF Index Dataframe
    df_pf = pd.DataFrame(index = dataframe.index, columns = ["fx_return","krw_return", "daily_simple_return","daily_log_return","daily_simple_return(KRW)","daily_log_return(KRW)","portfolio_index","drawdown","portfolio_index(KRW)","drawdown(KRW)"])
    
    for i in range(len(df_scaled_ticker)):        
        if i == 0:
            df_pf["daily_simple_return"].iloc[0] = -df_turnover['turnover'].iloc[0]*Fee*2
        else:
            df_pf["daily_simple_return"].iloc[i] = ((df_scaled_ticker.iloc[i-1] * (1+df_daily_return.iloc[i])).sum()-1)-df_turnover['turnover'].iloc[i]*Fee*2
    
    df_pf['fx_return'] = pd.concat([df_pf, df_fx_daily_return], axis = 1).fillna(0)['krwusd']
    
    temp_list = [(ticker) for ticker in df_scaled_ticker.columns if ticker in hedged_tickers]
    
    if len(temp_list) != 0 :
        for ticker in hedged_tickers:
            for i in range(len(df_pf-1)):
                df_pf["krw_return"].iloc[i] = - (df_scaled_ticker.iloc[i][ticker])*(df_pf['fx_return'].iloc[i])
    else: df_pf["krw_return"] = 0    

    df_pf["daily_simple_return(KRW)"] = (1+df_pf["daily_simple_return"])*(1+df_pf['fx_return'])-1                  
    df_pf["daily_simple_return(KRW)"] = df_pf["daily_simple_return(KRW)"] + df_pf["krw_return"]
    del df_pf['krw_return']

    df_pf['daily_log_return'] = np.log(1 + df_pf['daily_simple_return'].astype(float))
    df_pf['daily_log_return(KRW)'] = np.log(1 + df_pf['daily_simple_return(KRW)'].astype(float))
    df_pf['portfolio_index'] = (1+df_pf['daily_simple_return']).cumprod()*1000
    df_pf['portfolio_index(KRW)'] = (1+df_pf['daily_simple_return(KRW)']).cumprod()*1000
    
    df_pf['drawdown'] = drawdown(df_pf['daily_simple_return'])
    df_pf['drawdown(KRW)'] = drawdown(df_pf['daily_simple_return(KRW)'])
    
    return df_pf

# 1D, 1W, 1M, 1Q, 1Y 식으로 periodiic type 활용
def generate_return_statistics(dataframe, periodic_type):
    #Create periodic statistics
    periodic_index = dataframe.resample(periodic_type).ffill()
    periodic_return = periodic_index.pct_change().groupby\
        ([periodic_index.index.year, periodic_index.index.month]).\
            mean()
    periodic_return.iloc[0] = periodic_index[0] / 1000 - 1
    
    periodic_return_list = []
    for i in range(len(periodic_return)):
        periodic_return_list.append\
            ({'period':periodic_return.index[i][1],
              'periodic_return': periodic_return.values[i]})
    df_periodic_return = pd.DataFrame(periodic_return_list, columns = ('period','periodic_return'))
    
    return df_periodic_return

def sum_returns(returns, groupby, compounded=True):
    def returns_prod(data):
        return (data + 1).prod() - 1
    if compounded:
        return returns.groupby(groupby).apply(returns_prod)
    return returns.groupby(groupby).sum()    
    
def monthly_return_table(returns, eoy=False, is_prices=False, compounded=True):

    # get close / first column if given DataFrame
    if isinstance(returns, pd.DataFrame):
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and 'close' in returns.columns:
            returns = returns['close']
        else:
            returns = returns[returns.columns[0]]

    # convert price data to returns
    if is_prices:
        returns = returns.pct_change()

    original_returns = returns.copy()

    # build monthly dataframe
    returns = pd.DataFrame(sum_returns(returns,
                                       returns.index.strftime('%Y-%m-01'),
                                       compounded))
    returns.columns = ['Returns']
    returns.index = pd.to_datetime(returns.index)

    # get returnsframe
    returns['Year'] = returns.index.strftime('%Y')
    returns['Month'] = returns.index.strftime('%b')

    # make pivot table
    returns = returns.pivot('Year', 'Month', 'Returns').fillna(0)

    # handle missing months
    for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        if month not in returns.columns:
            returns.loc[:, month] = 0

    # order columns by month
    returns = returns[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]

    if eoy:
        returns['eoy'] = sum_returns(original_returns,
                                     original_returns.index.year).values

    return returns

"""
-------------------------------------------포트폴리오 리스크 분석-------------------------------------------
"""
def skewness(returns):
    return returns.skew()

def kurtosis(returns):
    return returns.kurtosis()

def drawdown(returns):
    cumulative = (1+ returns).cumprod()
    highwatermark = cumulative.cummax()
    drawdown = (cumulative/ highwatermark) - 1    
    return drawdown

def max_drawdown(returns):
    return np.min(drawdown(returns))

def drawdown_duration(returns):
    dd = drawdown(returns)
    drawdownduration = list(chain.from_iterable((np.arange(len(list(j)))+1).tolist() if i == 1 else [0] * len(list(j)) for i, j in groupby(dd != 0)))
    drawdownduration = pd.DataFrame(drawdownduration)
    drawdownduration.index = returns.index
    return drawdownduration

def max_drawdown_duration(returns):
    return drawdown_duration(returns).max()[0]

def VaR(returns, percentile = 99):
    return returns.quantile(1-percentile /100)

def CVaR(returns, percentile = 99):
    return returns[returns < VaR(returns, percentile)].mean()

def print_risk(returns, percentile = 99):
    skew = skewness(returns)
    kurt = kurtosis(returns)
    mdd = max_drawdown(returns)
    ddur = max_drawdown_duration(returns)
    vvar = VaR(returns, percentile)
    cvar = CVaR(returns)
    
    risk_result = {
        'Skewness': skew,
        'Kurtosis': kurt,
        'MDD': mdd,
        'Drawdown Duration':ddur,
        'VaR': vvar,
        'CVaR': cvar
        }
    return risk_result


"""
-------------------------------------------포트폴리오 성과 분석-------------------------------------------
"""

    
def total_return(returns):
    return (1+returns).prod() - 1        

def geometric_return( returns):
    return (1+returns).prod() ** (252 / len(returns)) -1

def stdev(returns):
    return returns.std() * np.sqrt(252)

def downdev(returns, target = 0):
    returns = returns.copy()
    returns.loc[returns > target] = 0
    summation = (returns **2).sum()
    return np.sqrt(252 * summation / len(returns))

def updev(returns, target = 0):
    returns = returns.copy()
    returns.loc[returns > target] = 0
    summation = (returns **2).sum()
    return np.sqrt(252 * summation / len(returns))

def sharpe_ratio(returns):
    return (geometric_return(returns) ) /  stdev(returns)

def sortino_ratio(returns, target = 0):
    return (geometric_return(returns) ) /  downdev(returns)

def hit_ratio(returns):
    return (len(returns[returns>=0]) / len(returns))
                     
def print_performance(returns, target = 0):
    
    tr = total_return(returns)
    cagr = geometric_return(returns)
    std = stdev(returns)
    ddev =  downdev(returns, target)
    udev = updev(returns, target)
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    hit = hit_ratio(returns)
    
    performance_result = {
        'Total Return': tr,
        'Compound Return' : cagr,
        'Volatility' : std,
        'Downside Deviation' : ddev,
        'Upside Deviation' :udev,
        'Sharpe Ratio' : sharpe,
        'Sortino Ratio': sortino,
        'Hit Ratio': hit
        }
    
    return performance_result

"""
-------------------------------------------시각화-------------------------------------------
"""


def draw_historical_graph(series):
    fig = plt.figure(figsize=(8,6), dpi = 80)    
    ax1 = fig.add_subplot(111, ylabel = "Index")
    series.plot(ax = ax1, color= 'b', lw = 1., legend = True)
    plt.show()

def draw_monthly_return_statistics(series):

    close = series.resample("1M").ffill()
    
    periodic_return = close.pct_change().groupby\
        ([close.index.year, close.index.month]).\
            mean()
    periodic_return.iloc[0] = close.iloc[1] / 1000 - 1
        
    periodic_return_list = []
    
    for i in range(len(periodic_return)):
        periodic_return_list.append\
            ({'month': periodic_return.index[i][1],
              'monthly_return': periodic_return.values[i]})
            
    df_periodic_return = pd.DataFrame(periodic_return_list, columns = ('month','monthly_return'))
    
    fig = plt.figure(figsize=(8,6), dpi = 80)
    ax1 = fig.add_subplot(111, ylabel = "Return")
    fig2 = df_periodic_return.boxplot(ax = ax1, column = 'monthly_return', by = 'month')
    fig2.set(title='')
    fig.suptitle('Monthly Return Statistics Box Plot', fontsize = 14)
    plt.xlabel('Month', fontsize = 10)
    plt.ylabel('Monthly Return', fontsize = 10)
    plt.show()

  
def draw_monthly_return_plot(returns,
         title="Monthly Returns (%)\n",
         title_color="black",
         title_size=12,
         annot_size=10,
         figsize=(12,8),
         cmap='RdYlGn',
         cbar=True,
         square=False,
         compounded=True,
         eoy=False,
         ax=None):

    returns = monthly_return_table(returns, eoy=eoy, compounded=compounded)
    returns *= 100

    if figsize is None and ax is None:
        size = list(plt.gcf().get_size_inches())
        figsize = (size[0], size[0] // 2)
        plt.close()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax = sns.heatmap(returns, ax=ax, annot=True, center=0,
                     annot_kws={"size": annot_size},
                     fmt="0.2f", linewidths=0.5,
                     square=square, cbar=cbar, cmap=cmap)
    ax.set_title(title, fontsize=title_size,
                 color=title_color, fontweight="bold")

    if ax is None:
        fig.subplots_adjust(hspace=0)
        plt.yticks(rotation=0)
        plt.show()
        plt.close()

    plt.show()    

def draw_ticker_weight(dataframe):        
    fig = plt.figure(figsize=(8,6), dpi = 80)
    ax1 = fig.add_subplot(111, ylabel = "Weight")
    dataframe.plot(kind="area", ax=ax1, linewidth=0.01, alpha=0.9)
    ax1.set_ylabel("weight")
    ax1.set_xlabel("Date")
    plt.tight_layout()
    plt.show()


def draw_mdd_graph(dataframe_pf, dataframe_bm, fx_category):

    date = dataframe_pf.index
    
    if fx_category == "KRW":
        y0, y0_name = dataframe_pf['drawdown(KRW)'], "Evergreen"
        y1, y1_name = dataframe_bm['drawdown(KRW)'], "BM"
    else :
        y0, y0_name = dataframe_pf['drawdown'], "Evergreen"
        y1, y1_name = dataframe_bm['drawdown'], "BM"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date, y=y0,
                    mode='lines',
                    name=y0_name,
                    line = dict(color = "royalblue", width = 1)
                    ))
    fig.add_trace(go.Scatter(x=date, y=y1,
                    mode='lines',
                    name=y1_name,
                    line = dict(color = "firebrick", width = 1)
                    ))
    
    fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        #gridcolor = "white",
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor = "grey",
        zeroline=True,
        showline=True,
        showticklabels=True,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
        tickformat = '0.0%',
    ),
    autosize=False,
    margin=dict(
        autoexpand=True,
        l=100,
        r=20,
        t=110,
    ),
    legend=dict(
    #orientation="h",
    yanchor="bottom",
    y=0.99,
    xanchor="left",
    x=0.01
    ),
    showlegend=True,
    plot_bgcolor='white'
    )
    
    fig.show()

def draw_index_graph(dataframe_pf, fx_category):

    date = dataframe_pf.index
    
    if fx_category == "KRW":
        y0, y0_name = dataframe_pf['portfolio_index(KRW)'], "Evergreen"
    else :
        y0, y0_name = dataframe_pf['portfolio_index'], "Evergreen"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date, y=y0,
                    mode='lines',
                    name=y0_name))
    
    fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        #gridcolor = "white",
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=1,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor = "grey",
        zeroline=True,
        showline=True,
        showticklabels=True,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    autosize=False,
    margin=dict(
        autoexpand=True,
        l=100,
        r=20,
        t=110,
    ),
    legend=dict(
    #orientation="h",
    yanchor="bottom",
    y=0.99,
    xanchor="left",
    x=0.01
    ),
    showlegend=True,
    plot_bgcolor='white'
    )
    
    fig.show()
    
def draw_index_graph_with_bm(dataframe_pf, dataframe_bm, fx_category):

    date = dataframe_pf.index
    
    if fx_category == "KRW":
        y0, y0_name = dataframe_pf['portfolio_index(KRW)'], "Evergreen"
        y1, y1_name = dataframe_bm['portfolio_index(KRW)'], "BM"
    else :
        y0, y0_name = dataframe_pf['portfolio_index'], "Evergreen"
        y1, y1_name = dataframe_bm['portfolio_index'], "BM"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date, y=y0,
                    mode='lines',
                    name=y0_name))
    fig.add_trace(go.Scatter(x=date, y=y1,
                    mode='lines',
                    name=y1_name))
    
    fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        #gridcolor = "white",
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor = "grey",
        zeroline=True,
        showline=True,
        showticklabels=True,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    autosize=False,
    margin=dict(
        autoexpand=True,
        l=100,
        r=20,
        t=110,
    ),
    legend=dict(
    #orientation="h",
    yanchor="bottom",
    y=0.99,
    xanchor="left",
    x=0.01
    ),
    showlegend=True,
    plot_bgcolor='white'
    )
    
    fig.show()
"""
-------------------------------------------구글스프레드시트공유-------------------------------------------
"""    
            

def b2b_mp_post_googlespread(dataframe, sheet_seq, print_location):
    
    json_file_name = 'C:\\Users\\iruda\\Downloads\\gs-portfolio-management-6f6eab126bf1.json'

    #authorization
    gc = pygsheets.authorize(service_file=json_file_name)
    
    spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1zHQ2K4u07YV_siAHmoOHX0iLMdsUndJKVxDus6mrn_o/edit#gid=0'
    
    # 스프레스시트 문서 가져오기 
    doc = gc.open_by_url(spreadsheet_url)
    
    #select the sheet 
    wks = doc[sheet_seq]
    
    #update the first sheet with df, starting at cell B2. 
    dataframe = dataframe.reset_index().rename(columns={"index": "date"})
    wks.set_dataframe(dataframe,print_location)