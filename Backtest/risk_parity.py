# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 22:06:56 2022

@author: roby jeon
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import bt

from fig import post_info as info


def _cal_log_returns_series(config, max_first_date, max_last_date):
    """
    Parameters
    ----------
    config : yaml
        전략에 대한 기초 데이터 및 정보를 가지고 있는 공간
    max_first_date : Date
        모든 투자 유니버스의 데이터를 보유한 최초의 날짜
    max_last_date : Date
        마지막 조회 날짜
        
    Returns
    -------
    config에 입력받은 Paths에서 기초 데이터를 인식하여 로그 데이터로 전환하여 폴더 내 csv 형태로 반환한다

    """
    #set max date to yesterday
    df_merge = pd.DataFrame()
    
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)

    for ticker in tickers:
        price_df = pd.read_csv(data_path+ticker+'.csv')
        price_df['log_return'] = np.log(price_df.adjusted_close / price_df.adjusted_close.shift(1))
        price_df = price_df.loc[(price_df.date>=max_first_date) & (price_df.date<=max_last_date)]
        
        
        test = price_df[['date','log_return']]
        log_returns = test.set_index(['date']).iloc[:,0]
        
        df = log_returns.to_frame().rename(columns={"log_return": ticker})
        df.index.name = 'Index'

        df_merge = pd.concat([df_merge,df],axis = 1)
        
    df_merge = df_merge.dropna()
    df_merge.to_csv(data_path +'daily-log-returns.csv', header = True)
    
def _cal_log_returns_dataframe(config, dataframe, max_first_date, max_last_date):
    """
    Parameters
    ----------
    config : yaml
        전략에 대한 기초 데이터 및 정보를 가지고 있는 공간
    dataframe : dataframe
        종가 데이터 프레임
    max_first_date : Date
        모든 투자 유니버스의 데이터를 보유한 최초의 날짜
    max_last_date : Date
        마지막 조회 날짜
        
    Returns
    -------
    config에 입력받은 Paths에서 기초 데이터를 인식하여 로그 데이터로 전환하여 폴더 내 csv 형태로 반환한다

    """
    
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)

    price_df = dataframe
    price_df = np.log(price_df / price_df.shift(1))
    price_df = price_df.replace(np.nan, 0)
    price_df = price_df.loc[(price_df.index>=max_first_date) & (price_df.index<=max_last_date)]
    price_df.index.name = 'Index'
    
    price_df.to_csv(data_path +'daily-index-log-returns.csv', header = True)
        
    
def _get_log_returns_series(config, filename):
    """
    Parameters
    ----------
    config : yaml
        전략에 대한 기초 데이터 및 정보를 가지고 있는 공간
        
    Returns
    -------
    daily_log_returns : dataframe
        투자 유니버스에 대한 일별 로그 수익률 데이터 리턴한다

    """
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments =  info._post_type(config)
    daily_log_returns = pd.read_csv(data_path + filename + '.csv')
    
    return daily_log_returns


def _weight_sum_constraint(x) :
    """
    Parameters
    ----------
    x : float or int
        투자 비중이 100% 한도 내에서 집행될 수 있는 제약 조건

    Returns
    -------
    None.

    """
    return(x.sum() - 1.0 )


def _weight_longonly(x) :
    """
    Parameters
    ----------
    x : float or int
        숏 포지션을 잡지 않도록 하는 제약 조건

    Returns
    -------
    None.

    """
    return(x)


def _RiskParity(risk_budget, cov) :
    
    def __RiskParity_objective(x) :
        variance = x.T @ cov @ x
        sigma = variance ** 0.5
        mrc = (cov @ x)
        rc = x * mrc * 1/sigma
        a = np.reshape(rc, (len(rc), 1))
        risk_diffs = a - a.T
        sum_risk_diffs_squared = np.sum(np.square(np.ravel(risk_diffs)))
        return (sum_risk_diffs_squared)

    constraints = ({'type': 'eq', 'fun': _weight_sum_constraint},
                  {'type': 'ineq', 'fun': _weight_longonly})
    options = {'ftol': 1e-20, 'maxiter': 800}
    
    result = minimize(fun = __RiskParity_objective,
                      x0 = risk_budget,
                      method = 'SLSQP',
                      constraints = constraints,
                      options = options)
    return(result.x)


def _calc_risk_parity_weights(cov):
    """
    Parameters
    ----------
    cov : dataframe
        투자 유니버스에 대한 cov matrix를 입력값으로 받는다

    Raises
        부여된 리스크 기여도가 일치하지 않는다면 에러를 발생시킨다
    ------
    Exception
        예외 사항은 없고 모델을 중단한다

    Returns
    -------
    weights : float
        투자 비중을 역산하여 리턴한다
    risk_contributions : float
        포트폴리오 내 위혐 기여도를 역산하여 리턴한다

    """
    # create the desired risk budgeting vector (i.e. equal risk contributions)
    risk_budget = np.ones(len(cov)) / len(cov)
    # get the portfolio weights
    weights = _RiskParity(risk_budget, cov)
    
    # check risk contributions
    risk_contributions = (weights @ (cov * weights)) / np.sum((weights @ (cov * weights)))
    if (not np.array_equal(risk_contributions.round(2), risk_budget.round(2))):
        raise Exception('Error! the risk contributions =! the risk budget')
    return weights, risk_contributions

def _calc_sub_regime_index(config, max_first_date,max_last_date):
    
    #Get config data
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)
    
    df_prices = pd.read_csv(data_path+"index_prices"+'.csv', index_col = 0)
    df_prices.index  = pd.to_datetime(df_prices.index)
    
    runAfterDaysAlgo = bt.algos.RunAfterDays(
        lookback + 1
    )
      
    weighERCAlgo = bt.algos.WeighERC(
        lookback=pd.DateOffset(days=lookback),
        covar_method='standard',
        maximum_iterations=1000,
        tolerance=1e-9,
        lag=pd.DateOffset(days=1)
    )
    rebalAlgo = bt.algos.Rebalance()
    
    subportfolios = []
    for k in regimes.keys():
      df_price = df_prices[regimes[k]]
      subportfolios.append(
        bt.Backtest(bt.Strategy(k, [runAfterDaysAlgo, bt.algos.RunMonthly(run_on_first_date = False, run_on_end_of_period=True, run_on_last_date=False), bt.algos.SelectAll(), weighERCAlgo, rebalAlgo]), df_price,integer_positions=False)
      )
    
    res_target = bt.run(*subportfolios)
    
    #test = res_target.prices
    #test = res_target.get_transactions()
    #test = res_target.get_security_weights()
    #test = res_target.prices['RISING_GROWTH'].eq(100)
    
    df = res_target.prices[res_target.prices.index >= (res_target.prices.ne(res_target.prices.shift(-1)).apply(lambda x: x.index[x].tolist()).iloc[0][0])]
    df.reset_index(level=0, inplace=True)
    df.to_csv(data_path +'sub_portoflios_index.csv', index = False)

    bod = df['index'].iloc[0] # first element 
    bod = bod.strftime('%Y-%m-%d')
    
    eod = df['index'].iloc[-1] # last element 
    eod = eod.strftime('%Y-%m-%d')
    
    price_df = df.set_index('index')
    price_df = np.log(price_df / price_df.shift(1))
    price_df = price_df.replace(np.nan, 0)
    price_df = price_df.loc[(price_df.index>=max_first_date) & (price_df.index<=max_last_date)]
    price_df.index.name = 'Index'
    
    price_df.to_csv(data_path +'daily-subpf-log-returns.csv', header = True)
    
    return df, bod, eod


def _get_weights_by_rp(config, daily_log_returns, max_first_date, max_last_date):
    """
    Parameters
    ----------
    config : dictionary
        전략에 대한 기초 데이터 및 정보
    daily_log_returns : dataframe
        투자 유니버스에 대한 로그 수익률 데이터프레임
    max_first_date : Date
        모든 투자 유니버스의 데이터를 보유한 최초의 날짜
    max_last_date : Date
        마지막 조회 날짜

    Returns
    -------
    None.

    """
    #Get config data
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)

    #Create a covariance matrix
    cov = daily_log_returns.cov().to_numpy()
    
    #Take cov as input and calculate the capital weight %
    weights, risk_contributions = _calc_risk_parity_weights(cov)

    #Make df
    df = pd.DataFrame({ 'regime': daily_log_returns.columns[1:],
                        'weight': list(weights),
                        'risk_contribution': list(risk_contributions)})        
    return df

def _get_weights_within_regime(config, daily_log_returns, max_first_date, max_last_date):
    """
    Parameters
    ----------
    config : dictionary
        전략에 대한 기초 데이터 및 정보
    daily_log_returns : dataframe
        투자 유니버스에 대한 로그 수익률 데이터프레임
    max_first_date : Date
        모든 투자 유니버스의 데이터를 보유한 최초의 날짜
    max_last_date : Date
        마지막 조회 날짜

    Returns
    -------
    None.

    """
    #Get config data
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)
    
    #Create empty df
    df_merge = pd.DataFrame({'regime': [], 'ticker': [], 'weight': [],
                            'risk_contribution': []})

    for regime in regimes:
        #Create a covariance matrix
        cov = daily_log_returns[regimes[regime]].cov().to_numpy()
        
        #Take cov as input and calculate the capital weight %
        weights, risk_contributions = _calc_risk_parity_weights(cov)
        
        #Make df
        df = pd.DataFrame({ 'regime': regime,
                            'ticker': list(regimes[regime]),
                            'weight': list(weights),
                            'risk_contribution': list(risk_contributions)})
        df_merge = pd.concat([df_merge, df])
        df_merge = df_merge.reset_index(drop=True)
        df_merge.to_csv(result_path+'weights_within_regime.csv', index = False)
        
    return df_merge

def _get_weights_between_regimes(config, daily_log_returns, weights_within_regime, max_first_date, max_last_date):
    """
    Parameters
    ----------
    config : dictionary
        전략에 대한 기초 데이터 및 정보
    daily_log_returns : dataframe
        투자 유니버스에 대한 로그 수익률 데이터프레임
    weights_within_regime : dataframe
        주어진 국면 내에서 부여된 투자 비중
    max_first_date : Date
        모든 투자 유니버스의 데이터를 보유한 최초의 날짜
    max_last_date : Date
        마지막 조회 날짜

    Returns
    -------
    weights_between_regimes : dataframe
        프록시 버전으로 나오는 값으로, 국면간 리스크 패리티된 비중을 리턴한다

    """
    #Get config data
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)
    
    #Calculate simple returns for combinding into weighted portfolios
    daily_simple_returns = daily_log_returns.set_index('Index', drop = True)
    daily_simple_returns = daily_simple_returns.apply(np.exp)-1
    daily_simple_returns = daily_simple_returns.reset_index()
    
    #Create empty df
    df_merge = pd.DataFrame(index=daily_log_returns.index)
    for regime in regimes:
        w = weights_within_regime['weight'].\
            loc[weights_within_regime['regime'] == regime]
        R = daily_simple_returns[regimes[regime]]
        regime_simple_returns = R.to_numpy() @ w.to_numpy()
        df = pd.DataFrame({regime: regime_simple_returns}, index=daily_log_returns.index)
        df_merge = df_merge.join(df)
    df_merge.to_csv(result_path+'weighted-simple-returns-per-regime.csv')
    
    #Convert to log returns per regime:
    df_merge = df_merge + 1
    weighted_log_returns = df_merge.apply(np.log)
    weighted_log_returns.to_csv(result_path +'weighted-log-returns-per-regime.csv')
    
    #Create a covariance matrix using series of weighted log returns
    cov = weighted_log_returns.cov().to_numpy()
    
    #Take cov as input and calculate the capital weight %
    weights, risk_contributions = _calc_risk_parity_weights(cov)
    weights_between_regimes = pd.DataFrame({'regime': list(regimes.keys()),
                                            'weight': list(weights),
                                            'risk_contribution': list(risk_contributions)})
    weights_between_regimes.to_csv(result_path+'weights_between_regimes.csv', index = False)
    return weights_between_regimes

def _get_portfolio(config, weights_within_regime, weights_between_regimes, max_first_date):
    """
    Parameters
    ----------
    config : dictionary
        전략에 대한 기초 데이터 및 정보
    weights_within_regime : dataframe
        투자국면안에서 투자비중
    weights_between_regimes : dataframe
        서브-포트폴리오에 부여된 투자비중
    max_first_date : Date
        모든 투자 유니버스의 데이터를 보유한 최초의 날짜

    Returns
    -------
    portfolio : dataframe
        종목별 투자 비중과 리스크
        *해당 리스크는 산술적 리스크이며 실질 리스크는 아니다

    """

    #Get config data
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)
    
    temp_weights_within_regime = weights_within_regime.rename(columns={"weight": "ticker_weight","risk_contribution": "ticker_contribution"})
    temp_weights_within_regime = temp_weights_within_regime.set_index('regime')
                                 
    temp_weights_between_regimes = weights_between_regimes.rename(columns={"weight": "regime_weight","risk_contribution": "regime_contribution"})
    temp_weights_between_regimes = temp_weights_between_regimes.set_index('regime')
    
    df_merge = temp_weights_within_regime.join(temp_weights_between_regimes)
    df_merge['weight'] = df_merge['ticker_weight']*df_merge['regime_weight']
    df_merge['risk'] = df_merge['ticker_contribution']*df_merge['regime_contribution']
    
    portfolio = df_merge[['ticker','weight','risk']].groupby(['ticker']).sum()
    portfolio.to_csv(result_path+'model_portfolio.csv')

    return portfolio
    

def _get_subpf_index(config, max_first_date,max_last_date):
    
    #Get config data
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)
    
    df_prices = pd.read_csv(data_path+"pf_prices"+'.csv', index_col = 0)
    df_prices.index  = pd.to_datetime(df_prices.index)
    
    runAfterDaysAlgo = bt.algos.RunAfterDays(
        lookback + 1
    )
      
    weighERCAlgo = bt.algos.WeighERC(
        lookback=pd.DateOffset(days=lookback),
        covar_method='standard',
        maximum_iterations=1000,
        tolerance=1e-9,
        lag=pd.DateOffset(days=1)
    )
    rebalAlgo = bt.algos.Rebalance()
    
    subportfolios = []
    for k in regimes.keys():
      df_price = df_prices[regimes[k]]
      subportfolios.append(
        bt.Backtest(bt.Strategy(k, [runAfterDaysAlgo, bt.algos.RunMonthly(run_on_first_date = False, run_on_end_of_period=True, run_on_last_date=False), bt.algos.SelectAll(), weighERCAlgo, rebalAlgo]), df_price,integer_positions=False)
      )
    
    res_target = bt.run(*subportfolios)
    
    df = res_target.prices[res_target.prices.index >= (res_target.prices.ne(res_target.prices.shift(-1)).apply(lambda x: x.index[x].tolist()).iloc[0][0])]
    df.reset_index(level=0, inplace=True)
    df.to_csv(result_path +'sub_portoflios_index.csv', index = False)
    
    return df

def _RiskBudget(risk_budget, cov):

    def __RiskBudget_objective(x, cov, risk_budget):
        sum_risk_diffs_squared = np.sum(np.square((x*np.dot(cov, x)/np.dot(x.transpose(), np.dot(cov, x))-risk_budget)))    
        return sum_risk_diffs_squared

    w0 =  np.ones(len(cov)) / len(cov)
    constraints = ({'type': 'eq', 'fun': _weight_sum_constraint},
                  {'type': 'ineq', 'fun': _weight_longonly})
    options = {'ftol': 1e-20, 'maxiter': 800}
    
    result = minimize(fun = __RiskBudget_objective,
                      x0 = w0,
                      args=(cov, risk_budget),
                      method = 'SLSQP',
                      constraints = constraints,
                      options = options)
    return(result.x)

def _calc_risk_budget_weights(cov, risk_budget):

    # create the desired risk budgeting vector (i.e. equal risk contributions)
    risk_budget = risk_budget
    # get the portfolio weights
    weights = _RiskBudget(risk_budget, cov)
    
    # check risk contributions
    risk_contributions = (weights @ (cov * weights)) / np.sum((weights @ (cov * weights)))
    if (not np.array_equal(risk_contributions.round(2), risk_budget.round(2))):
        raise Exception('Error! The risk contributions do not match the risk budget')
    return weights, risk_contributions


def _get_riskbudget_weights(daily_log_returns, max_first_date, max_last_date, risk_budget):
   
    # create empty df
    df_merge = pd.DataFrame({'ticker': [], 'weight': [],'risk_contribution': []})

    # creates a covariance matrix 
    cov = daily_log_returns.cov().to_numpy()
    
    # take cov as input and calculate the capital weight %
    weights, risk_contributions = _calc_risk_budget_weights(cov, risk_budget)
    # make df
    df = pd.DataFrame({ 'ticker': list(daily_log_returns.columns[1:]),
                        'weight': list(weights),
                        'risk_contribution': list(risk_contributions)})
    df_merge = df_merge.append(df, sort=False)
    df_merge = df_merge.reset_index(drop=True)
    return df_merge


def _get_rolling_correlation_pf(config, daily_log_returns, weights_within_environment, max_last_date, window_size):

    #Get config data
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)
    
    weighted_log_returns = pd.read_csv(data_path+'weighted-log-returns-per-regime.csv')

    # create empty df
    df_merge = pd.DataFrame(index=daily_log_returns.index)
    for regime in regimes:
        for i in range(len(regimes[regime])):
            temp = weighted_log_returns[regime].rolling(window_size).corr(daily_log_returns[[regimes[regime][i]]])
            df_merge[regime+"_"+regimes[regime][i]]  = temp
    
    rolling_corr_within_regimes_pf = df_merge.set_index(daily_log_returns['Index'].str[:10]).dropna()        
    rolling_corr_within_regimes_pf.reset_index(level=0, inplace=True)
    
    rolling_corr_within_regimes_pf.to_csv(result_path+'rolling_corr_within_regimes_pf.csv', index = False)            
    
    return rolling_corr_within_regimes_pf


def _get_rolling_correlation_ticker(config, daily_log_returns, weights_within_environment, window_size):

    #Get config data
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)
    
    # create empty df
    df_merge = pd.DataFrame(index=daily_log_returns.index)
    for regime in regimes:
        for i in range(len(regimes[regime])):
            temp = daily_log_returns['VTI'].rolling(window_size).corr(daily_log_returns[[regimes[regime][i]]])
            df_merge[regime+"_"+regimes[regime][i]]  = temp
            
    rolling_corr_within_regime = df_merge.set_index(daily_log_returns['Index'].str[:10]).dropna()        
    rolling_corr_within_regime.reset_index(level=0, inplace=True)
    
    rolling_corr_within_regime.to_csv(result_path+'rolling_corr_within_environments.csv', index = False)            
    return rolling_corr_within_regime


def _get_historical_weight(config, daily_log_returns):
    
    #Get config data
    portfolio_name, lookback, tickers, benchmark_tickers, benchmark_ticker_weights, data_path, result_path, custom_data_list, regimes, instruments  =  info._post_type(config)
    
    #Read log data set up for loop
    df_daily_log_returns = daily_log_returns
    
    #Create Empty dataframe
    df_mp = pd.DataFrame()

    for i in range(1,len(df_daily_log_returns)-(lookback-1)):
        df_temp_log_returns = df_daily_log_returns.iloc[i-1:lookback+i,:]
    
        max_first_date = df_temp_log_returns['Index'].iloc[0]
        max_last_date = df_temp_log_returns.Index.iloc[-1]
        
        # calculate weights
        weights_within_regime = _get_weights_within_regime(config, df_temp_log_returns, max_first_date, max_last_date)
        weights_between_regimes = _get_weights_between_regimes(config, df_temp_log_returns, weights_within_regime, max_first_date, max_last_date)
        final_weights = _get_portfolio(config, weights_within_regime, weights_between_regimes, max_first_date)    
    
        temp_weights = final_weights[['weight']]
        temp_weights = temp_weights.rename(columns={'weight': max_last_date}).transpose()    
        
        df_mp = pd.concat([df_mp, temp_weights])
        print(str(i)+"/"+str((len(df_daily_log_returns)-(lookback))))
        
    return df_mp