# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:10:47 2022
model portfolio 구축 및 actual portfolio 조회, orderbook 생성, trading module 구성 
"""

# 필요 라이브러리 임포트
import pandas as pd
import numpy as np
import win32com.client
from pywinauto.application import Application
from pywinauto import Desktop
import os
import time
import datetime as dt
import ctypes
import pymysql
import sys

class Cybos:
    """ 사이보스 자동 로그인 및 복수계좌 연동"""
    def __init__(self):
        self.obj_CpUtil_CpCybos = win32com.client.Dispatch('CpUtil.CpCybos')
        self.obj_CodeMgr = win32com.client.Dispatch('CpUtil.CpCodeMgr') 
        self.objCpStatus = win32com.client.Dispatch('CpUtil.CpCybos') # 사이보스플러스 연결 상태 확인
        self.objCpTrade = win32com.client.Dispatch('CpTrade.CpTdUtil') # 주문 관련 
        self.objStockOrder = win32com.client.Dispatch('CpTrade.CpTd0311') # 주식 주문
        self.objCpTd6033 = win32com.client.Dispatch('CpTrade.CpTd6033') # 잔고 조회
        self.objStockMst = win32com.client.Dispatch('DsCbo1.StockMst') # 현재가, n차 매도,매수호가, 전일종가 조회
        self.objCpTd5339 = win32com.client.Dispatch('CpTrade.CpTd5339') # 미체결 데이터 조회
        self.objStockChart = win32com.client.Dispatch('CpSysDib.StockChart') # 수정주가
    
    def kill_client(self):
        os.system('taskkill /IM coStarter* /F /T')
        os.system('taskkill /IM CpStart* /F /T')
        os.system('taskkill /IM DibServer* /F /T')
        os.system('wmic process where "name like \'%coStarter%\'" call terminate')
        os.system('wmic process where "name like \'%CpStart%\'" call terminate')
        os.system('wmic process where "name like \'%DibServer%\'" call terminate')
    
    def connect(self, id_, pwd, pwdcert):
        if not self.connected():
            self.disconnect()
            self.kill_client()
            app = Application()
            app.start(
                'C:/DAISHIN/STARTER/ncStarter.exe /prj:cp /id:{id} /pwd:{pwd} /pwdcert:{pwdcert} /autostart'.format(
                    id=id_, pwd=pwd, pwdcert=pwdcert)
                )
        while not self.connected():
            d = Desktop(backend='uia')
            #try:
            #    d['보안토큰을 선택하여 주십시오'].child_window(title='취소', auto_id='2', control_type='Button').click_input(button='left')
            #except:
            #    pass
            # 키보드 보안 프로그램 해제하니까 잘됨 
            try:
                d['종합계좌 비밀번호 확인 입력'].Edit.type_keys('4689')
                d['종합계좌 비밀번호 확인 입력'].child_window(title='입력완료', auto_id='1', control_type='Button').click_input(button='left')
            except:
                pass
        return True
    
    def connected(self):
        b_connected = self.obj_CpUtil_CpCybos.IsConnect
        if b_connected == 0:
            return False
        return True
    
    def disconnect(self):
        if self.connected():
            self.obj_CpUtil_CpCybos.PlusDisconnect()
    
    def n_account_connect(self):
        """"복수계좌 연동->작업표시줄에서 자동 시행"""
        #d.windows() - 윈도우 실행영역 찾기
        #d['작업표시줄'].dump_tree()
        #print(main_tray_toolbar.texts())
        #app = Application(backend='uia').connect(path='C:/DAISHIN/CYBOSPLUS/CpStart.exe')
        #diag = app.window(title='복수계좌 사인온 변경')
        #diag.사인온.click()
        d = Desktop(backend='uia')
        # api 공지사항 창 닫기 
        d['공지사항'].child_window(title='닫기',control_type='Button').click_input(button='left')
        # 작업 표시줄 선택
        main_tray_toolbar = d.작업표시줄.child_window(title='사용자 지정 알림 영역', control_type='ToolBar')
        # 표시줄 우클릭
        main_tray_toolbar.child_window(title_re='CybosPlus Start', control_type='Button').click_input(button='right')
        # 복수계좌 사인온 클릭
        d.컨텍스트Menu.child_window(title='복수계좌 사인온 변경', control_type='MenuItem').click_input(button='left')
        # 전체선택 클릭
        d['복수계좌 사인온 변경'].child_window(title='전체선택', auto_id = '7001', control_type='Button').click_input(button='left')
        # 사인온 클릭
        d['복수계좌 사인온 변경'].child_window(title='사인온', auto_id= '7002', control_type='Button').click_input(button='left')
        # 복수계좌 변경 창 닫기
        d['복수계좌 사인온 변경'].child_window(title='닫기', control_type='Button').click_input(button='left')
    
    def plusinitcheck(self):
        if ctypes.windll.shell32.IsUserAnAdmin():
            print('정상: 관리자권한으로 실행된 프로세스입니다.')
        else:
            print('오류: 일반권한으로 실행됨. 관리자 권한으로 실행해 주세요')
            
        # PLUS가 연결이 되었는가?
        if (self.objCpStatus.IsConnect == 0):
            print('오류: PLUS가 정상적으로 연결되지 않음.')
        else:
            print('정상: PLUS가 정상적으로 연결됨')
        # 주문 관련 초기화 알림
        if (self.objCpTrade.TradeInit(0) != 0):
            print('오류: 주문 초기화 실패')
        else:
            print('정상: 주문 초기화')
    
    def OrderSell(self, acc, accflag, code, price, amount):
        self.objStockOrder.SetInputValue(0,'1') # 매도
        self.objStockOrder.SetInputValue(1, acc) # 주문할 계좌번호
        self.objStockOrder.SetInputValue(2, accflag) # 상품-주식
        self.objStockOrder.SetInputValue(3, code) # 해당 종목 코드
        self.objStockOrder.SetInputValue(4, amount) # 해당 종목 코드 거래 수량
        self.objStockOrder.SetInputValue(5, price) # 해당 종목 현재가
        self.objStockOrder.SetInputValue(7,'0') #  주문 조건 구분 코드, 0: 기본 1: IOC 2:FOK
        self.objStockOrder.SetInputValue(8,'01') # 주문호가 구분코드 - 01: 보통 / 03: 시장가
        
        # 매도 주문 요청
        self.objStockOrder.BlockRequest()
        
        rqStatus = self.objStockOrder.GetDibStatus()
        rqRet = self.objStockOrder.GetDibMsg1()
        print('통신상태', rqStatus, rqRet)
        if rqStatus != 0:
            sys.exit()
    
    def OrderBuy(self, acc, accflag, code, price, amount):
        self.objStockOrder.SetInputValue(0,'2') # 매도
        self.objStockOrder.SetInputValue(1, acc) # 주문할 계좌번호
        self.objStockOrder.SetInputValue(2, accflag) # 상품-주식
        self.objStockOrder.SetInputValue(3, code) # 해당 종목 코드
        self.objStockOrder.SetInputValue(4, amount) # 해당 종목 코드 거래 수량
        self.objStockOrder.SetInputValue(5, price) # 해당 종목 현재가
        self.objStockOrder.SetInputValue(7,'0') #  주문 조건 구분 코드, 0: 기본 1: IOC 2:FOK
        self.objStockOrder.SetInputValue(8,'01') # 주문호가 구분코드 - 01: 보통 / 03: 시장가
        
        # 매수 주문 요청
        self.objStockOrder.BlockRequest()
     
        rqStatus = self.objStockOrder.GetDibStatus()
        rqRet = self.objStockOrder.GetDibMsg1()
        print("통신상태", rqStatus, rqRet)
        if rqStatus != 0:
            sys.exit()
    
    def get_price(self):
        codes = ['주식코드 리스']
        df = pd.DataFrame(index=codes, columns=['names','ticker', 'cur_price','close_price','adj_price','5tick_sellprice','5tick_buyprice'])
        for code in df.index:
            self.objStockMst.SetInputValue(0, code)
            self.objStockMst.BlockRequest()
            
            self.objStockChart.SetInputValue(0, code)
            self.objStockChart.SetInputValue(1, ord('2')) # 개수로 조회
            self.objStockChart.SetInputValue(4, 1) # 전일 수정종가
            self.objStockChart.SetInputValue(5, [5]) # 종가
            self.objStockChart.SetInputValue(9, ord('1')) # 수정종가 여부
            self.objStockChart.BlockRequest()
        
            # 통신처리
            rqStatus = self.objStockMst.GetDibStatus()
            rqRet = self.objStockMst.GetDibMsg1()
            if rqStatus != 0:
                print('통신 오류')
            
            df['names'][code] = self.objStockMst.GetHeaderValue(1)
            df['ticker'][code] = code
            df['cur_price'][code] = self.objStockMst.GetHeaderValue(11) # 현재가
            df['close_price'][code] = self.objStockMst.GetHeaderValue(10) # 전일 종가
            df['adj_price'][code] = self.objStockChart.GetHeaderValue(6) # 전일 수정종가
            df['5tick_sellprice'][code] = self.objStockMst.GetDataValue(0,4) # 5차 매도호가 조회
            df['5tick_buyprice'][code] = self.objStockMst.GetDataValue(1,4) # 5차 매수호가 조회
            
            # 5호가 단위가 없는 경우 2호가로 변경
            if df['5tick_sellprice'][code] == 0:
                df['5tick_sellprice'][code] = self.objStockMst.GetDataValue(0,1)
            if df['5tick_buyprice'][code] == 0:
                df['5tick_buyprice'][code] = self.objStockMst.GetDataValue(1,1)
            
            # 종가 수정종가 차이 있을 시 빈 데이터 프레임 반환
            if df['close_price'][code] != df['adj_price'][code]:
                print('종가와 수정종가 차이 오류')
                df = pd.DataFrame()
            
        return df
    
    def get_aum(self):
        df = pd.DataFrame(columns=['acc_number','ticker','acc_name','aum','cash','val'])
        for i in range(len(self.objCpTrade.AccountNumber)):
            acc = self.objCpTrade.AccountNumber[i]
            accflag = self.objCpTrade.GoodsList(acc,1)
            
            self.objCpTd6033.SetInputValue(0, acc)
            self.objCpTd6033.SetInputValue(1, accflag[0])
            self.objCpTd6033.SetInputValue(2, 50)
            self.objCpTd6033.BlockRequest()
            time.sleep(1)
            
            acc_name = self.objCpTd6033.GetHeaderValue(0) # 계좌명
            aum = self.objCpTd6033.GetHeaderValue(3) # 총평가금액
            cash = self.objCpTd6033.GetHeaderValue(9) # D+2 예상예수금
            cnt = self.objCpTd6033.GetHeaderValue(7) # 계좌별 수신개수(종목수)
            
            for j in range(cnt):
                ticker = self.objCpTd6033.GetDataValue(12, j) # 종목코드
                val = self.objCpTd6033.GetDataValue(9, j) # 종목별 평가금액
                
                new = pd.DataFrame({'acc_number': [acc], 'ticker': [ticker],'acc_name': [acc_name], 'aum': [aum], 'cash': [cash], 'val': [val]})
                df = pd.concat([df, new])
            
        df = df.reset_index(drop=True)
        return df
    
    def get_aumlist(self):
        aumlist = []
        for i in range(len(self.objCpTrade.AccountNumber)):
            acc = self.objCpTrade.AccountNumber[i]
            accflag = self.objCpTrade.GoodsList(acc, 1)
            
            self.objCpTd6033.SetInputValue(0, acc)
            self.objCpTd6033.SetInputValue(1, accflag[0])
            self.objCpTd6033.SetInputValue(2, 50)
            self.objCpTd6033.BlockRequest()
            
            aum = self.objCpTd6033.GetHeaderValue(3)
            aumlist.append(aum)
            time.sleep(1)
        return aumlist
    
    def get_cashlist(self):
        cashlist = []
        for i in range(len(self.objCpTrade.AccountNumber)):
            acc = self.objCpTrade.AccountNumber[i]
            accflag = self.objCpTrade.GoodsList(acc,1)
            
            self.objCpTd6033.SetInputValue(0, acc)
            self.objCpTd6033.SetInputValue(1, accflag[0])
            self.objCpTd6033.SetInputValue(2, 50)
            self.objCpTd6033.BlockRequest()
            
            cash = self.objCpTd6033.GetHeaderValue(9)
            cashlist.append(cash)
        
        return cashlist
    
    def get_ap(self):
        total_ap = pd.DataFrame(columns=['acc_number','ticker','ticker_quantity'])
        
        for i in range(len(self.objCpTrade.AccountNumber)):
            acc = self.objCpTrade.AccountNumber[i]
            accflag = self.objCpTrade.GoodsList(acc,1)
            
            self.objCpTd6033.SetInputValue(0, acc)
            self.objCpTd6033.SetInputValue(1, accflag[0])
            self.objCpTd6033.SetInputValue(2,50)
            self.objCpTd6033.BlockRequest()
            
            cnt = self.objCpTd6033.GetHeaderValue(7)
            
            for j in range(cnt):
                ticker = self.objCpTd6033.GetDataValue(12,j)
                ticker_quantity = self.objCpTd6033.GetDataValue(7,j)
                
                new_data = pd.DataFrame({'acc_number': acc, 'ticker': ticker, 'ticker_quantity': [ticker_quantity]})
                total_ap = pd.concat([total_ap,new_data])
        total_ap = total_ap.reset_index(drop=True)
        total_ap = total_ap[['acc_number','ticker','ticker_quantity']]
        return total_ap
    
    def get_orderbook(self):
        orderbook = pd.merge(ap, mp, how='outer', left_on=['acc_number','ticker'], right_on=['acc_number','ticker'])
        orderbook = orderbook[['acc_number','ticker','weight','ticker_quantity','qty_final']]
        orderbook['weight'] = orderbook.weight.fillna(0)
        orderbook['qty_final'] = orderbook.qty_final.fillna(0).astype('int')
        orderbook['ticker_quantity'] = orderbook['ticker_quantity'].fillna(0).astype('int')
        orderbook['trade_quantity'] = orderbook['qty_final'] - orderbook['ticker_quantity']
        orderbook = orderbook.sort_values(by=['acc_number','ticker']).reset_index(drop=True)
        return orderbook

# 사이보스 로그인 연동
cybos = Cybos()
cybos.connect('daisin_id','daishin_pwd','daishin_cert_pwd')
cybos.n_account_connect()
cybos.plusinitcheck()

class pf_building():
    def __init__(self):
        """포트폴리오 구축을 위한 투자 유니버스 선정"""
        tickers = ['etf_code_list']
        
        etf_class = ['etf_class']
        
        df_univ = pd.DataFrame(index = tickers)
        df_univ['classification'] = etf_class
        data = cybos.get_price()
        df_univ['last_price'] = cybos.get_price()['adj_price'].T
        df_univ['cur_price'] = data.cur_price
        
        self.df_univ = df_univ.copy()
    
    def holdings(self, str_stk_exp, pf_aum, cho_strategy):
        df_univ = self.df_univ.copy()
        df_univ['tgt_mp_wt'] = np.nan
        
        list_strategy = ['strat1', 'strat2', 'strat3'] 
        strategy = list_strategy[cho_strategy - 1]
        
        df_coef = pd.DataFrame(index=list_strategy, columns=['coef_lev', 'coef_stk'])
        df_coef.loc['strat1'] = [0, 0]
        df_coef.loc['strat2'] = [0, 0]
        df_coef.loc['strat3'] = [0, 0]
        
        mp_tgt_lev = df_coef.loc[strategy].coef_lev * str_stk_exp
        #mp_tgt_lev = df_coef.iloc[0].coef_lev * str_stk_exp
        
        mp_tgt_stk = df_coef.loc[strategy].coef_stk * str_stk_exp 
        mp_tgt_dlr = (1 - str_stk_exp) *(2*df_coef.loc[strategy,].coef_lev + df_coef.loc[strategy,].coef_stk)
        mp_tgt_mmf = 1-mp_tgt_lev -mp_tgt_stk -mp_tgt_dlr 
                
        df_univ.loc[df_univ.classification =='lev', 'tgt_mp_wt'] = mp_tgt_lev 
        df_univ.loc[df_univ.classification =='stk', 'tgt_mp_wt'] = mp_tgt_stk  
        df_univ.loc[df_univ.classification =='dlr', 'tgt_mp_wt'] = mp_tgt_dlr 
        df_univ.loc[df_univ.classification =='usb', 'tgt_mp_wt'] = mp_tgt_dlr 
        df_univ.loc[df_univ.classification =='mmf', 'tgt_mp_wt'] = mp_tgt_mmf 
        
        df_univ['qty_float'] = pf_aum * df_univ.tgt_mp_wt / df_univ.cur_price
        df_univ['qty_integ'] = df_univ.qty_float.apply(lambda x: round(x, 0))
        df_univ['qty_final'] = df_univ.qty_integ
        
        df_univ.loc['A122630', 'qty_final'] = max(1, df_univ.loc['A122630','qty_final'])
        
        while sum(df_univ.qty_final * df_univ.cur_price) > pf_aum:
            top = df_univ.sort_values(by='qty_final', ascending=False).index[0]
            df_univ.loc[top,'qty_final'] = max(df_univ.loc[top,'qty_final']-1, 0)
        
        self.df_univ = df_univ.copy()
        self.strategy = df_coef.iloc[cho_strategy-1,]
        
        return self.df_univ, self.strategy

def df_result(str_stk_exp, product_id, created_at, set_aum, set_id, set_type, set_account):
    pf_build = pf_building()
    
    df_dbaws = pd.DataFrame(columns=['product_id', 'portfolio_id', 'created_at', 'symbol', 'weight'])
    df_trade = pd.DataFrame(columns=['type', 'acc_number', 'ticker', 'weight','qty_final'])
    
    df_pf = pd.DataFrame(columns=['classification', 'last_price', 'cur_price', 'tgt_mp_wt', 'qty_float',
           'qty_integ', 'qty_final', 'final_tgt_wt'])
    df_strat = pd.DataFrame(columns=['JGTJ', 'JLTZ', 'AJTZ', 'AJHH', 'CGHH'])
    for i in range(len(set_id)):
        pf_aum = cybos.get_aumlist()[i]
        cho_type = set_type[i]
        pf, strat = pf_build.holdings(str_stk_exp, pf_aum, cho_type)
        df_pf = pd.concat([df_pf,pf])
        df_strat = pd.concat([df_strat,strat])
        
        df_i = pf[['tgt_mp_wt','qty_final']].reset_index().copy()
        
        df_i['product_id'] = product_id
        df_i['portfolio_id'] = set_id[i]
        df_i['created_at'] = created_at
        df_i['type'] = strat.name
        df_i['acc_number'] = set_account[i]
        
        df_i_db = df_i[['product_id', 'portfolio_id', 'created_at', 'index', 'tgt_mp_wt']].copy()
        df_i_db.columns = df_dbaws.columns
        df_dbaws = pd.concat([df_dbaws, df_i_db])
        
        df_i_tr = df_i[['type', 'acc_number', 'index', 'tgt_mp_wt','qty_final']].copy()
        df_i_tr.columns = df_trade.columns
        df_trade = pd.concat([df_trade, df_i_tr])
    
    return df_pf, df_strat, df_dbaws, df_trade


df_pf, df_strat, df_dbaws, df_trade = df_result(str_stk_exp, product_id, created_at, set_aum, set_id, set_type, set_account)

def mp_join():
    global df_dbaws
    pf_type = pd.read_excel('local 위치' +'.xlsx', header = 0, sheet_name='pf_type')
    pf_type2 = pd.read_excel('local 위치' +'.xlsx', header = 0, sheet_name='pf_type2')
    ticker_= pd.read_excel('local 위치' +'.xlsx', header = 0, sheet_name='ticker')

    df_trade['acc_number'] = df_trade['acc_number'].astype('str').str.strip()

    pf_type['acc_number'] = pf_type['acc_number'].str.replace('-','')
    pf_type['acc_number'] = pf_type['acc_number'].str.strip()

    df = df_trade.join(pf_type.set_index('acc_number'), on='acc_number')
    df = df.join(ticker_.set_index('ticker'),on='ticker')
    df.weight = df_dbaws.weight
    df = df.loc[:,['type','acc_number','ticker','weight','qty_final','type1','type2','종목코드(ISIN코드)','종목명']]
    
    return df

# 주문내역 생성 - AP, MP, price
ap = cybos.get_ap()
mp = df_trade.copy()
orderbook = cybos.get_orderbook()

# 주문
def Order(orderbook):
    for i in range(len(cybos.objCpTrade.AccountNumber)):
        acc = cybos.objCpTrade.AccountNumber[i] # 계좌번호
        accflag = cybos.objCpTrade.GoodsList(acc, 1) # 주식상품
        
        price = cybos.get_price()
        order = orderbook.join(price.set_index('ticker'), on='ticker')
        order = order[order['acc_number'] == acc]
        order['accflag'] = accflag[0]
        order = order.loc[:,['acc_number','accflag','ticker','trade_quantity','5tick_sellprice','5tick_buyprice']]
        
        # 매도
        sellorder = order[order['trade_quantity'] < 0]
        sellorder['trade_quantity'] = abs(sellorder['trade_quantity'])
        sellcode = sellorder['ticker']
        sellamount = sellorder['trade_quantity']
        sellprice = sellorder['5tick_buyprice']
        sellacc = sellorder['acc_number']
        sellaccflag = sellorder['accflag']
        
        # 매수 
        buyorder = order[order['trade_quantity'] > 0]
        buycode = buyorder['ticker']
        buyamount = buyorder['trade_quantity']
        buyprice = buyorder['5tick_sellprice']
        buyacc = buyorder['acc_number']
        buyaccflag = buyorder['accflag']
        
        # 매도 주문
        for a, b, c, d, e in zip(sellacc, sellaccflag, sellcode, sellprice, sellamount):
            time.sleep(1)
            cybos.OrderSell(a,b,c,d,e)
        
        # 매도 주문 후 매도 주문이 완료될 때까지 매수 주문 홀드
        #objCpTd5339 = win32com.client.Dispatch('CpTrade.CpTd5339')
        cybos.objCpTd5339.SetInputValue(0, acc)
        cybos.objCpTd5339.SetInputValue(1, accflag[0])
        cybos.objCpTd5339.SetInputValue(7, 20) # 요청 개수 - 최대 20개
        print('미체결 데이터 조회 시작')
    
        while True:
            ret = cybos.objCpTd5339.BlockRequest()
            if cybos.objCpTd5339.GetDibStatus() != 0:
                print("통신상태", cybos.objCpTd5339.GetDibStatus(), cybos.objCpTd5339.GetDibMsg1())
                if (ret == 2 or ret == 3):
                    print("통신 오류", ret)
    
            # 통신 초과 요청 방지에 의한 요류 인 경우
            while (ret == 4) : # 연속 주문 오류 임. 이 경우는 남은 시간동안 반드시 대기해야 함.
                remainTime = cybos.objCpStatus.LimitRequestRemainTime
                #print("연속 통신 초과에 의해 재 통신처리 : ", remainTime/1000, "초 대기" )
                time.sleep(remainTime / 1000)
                ret = cybos.objCpTd5339.BlockRequest()
        
            # 수신개수
            unconcluded = cybos.objCpTd5339.GetHeaderValue(5) # 미체결 수신개수
            print('미체결 수신 개수', unconcluded)
            if unconcluded == 0:
                print('매도 주문 체결 완료 혹은 매도 주문이 없습니다.')
                break
    
        # 매수 주문
        for a, b, c, d, e in zip(buyacc, buyaccflag, buycode, buyprice, buyamount):
            time.sleep(1)
            cybos.OrderBuy(a,b,c,d,e)

Order(orderbook)


ap = cybos.get_ap()
final_orderbook = cybos.get_orderbook()
#final_orderbook['add_order'] = abs(final_orderbook['trade_quantity'])
#final_orderbook['trade_quantity'] = final_orderbook['ticker_quantity'] - (final_orderbook['qty_final'] + final_orderbook['add_order'])
final_aum = cybos.get_aum()
final_orderbook = final_orderbook.join(final_aum.set_index(['acc_number','ticker']), on =['acc_number','ticker'])
price = cybos.get_price()
final_orderbook = final_orderbook.join(price.set_index('ticker'), on ='ticker')
final_orderbook['ap'] = final_orderbook.val / final_orderbook.aum
final_orderbook['ap'] = final_orderbook.ap.fillna(0)
