from cmath import nan
import win32com.client
import time
import pandas as pd
import numpy as np
import ctypes
from datetime import date
import math
import sys

### 대신증권 트레이딩 자동화 
## excel 데이터를 가져와서 주문이 되게끔 구현 

## PLUS 접속 확인 실행 체크
ObjCodeMgr = win32com.client.Dispatch('CpUtil.CpCodeMgr')
ObjCpStatus = win32com.client.Dispatch('CpUtil.CpCybos')
ObjCpTrade = win32com.client.Dispatch('CpTrade.CpTdUtil')

# 프로세스가 관리자 권한으로 실행되었는가?
if ctypes.windll.shell32.IsUserAnAdmin():
    print('정상: 관리자권한으로 실행된 프로세스입니다.')
else:
    print('오류: 일반권한으로 실행됨. 관리자 권한으로 실행해 주세요')
    
# PLUS가 연결이 되었는가?
if (ObjCpStatus.IsConnect == 0):
    print('PLUS가 정상적으로 연결되지 않음.')

# 주문 관련 초기화 알림
if (ObjCpTrade.TradeInit(0) != 0):
    print('주문 초기화 실패')

####################### MP 내역 불러오기 ##############################
MP_df = pd.read_excel('local 위치'+ '.xlsx', header=0)
pf_type = pd.read_excel('local 위치'+ '.xlsx', header=0, sheet_name='pf_type')
ticker = pd.read_excel('local 위치'+ '.xlsx', header=0, sheet_name='ticker')

# data transformation
join_total_df = MP_df.copy()
join_total_df['ticker'] = 'A' + join_total_df['종목코드'].str.split('.').str.get(0)
join_total_df = join_total_df.join(pf_type.set_index('계좌번호'), on='계좌번호')
join_total_df = join_total_df.join(ticker.set_index('ticker'), on='ticker')

MP_df['계좌번호'] = MP_df['계좌번호'].astype('str').str.strip()
MP_df['계좌번호'] = MP_df['계좌번호'].str.replace('-','')

MP_df = MP_df.loc[:,['타입','계좌번호','종목코드','종목비중']]

# 종목비중 이상 test
group_weight = MP_df.groupby(['계좌번호']).sum()
for i in range(len(group_weight)):
    if math.isclose(group_weight['종목비중'][i] , 1):
        print(group_weight.index[i],'MP비중에 이상이 없습니다. 잔고명세내역과 오더북을 확인해주세요')
    else:
        print(group_weight.index[i],'MP비중에 이상이 있습니다. 코드를 강제로 종료합니다.')
        sys.exit(0)
####################################################################

total_df = pd.DataFrame()
total_ob_df = pd.DataFrame()
acc_list = []
accflag_list = []

# 총매수,매도 값 반환 함수
def func(ob_df):
    if ob_df['총매수,매도량'] > 0:
        val = '매수'
    elif ob_df['총매수,매도량'] == 0:
        val = '없음'
    else:
        val = '매도'
    return val

## 복수 계좌 조희
for i in range(len(ObjCpTrade.AccountNumber)):
    acc = ObjCpTrade.AccountNumber[i] # 계좌번호
    acc_list.append(acc)
    accflag = ObjCpTrade.GoodsList(acc, 1) # 주식상품
    accflag_list.append(accflag_list)
    
    # 계좌에 일치하는 MP_df 필터링
    MP_df_ = MP_df[MP_df['계좌번호'] == acc]
    
    # 잔고조회 메서드
    CpTd6033 = win32com.client.Dispatch('CpTrade.CpTd6033')
    CpTd6033.SetInputValue(0, acc) # 계좌 input
    CpTd6033.SetInputValue(1, accflag[0]) # 잔고 상품 구분 input
    CpTd6033.SetInputValue(2, 50) # 요청 개수 input
    CpTd6033.BlockRequest()
    
    acc_name = CpTd6033.GetHeaderValue(0) # 계좌명
    ctime = date.today() #날짜
    cnt = CpTd6033.GetHeaderValue(7) # 잔고 보유종목 수신개수
    
    # 잔고평가금액 - HeaderValue(3)은 부정확 data. 사용 x
    예수금 = CpTd6033.GetHeaderValue(9) # D+2 예수금
    대주평가금액 = CpTd6033.GetHeaderValue(10)
    대출금액 = CpTd6033.GetHeaderValue(6) 
    대주금액 = CpTd6033.GetHeaderValue(12)  
    
    df = pd.DataFrame(columns=['계좌명','account','출력시간','총평가금액','평가금액합','D+2예수금','종목코드','종목명','현재가','체결잔고수량','평가금액','평가금액비중'])
    
    # 현재가 조회 메서드
    objStockMst = win32com.client.Dispatch('DsCbo1.StockMst')
    
    ## 잔고명세내역 dataframe 형성
    # 잔고에 보유종목이 없을 경우
    if cnt == 0:
        df = df.append(pd.DataFrame([[acc_name, acc, ctime, 예수금-대주평가금액-대출금액+대주금액 ,0 ,예수금,np.nan, np.nan, np.nan, np.nan, np.nan ,np.nan]], columns=['계좌명','account','출력시간','총평가금액','평가금액합','D+2예수금','종목코드','종목명','현재가','체결잔고수량','평가금액','평가금액비중']))
        df.loc[0, '평가금액합'] = 0
        df.loc[0, '총평가금액'] = 0-대주평가금액+예수금-대출금액+대주금액 
        df = df.reset_index()
    # 잔고에 보유종목이 있을 경우
    else:
        for i in range(cnt):
            종목코드 = CpTd6033.GetDataValue(12,i)
            종목명 = CpTd6033.GetDataValue(0,i)
            체결잔고수량 = CpTd6033.GetDataValue(7,i)

            # 현재가 요청
            objStockMst.SetInputValue(0, 종목코드)
            objStockMst.BlockRequest()
            
            # 통신 처리
            rqStatus = objStockMst.GetDibStatus()
            rqRet = objStockMst.GetDibMsg1()
            if rqStatus != 0:
                print('통신 오류')
            
            현재가 = objStockMst.GetHeaderValue(11)
            평가금액 = 체결잔고수량 * 현재가
            if i == 0:
                df = df.append(pd.DataFrame([[acc_name, acc, ctime, 예수금-대주평가금액-대출금액+대주금액, 예수금, 종목코드, 종목명, 체결잔고수량, 현재가, 평가금액]], columns=['계좌명','account','출력시간','총평가금액','D+2예수금','종목코드','종목명','체결잔고수량','현재가','평가금액'])) 
            else:
                df = df.append(pd.DataFrame([[종목코드, 종목명, 체결잔고수량, 현재가, 평가금액]], columns=['종목코드','종목명','체결잔고수량','현재가','평가금액']))
        df = df.reset_index()
        평가금액합 = sum(df['평가금액'])
        df.loc[0, '총평가금액'] = 평가금액합-대주평가금액+예수금-대출금액+대주금액 
        df.loc[0, '평가금액합'] = 평가금액합
        df['평가금액비중'] = df['평가금액'] / df['총평가금액'][0]
    
    df = df[['계좌명','account','출력시간','총평가금액','평가금액합','D+2예수금','종목코드','종목명','현재가','체결잔고수량','평가금액','평가금액비중']]
    
    #------------------------------------------------------------------------------------------------------orderbook dataframe------------------------------------------------------------------------------------------
    ob_df = pd.DataFrame(columns=['계좌명','account','출력시간','타입','계좌번호','총평가금액','평가금액합','D+2예수금','종목코드','종목명','현재가','체결잔고수량','평가금액','평가금액비중','종목비중','비중구분','목표금액','목표금액-평가금액','매수,매도량','주문구분','5차매도호가','5차매수호가','예상보유비중','예상예수금','추가매수량','총매수,매도량','최종주문구분'])
    
    # 잔고에 보유종목이 없을 경우
    if cnt == 0:
        ob_df = pd.merge(df, MP_df_, how='right', on = '종목코드')
        ob_df.loc[0,'계좌명'] =  acc_name
        ob_df.loc[0,'account'] = acc
        ob_df.loc[0,'출력시간'] = ctime
        ob_df.loc[0, 'D+2예수금'] = 예수금
        ob_df.loc[0,'총평가금액'] = df.loc[0,'총평가금액']
        ob_df.loc[0,'평가금액합'] = df.loc[0,'평가금액합']
        ob_df['평가금액'] = 0
        ob_df = ob_df.dropna(axis=0, how = 'all', subset=['종목코드'])
        ob_df = ob_df.reset_index()
    # 잔고에 보유종목이 있을 경우
    else:
        ob_df = pd.merge(df, MP_df_, how='outer', on = '종목코드') # 현재 MP와 acc가 일치 안하므로 유의
        ob_df = ob_df.reset_index()
    ob_df['체결잔고수량'] = ob_df['체결잔고수량'].fillna(0)
    ob_df['평가금액'] = ob_df['평가금액'].fillna(0)
    ob_df['평가금액비중'] = ob_df['평가금액비중'].fillna(0)
    ob_df['종목비중'] = ob_df['종목비중'].fillna(0)
    ob_df = pd.DataFrame(ob_df, columns=['계좌명','account','출력시간','타입','계좌번호','총평가금액','평가금액합','D+2예수금','종목코드','종목명','현재가','체결잔고수량','평가금액','평가금액비중','종목비중','비중구분','목표금액','목표금액-평가금액','매수,매도량','주문구분','5차매도호가','5차매수호가','예상보유비중','예상예수금','추가매수량','총매수,매도량','최종주문구분'])
    ob_df = ob_df[['계좌명','account','출력시간','타입','계좌번호','총평가금액','평가금액합','D+2예수금','종목코드','종목명','현재가','체결잔고수량','평가금액','평가금액비중','종목비중','비중구분','목표금액','목표금액-평가금액','매수,매도량','주문구분','5차매도호가','5차매수호가','예상보유비중','예상예수금','추가매수량','총매수,매도량','최종주문구분']]
    
    #### orderbook dataframe
    for i in range(len(ob_df)):
        codelist = ob_df['종목코드'][i]
        
        objStockMst.SetInputValue(0, codelist)
        objStockMst.BlockRequest()

        # 통신 처리
        rqStatus = objStockMst.GetDibStatus()
        rqRet = objStockMst.GetDibMsg1()
        if rqStatus != 0:
            print('통신 오류')
        
        ob_df['종목명'][i] = objStockMst.GetHeaderValue(1)
        ob_df['현재가'][i] = objStockMst.GetHeaderValue(11)
        ob_df['5차매도호가'][i] = objStockMst.GetDataValue(0,4) # 5차매도호가
        ob_df['5차매수호가'][i] = objStockMst.GetDataValue(1,4) # 5차매수호가
        
        if np.isnan(ob_df['평가금액'][i] / ob_df['총평가금액'][0]):
            ob_df['평가금액비중'][i] = 0
        else:
            ob_df['평가금액비중'][i] = ob_df['평가금액'][i] / ob_df['총평가금액'][0]
        
        if ob_df['평가금액비중'][i] > ob_df['종목비중'][i]:
            ob_df['비중구분'][i] = 'overweight'
        else:
            ob_df['비중구분'][i] = 'underweight'
        
        ob_df['목표금액'][i] = ob_df['총평가금액'][0] * ob_df['종목비중'][i]
        ob_df['목표금액-평가금액'][i] = ob_df['목표금액'][i] - ob_df['평가금액'][i]

        # 매수,매도량
        ob_df['매수,매도량'][i] = math.floor(ob_df['목표금액-평가금액'][i] / ob_df['현재가'][i])
        
        # 매수,매도 의사결정
        if ob_df['매수,매도량'][i] > 0:
            ob_df['주문구분'][i] = '매수'
        elif ob_df['매수,매도량'][i] == 0:
            ob_df['주문구분'][i] = '없음'
        else:
            ob_df['주문구분'][i] = '매도'
        
        # 예상 보유비중
        if np.isnan( (ob_df['현재가'][i] * ob_df['매수,매도량'][i] + ob_df['평가금액'][i]) / ob_df['총평가금액'][0] ) :
            ob_df['예상보유비중'][i] = 0
        else:
            ob_df['예상보유비중'][i] = (ob_df['현재가'][i] * ob_df['매수,매도량'][i] + ob_df['평가금액'][i]) / ob_df['총평가금액'][0]
        
        ## 추가 매수, 매도량 구하기
        ob_df['예상예수금'] = 0 
        ob_df['예상예수금'][0] = (1 - sum(ob_df['예상보유비중'])) * ob_df['총평가금액'][0]
        cash = ob_df['예상예수금'][0]
        ob_df['추가매수량'] = 0
    
        # 예수금이 10000원 되기 전까지 종목 추가 매수
        while cash >= 10000:
            for i in range(len(ob_df)):
                cash = cash - ob_df['현재가'][i]
                if cash >= 10000:
                    ob_df['추가매수량'][i] += 1
                else:
                    break
        ob_df['총매수,매도량'] = ob_df['매수,매도량'] + ob_df['추가매수량']
        ob_df['최종주문구분'] = ob_df.apply(func, axis=1)
    #------------------------------------------------------------------------------------------------------orderbook dataframe------------------------------------------------------------------------------------------
    
    # 최종 주문구분
    total_df = total_df.append(df)
    total_ob_df = total_ob_df.append(ob_df)

with pd.ExcelWriter('orderbookfinal_n {}-{}-{}.xlsx'.format(ctime.year,ctime.month,ctime.day)) as writer:
    total_df.to_excel(writer, index=False, sheet_name='잔고명세내역')
    total_ob_df.to_excel(writer, index=False, sheet_name='주문지')

######### 주식 매도, 매수 함수 선언 ###########################
def OrderSell(acc, code, price, amount):
    objStockOrder = win32com.client.Dispatch('CpTrade.CpTd0311')
    objStockOrder.SetInputValue(0,'1') # 매도
    objStockOrder.SetInputValue(1, acc) # 주문할 계좌번호
    objStockOrder.SetInputValue(2, accflag[0]) # 상품-주식
    objStockOrder.SetInputValue(3, code) # 해당 종목 코드
    objStockOrder.SetInputValue(4, amount)   # 해당 종목 코드 매도 수량
    objStockOrder.SetInputValue(5, price) # 해당 종목 현재가
    objStockOrder.SetInputValue(7, "0")   #  주문 조건 구분 코드, 0: 기본 1: IOC 2:FOK
    objStockOrder.SetInputValue(8, "01")   # 주문호가 구분코드 - 01: 보통 / 03: 시장가
    
    # 매도 주문 요청
    objStockOrder.BlockRequest()
    
    rqStatus = objStockOrder.GetDibStatus()
    rqRet = objStockOrder.GetDibMsg1()
    print("통신상태", rqStatus, rqRet)
    if rqStatus != 0:
        exit()

def OrderBuy(acc, code, price ,amount):    
    objStockOrder = win32com.client.Dispatch("CpTrade.CpTd0311")
    objStockOrder.SetInputValue(0, "2")   # 2: 매수
    objStockOrder.SetInputValue(1, acc )   #  계좌번호
    objStockOrder.SetInputValue(2, accflag[0])   # 상품구분 - 주식 상품 중 첫번째
    objStockOrder.SetInputValue(3, code)   # 해당 종목 코드
    objStockOrder.SetInputValue(4, amount)   # 해당 종목 코드 매수 수량
    objStockOrder.SetInputValue(5, price) # 해당 종목 현재가
    objStockOrder.SetInputValue(7, "0")   # 주문 조건 구분 코드, 0: 기본 1: IOC 2:FOK
    objStockOrder.SetInputValue(8, "01")   # 주문호가 구분코드 - 01: 보통 / 03: 시장가

    # 매수 주문 요청
    objStockOrder.BlockRequest()
 
    rqStatus = objStockOrder.GetDibStatus()
    rqRet = objStockOrder.GetDibMsg1()
    print("통신상태", rqStatus, rqRet)
    if rqStatus != 0:
        exit()
#####################################################################
########################################계좌별 주문 #########################################################
for i in range(len(ObjCpTrade.AccountNumber)):
    acc = ObjCpTrade.AccountNumber[i] # 계좌번호
    accflag = ObjCpTrade.GoodsList(acc, 1) # 주식상품
    
    order = total_ob_df[total_ob_df['계좌번호'] == acc]
    order = order.loc[:,['계좌번호','종목코드','총매수,매도량','5차매도호가','5차매수호가']]
    
    # 매도
    sellorder = order[order['총매수,매도량'] < 0]
    sellorder['총매수,매도량'] = abs(sellorder['총매수,매도량'])
    sellcode = sellorder['종목코드']
    sellamount = sellorder['총매수,매도량']
    sellprice = sellorder['5차매수호가']
    sellacc = sellorder['계좌번호']
    
    # 매수
    buyorder = order[order['총매수,매도량'] > 0]
    buycode = buyorder['종목코드']
    buyamount = buyorder['총매수,매도량']
    buyprice = buyorder['5차매도호가']
    buyacc = buyorder['계좌번호']
    
    # 매도 주문
    for a, b, c, d in zip(sellacc, sellcode, sellprice, sellamount):
        time.sleep(1) 
        OrderSell(a, b, c, d)
    
    # 매도 주문 후 매도 주문이 완료될 때까지 매수 주문 홀드
    objCpTd5339 = win32com.client.Dispatch('CpTrade.CpTd5339')
    objCpTd5339.SetInputValue(0, acc)
    objCpTd5339.SetInputValue(1, accflag[0])
    objCpTd5339.SetInputValue(7, 20) # 요청 개수 - 최대 20개
    print('미체결 데이터 조회 시작')
    
    while True:
        ret = objCpTd5339.BlockRequest()
        if objCpTd5339.GetDibStatus() != 0:
            print("통신상태", objCpTd5339.GetDibStatus(), objCpTd5339.GetDibMsg1())
        if (ret == 2 or ret == 3):
            print("통신 오류", ret)
    
        # 통신 초과 요청 방지에 의한 요류 인 경우
        while (ret == 4) : # 연속 주문 오류 임. 이 경우는 남은 시간동안 반드시 대기해야 함.
            remainTime = ObjCpStatus.LimitRequestRemainTime
            #print("연속 통신 초과에 의해 재 통신처리 : ", remainTime/1000, "초 대기" )
            time.sleep(remainTime / 1000)
            ret = objCpTd5339.BlockRequest()
        
        # 수신개수
        unconcluded = objCpTd5339.GetHeaderValue(5) # 미체결 수신개수
        print('미체결 수신 개수', unconcluded)
        if unconcluded == 0:
            print('매도 주문 체결 완료 혹은 매도 주문이 없습니다.')
            break
    
    # 매수 주문
    for a, b, c, d in zip(buyacc, buycode, buyprice, buyamount):
        time.sleep(1)
        OrderBuy(a, b, c, d)