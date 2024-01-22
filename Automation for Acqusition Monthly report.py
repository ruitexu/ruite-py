# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:11:38 2020

@author: rxu
"""

#all the package needed
import pandas as pd
import pyodbc
#import datetime

#How to Connect Python and SQL Server

def acquisition(banner, country, year, month, aftermonth, channel):

    conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                          "Server=HIDDEN;"
                          "Database=SandBox;"
                          "Trusted_Connection=yes;")
    if channel == 'Ecom':
        channel1 = 'Ecom'
        channel2 = 'Ecom'
    elif channel == 'B&M':
        channel1 = 'B&M'
        channel2 = 'B&M'
    else:
        channel1 = 'Ecom'
        channel2 = 'B&M'
    
    sql = '''
    SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED
    declare @Banner as nvarchar(10),
    		@Country as nvarchar(10),
    		@Channel1 as nvarchar(10),
    		@Channel2 as nvarchar(10),
    		@Year as int,
    		@Month as int,
    		@After#Month as int,
    		@Channel as nvarchar(10);
    
    set @Banner = ''' + "'" + banner + "'" + ''';
    set @Country = '''+ "'" + country + "'" +''' ;
    set @Channel1 = ''' + "'" + channel1 + "'" + ''';
    set @Channel2 = ''' + "'" + channel2 + "'" + ''';
    set @Year = ''' + year +  ''';
    set @Month = ''' + month +  ''';
    set @After#Month = ''' + aftermonth + ''';
    set @Channel = ''' + "'" + channel + "'" +  ''';
    
	SELECT @Banner as AcqBanner
		,@Country as AcqCountry
		,@Channel as AcqChannel 
		,CAST(CAST(@Year AS varchar) + '-' + CAST(@Month AS varchar) + '-01' AS DATE) as AcquisitionFiscalMonthDate
		, @After#Month as After#Month
		, DATEADD(month, @After#Month ,CAST(CAST(@Year AS varchar) + '-' + CAST(@Month AS varchar) + '-28' AS DATE)) as TillFiscalMonthDate
        , SUM(SalesQuantity) AS SalesQuantity
        , SUM(GMAmount) AS GmAmount
		, (select count(distinct Email)
	from crmsandbox.dbo.Ruite_AcquisitionCohort with(nolock)
	where Banner = @Banner
	and Country = @Country
	and Channel in (@Channel1, @Channel2)
	and convert(date,CAST(Fiscalyear AS varchar) + '-' + CAST(FiscalMonth AS varchar) + '-01') = CONVERT(DATE,CAST(@Year AS varchar) + '-' + CAST(@Month AS varchar) + '-01')
	) as NewCustomerCount
		, count(distinct F_TransactionNo) AS TransactionCount
		, SUM(case when T.Channel = 'Ecom' then SalesAmount else 0 end) AS Ecom_NetSalesAmount
		, SUM(case when T.Channel = 'Ecom' then SalesQuantity else 0 end) AS Ecom_SalesQuantity
		, sum(RegularPriceSaleAmount) as RegularPriceSalesAmount
		, SUM(SalesAmount) AS NetSalesAmount
		, sum(RegularPriceSalesQuantity) as RegularPriceSalesQuantity

	FROM crmsandbox.dbo.Ruite_Total_sales_2018_20200309 T with(nolock)	
	inner join (select Email
	, convert(date,CAST(Fiscalyear AS varchar) + '-' + CAST(FiscalMonth AS varchar) + '-01') as FiscalMonthDate
	from crmsandbox.dbo.Ruite_AcquisitionCohort with(nolock)
	where Banner = @Banner
	and Country = @Country
	and Channel in (@Channel1, @Channel2)
	and convert(date,CAST(Fiscalyear AS varchar) + '-' + CAST(FiscalMonth AS varchar) + '-01') = CONVERT(DATE,CAST(@Year AS varchar) + '-' + CAST(@Month AS varchar) + '-01')
	) S 
	on S.Email = T.F_EmailAddress
	where
	convert(date,CAST(F_FiscalYearNo AS varchar) + '-' + CAST(F_FiscalPeriodNo AS varchar) + '-01') between 
	--Dateadd(month, 1,
	CONVERT(DATE,CAST(@Year AS varchar) + '-' + CAST(@Month AS varchar) + '-01')
	--) 
	and DATEADD(month, @After#Month ,CONVERT(DATE,CAST(@Year AS varchar) + '-' + CAST(@Month AS varchar) + '-01'))
;
    
    '''
    
    df = pd.read_sql(sql,conn)
    return df


                            
df = pd.read_csv(r'\starter.csv')                         

count = 1  
for year in [2018]:
    for banner in ['Dy'] :
        for country in ['US']:
            for month in range(1,7):
                for aftermonth in range(0,6):
                    for channel in ['Ecom', 'B&M', 'Omni']:
                        a = acquisition(banner, country, str(year), str(month), str(aftermonth), channel)
                        if a['NewCustomerCount'].iloc[0] != 0:
                            df = df.append(a)
                            count += 1
                            print(count)

df['RegularPriceSales%'] = df['RegularPriceSalesAmount']/df['NetSalesAmount']
df['% total sales ecom'] = df['Ecom_NetSalesAmount']/df['NetSalesAmount']

df = df[['AcqBanner',
 'AcqChannel',
 'AcqCountry',
 'AcquisitionFiscalMonthDate',
 'After#Month',
 'TillFiscalMonthDate',
 'SalesQuantity',
 'GmAmount',
 'NewCustomerCount',
 'TransactionCount',
 'Ecom_NetSalesAmount',
 'RegularPriceSalesAmount',
 'NetSalesAmount',
 'RegularPriceSalesQuantity',
 'Ecom_SalesQuantity',
 'RegularPriceSales%',
 '% total sales ecom']]


df.to_csv(r'\\20200309_2018 Fiscal 1 to 6.csv')   
