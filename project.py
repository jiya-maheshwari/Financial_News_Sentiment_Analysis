from bs4 import BeautifulSoup # type: ignore
from urllib.request import urlopen,Request
import pandas as pd # type: ignore
from datetime import date
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore
import matplotlib.pyplot as plt # type: ignore
import yfinance as yf # type: ignore
from pytz import timezone # type: ignore

#web scraping 

url = 'https://finviz.com/quote.ashx?t='

tickers = ['NVDA','TSLA','AAPL']

end = '&ty=c&ta=1&p=d'

news_tables = {}
for ticker in tickers:
    list_of_titles = []
    time_list = []
    date_list = []
    finviz = url+ticker+end
    req = Request(url = finviz, headers={'user-agent':'portfolio-project'})
    response = urlopen(req)
    html = BeautifulSoup(response,'html.parser')
    titles = html.find_all('a', class_='tab-link-news')
    for title in titles:
        list_of_titles.append(title.text.strip())
    news_tables[ticker] = list_of_titles
    timestamps = html.find_all('td', attrs={'align': 'right', 'width': '130'})
    for timestamp in timestamps:
        dates = timestamp.text.strip().split(' ')
        if len(dates) == 1:
            time = dates[0]
        else:
            date = dates[0]
            time = dates[1]
        time_list.append(time)
        date_list.append(date)
    news_tables[ticker+'_date'] = date_list
    news_tables[ticker+'_time'] = time_list

#preprocessing + sentiment analysis using nltk(vader) 

titles_df = pd.DataFrame(news_tables)
func = lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'] if x else 0

for ticker in tickers:
    titles_df[ticker+'_compound_score'] = titles_df[ticker].apply(func)

titles_df['NVDA_date_time'] = titles_df['NVDA_date'].str.cat(titles_df['NVDA_time'], sep=' ')
titles_df['TSLA_date_time'] = titles_df['TSLA_date'].str.cat(titles_df['TSLA_time'], sep=' ')
titles_df['AAPL_date_time'] = titles_df['AAPL_date'].str.cat(titles_df['AAPL_time'], sep=' ')

nyc_tz = timezone("America/New_York")

def parse_time(row):
    if "Today" in row:
        today = datetime.datetime.now(nyc_tz).date()  
        time_part = datetime.datetime.strptime(row.replace('Today ', ''), '%I:%M%p').time()
        combined = datetime.datetime.combine(today, time_part)
    else:
        combined = datetime.datetime.strptime(row, '%b-%d-%y %I:%M%p')
    return nyc_tz.localize(combined)

titles_df['NVDA_date_time'] = titles_df['NVDA_date_time'].apply(parse_time)
titles_df['TSLA_date_time'] = titles_df['TSLA_date_time'].apply(parse_time)
titles_df['AAPL_date_time'] = titles_df['AAPL_date_time'].apply(parse_time)

titles_df['NVDA_date'] = titles_df['NVDA_date'].replace('Today', datetime.datetime.now(nyc_tz).date().strftime('%b-%d-%y'))
titles_df['NVDA_date'] = pd.to_datetime(titles_df['NVDA_date'], format='%b-%d-%y').dt.date

titles_df['TSLA_date'] = titles_df['TSLA_date'].replace('Today', datetime.datetime.now(nyc_tz).date().strftime('%b-%d-%y'))
titles_df['TSLA_date'] = pd.to_datetime(titles_df['TSLA_date'], format='%b-%d-%y').dt.date

titles_df['AAPL_date'] = titles_df['AAPL_date'].replace('Today',datetime.datetime.now(nyc_tz).date().strftime('%b-%d-%y'))
titles_df['AAPL_date'] = pd.to_datetime(titles_df['AAPL_date'], format='%b-%d-%y').dt.date

nvidia_mean = titles_df[['NVDA_date','NVDA_compound_score']].groupby(['NVDA_date']).mean().reset_index()
tsla_mean = titles_df[['TSLA_date','TSLA_compound_score']].groupby(['TSLA_date']).mean().reset_index()
aapl_mean = titles_df[['AAPL_date','AAPL_compound_score']].groupby(['AAPL_date']).mean().reset_index()

#visualization 

nvidia_mean.rename(columns={"NVDA_date": "Date", "NVDA_compound_score": "NVDA"}, inplace=True)
tsla_mean.rename(columns={"TSLA_date": "Date", "TSLA_compound_score": "TSLA"}, inplace=True)
aapl_mean.rename(columns={"AAPL_date": "Date", "AAPL_compound_score": "AAPL"}, inplace=True)

mean_df = pd.merge(nvidia_mean,tsla_mean,on='Date',how='outer')
mean_df = pd.merge(mean_df,aapl_mean,on='Date',how = 'outer')

mean_df.plot(kind = 'bar',x = 'Date')
plt.xlabel("Date")
plt.ylabel("Mean Value of Compound Score")
plt.title("Comparison of Mean Compound Scores: NVIDIA, Tesla, Apple")
plt.legend()
plt.show()

#classification based on Vader 

def categorize_sentiment(compound_score):
    if compound_score > 0.05:
        return 'Positive'
    elif compound_score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    
for ticker in tickers:
    titles_df[ticker+'_sentiment'] = titles_df[ticker+'_compound_score'].apply(categorize_sentiment)

#time series analysis
NVDA_time_series_df = pd.DataFrame()
TSLA_time_series_df = pd.DataFrame()
AAPL_time_series_df = pd.DataFrame()

for ticker in tickers:
    start_date = titles_df[ticker+'_date'].iloc[-1]
    end_date = titles_df[ticker+'_date'].iloc[0]+datetime.timedelta(days=1)
    data = yf.download(ticker, start=start_date, end=end_date, interval="1m")
    data.columns = data.columns.get_level_values(0)
    data.index = data.index.tz_convert(nyc_tz)
    data = data[['Open','Close']].reset_index()
    data['price_change'] = data['Close']-data['Open']
    data['lagged_price_change'] = data['price_change'].shift(-1) 
    date_time_df = titles_df[[ticker+'_date_time',ticker+'_compound_score']]
    time_series_df = pd.merge(data,date_time_df,left_on='Datetime',right_on=ticker+'_date_time',how='right')
    if (ticker == 'NVDA'):
        NVDA_time_series_df = time_series_df.dropna()
        NVDA_time_series_df = NVDA_time_series_df[['NVDA_date_time',"NVDA_compound_score","lagged_price_change"]]
    elif (ticker == 'TSLA'):
        TSLA_time_series_df = time_series_df.dropna()
        TSLA_time_series_df = TSLA_time_series_df[['TSLA_date_time',"TSLA_compound_score","lagged_price_change"]]
    else:
        AAPL_time_series_df = time_series_df.dropna()
        AAPL_time_series_df = AAPL_time_series_df[['AAPL_date_time',"AAPL_compound_score","lagged_price_change"]]




















