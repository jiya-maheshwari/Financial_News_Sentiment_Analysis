from bs4 import BeautifulSoup
from urllib.request import urlopen,Request
import pandas as pd
from datetime import date
from datetime import timedelta
# import nltk
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

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

titles_df = pd.DataFrame(news_tables)






