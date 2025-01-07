from bs4 import BeautifulSoup # type: ignore
from urllib.request import urlopen,Request
import pandas as pd # type: ignore
from datetime import date
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore
import matplotlib.pyplot as plt # type: ignore

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

nvdia_mean = titles_df[['NVDA_date','NVDA_compound_score']].groupby(['NVDA_date']).mean().reset_index()
tsla_mean = titles_df[['TSLA_date','TSLA_compound_score']].groupby(['TSLA_date']).mean().reset_index()
aapl_mean = titles_df[['AAPL_date','AAPL_compound_score']].groupby(['AAPL_date']).mean().reset_index()

mean_df = pd.DataFrame()
mean_df['Date'] = nvdia_mean['NVDA_date']
mean_df['NVDA'] = nvdia_mean['NVDA_compound_score']
mean_df['TSLA'] = tsla_mean['TSLA_compound_score']
mean_df['AAPL'] = aapl_mean['AAPL_compound_score']
mean_df['Date'] = mean_df['Date'].replace('Today', datetime.today().strftime('%b-%d-%y'))
mean_df['Date'] = pd.to_datetime(mean_df['Date'], format='%b-%d-%y').dt.date

#visualization 
mean_df.plot(kind = 'bar',x='Date')
plt.legend()
plt.show()














