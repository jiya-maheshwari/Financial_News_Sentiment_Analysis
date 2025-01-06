from bs4 import BeautifulSoup
from urllib.request import urlopen,Request

url = 'https://finviz.com/quote.ashx?t='

tickers = ['NVDA','TSLA','AAPL']

end = '&ty=c&ta=1&p=d'

news_tables = {}

for ticker in tickers:
    finviz = url+ticker+end
    req = Request(url = finviz, headers={'user-agent':'portfolio-project'})
    response = urlopen(req)
    html = BeautifulSoup(response,'html')
    news_table = html.find(id = 'news-table')
    news_tables[ticker] = news_table
    break
print(news_table)

