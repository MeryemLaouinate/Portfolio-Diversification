########################## Web Scraping for daily moroccan stock prices #####################################
from datetime import datetime
import requests
import string
import csv
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd

# a simple helper function
def formatIt(s) :
    output = ''
    for i in s :
        if i in string.printable :
            output += i
    return output

# default url
uri = "http://www.casablanca-bourse.com/bourseweb/en/Negociation-History.aspx?Cat=24&IdLink=225"


def get_viewState_and_symVal (symbolName, session) :
    #session = requests.Session()
    r = session.get(uri)
    soup = BeautifulSoup(r.content,'html.parser') #soup = BeautifulSoup(r.text)
    # let's get the viewstate value
    viewstate_val = soup.find('input', attrs = {"id" : "__VIEWSTATE"})['value']
    # let's get the symbol value
    selectSymb = soup.find('select', attrs = {"name" : "HistoriqueNegociation1$HistValeur1$DDValeur"})
    for i in selectSymb.find_all('option') :
        if i.text == symbolName :
            symbol_val = i['value']
    # simple sanity check before return !
    try :
        symbol_val
    except :
        raise NameError ("Symbol Name not found !!!")
    else :
        return (viewstate_val, symbol_val)


def MainFun(symbolName, dateFrom, dateTo):
  session = requests.Session()
  request1 = get_viewState_and_symVal(symbolName, session)
  viewstate = request1[0]
  symbol = request1[1]
  payload = {
    'TopControl1$ScriptManager1': r'HistoriqueNegociation1$UpdatePanel1|HistoriqueNegociation1$HistValeur1$Image1',
    '__VIEWSTATE': viewstate,
    'HistoriqueNegociation1$HistValeur1$DDValeur': symbol,
    'HistoriqueNegociation1$HistValeur1$historique': r'RBSearchDate',
    'HistoriqueNegociation1$HistValeur1$DateTimeControl1$TBCalendar': dateFrom,
    'HistoriqueNegociation1$HistValeur1$DateTimeControl2$TBCalendar': dateTo,
    'HistoriqueNegociation1$HistValeur1$DDuree': r'6',
    'hiddenInputToUpdateATBuffer_CommonToolkitScripts': r'1',
    'HistoriqueNegociation1$HistValeur1$Image1.x': r'27',
    'HistoriqueNegociation1$HistValeur1$Image1.y': r'8'
  }

  request2 = session.post(uri, data=payload)
  soup2 = BeautifulSoup(request2.content,'html.parser')
  ops = soup2.find_all('table', id="arial11bleu")
  for i in ops:
    try:
      i['class']
    except:
      rslt = i
      break

  output = []
  for i in rslt.find_all('tr')[1:]:
    temp = []
    for j in i.find_all('td'):
      sani = j.text.strip()
      if not sani in string.whitespace:
        temp.append(formatIt(sani))
    if len(temp) > 0:
      output.append(temp)

  with open("output.csv", "w") as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerows(output)

  return writer

url = 'http://www.casablanca-bourse.com/bourseweb/en/Negociation-History.aspx?Cat=24&IdLink=225'

response = requests.get(url).text

soup = BeautifulSoup(response,"html.parser")

drop = soup.find('select', attrs = {"name" : "HistoriqueNegociation1$HistValeur1$DDValeur"}).find_all('option')

options = []

val = []

for i in range(0,len(drop)):

    options.append(drop[i].text)

    val.append(drop[i]['value'])

# Getting all the moroccan assets to a global dataframe
dfs = []
for i in range(1, len(options)):
  MainFun(options[i], "1/1/2018", datetime.today().strftime('%d/%m/%Y'))
  dfs.append(pd.read_csv('C:/Users/Meryem/PycharmProjects/PortfolioOpt/output.csv', sep=';'))

df_all = pd.concat(dfs, ignore_index=True)

df_mor_assets = df_all

#Delete unusefull columns
del df_all['Reference price']
del df_all['+Intraday high']
del df_all['+ Intraday low']
del df_all['Number of shares traded']
del df_all['Capitalisation']

#Setting Date as index
df_all.set_index('Session',inplace=True)

#Convert assets name from column into row
df_all = df_all.pivot(columns ='Instrument')

#drop multiindex level
df_all.columns = df_all.columns.droplevel()
df_all.dropna(inplace=True)
options.remove('Choose an instrument ')
df_mor = df_all.copy()
dftest = df_all.replace('[^\d.]','',regex=True).astype(float)
pd.options.display.float_format = '{:,.1f}'.format
dftest = dftest/100
dftest.to_excel('Bourse_De_Casa.xlsx')

