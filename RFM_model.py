import seaborn as sns
import pandas as pd
from datetime  import timedelta



orders = pd.read_csv('orders.csv')


orders['orderdate'] = pd.to_datetime(orders['orderdate'], format = '%Y-%m-%d')

today = pd.Timestamp(2017, 4, 11)

orders['period'] = today - orders['orderdate']

def classperiod(val):
    if val >timedelta(days=55):
        return '>55 day'
    
    elif val >=timedelta(days=31) and val <timedelta(days=55) :
        return '31-55 day'
    elif val >=timedelta(days=23) and val <timedelta(days=30) :
        return '23-30 day'
    elif val >=timedelta(days=16) and val <timedelta(days=22) :
        return '16-22 day'
    elif val >=timedelta(days=8) and val <timedelta(days=15) :
        return '8-15 day'
    else:
        return '0-7 day'



orders['periodclass']  = orders['period'].apply(classperiod)

orders['value'] = 1

countorder = orders[['clientId','periodclass','value']].groupby(['clientId','periodclass']).sum().reset_index()

sale = orders[['clientId', 'periodclass', 'gender', 'Sale']].groupby(['clientId', 'periodclass', 'gender']).sum().reset_index()



def classcount(val):
    if val >20:
        return '>20 freq'
    elif val > 12 and val <= 20 :
        return '12-20 freq'
    elif val > 8 and val <= 12 :
        return '8-12 freq'
    elif val > 4 and val <= 8 :
        return '4-8 freq'
    elif val > 2 and val <= 4 :
        return '2-4 freq'
    elif  val <=2 :
        return '<=2 freq'




countorder['frequency']  = countorder['value'].apply(classcount)

countfinal = countorder[['periodclass','frequency','clientId']].groupby(['periodclass','frequency']).count().reset_index()

df = sale.merge(countorder)

df2 = df[['periodclass', 'gender', 'Sale', 'frequency']].groupby(['periodclass', 'gender', 'frequency']).sum().reset_index()

df2.columns = ['days', 'gender', 'frequency', 'Sale']

g = sns.FacetGrid(df2, col="days",  row="frequency" ,
                  col_order= ['>55 day', '31-55 day', '23-30 day', '16-22 day', '8-15 day', '0-7 day'],
                  row_order= ['>20 freq','12-20 freq','8-12 freq','4-8 freq','2-4 freq' ,'<=2 freq'],
                  margin_titles=True)
g = g.map_dataframe(sns.barplot,x = 'gender', y='Sale',palette = sns.color_palette("muted"))
g = g.set_axis_labels('gender','Sale').add_legend()
