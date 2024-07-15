import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time as time
import numpy
import sys
from collections import Counter
from itertools import count

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

t0 = time.time()
df = pd.DataFrame()
chunks = []
df_iter = pd.read_csv('nft_data.csv', chunksize=200000, low_memory=False, iterator=True)
for iter_num, chunk in enumerate(df_iter, 1):
    print('Processing iteration {0}'.format(iter_num))
    print(chunk)
    chunks.append(chunk)
df = pd.concat(chunks, axis=0)
t1 = time.time()

print('reading: {:0.2f}s'.format(t1 - t0))

sales_counts = None
flag = 0

t2 = time.time()
for i in chunker(df, 200000):
    chunk = i['Price_USD'].groupby(i['Datetime_updated']).sum()
    if flag == 0:
        sales_counts = chunk
        flag = 1
    else:
        sales_counts = sales_counts.add(chunk, fill_value=0)
t3 = time.time()
print('find trade_volume_counts: {:0.2f}s'.format(t3 - t2))

print("sales count (unsorted index): ")
print(sales_counts)
print("sales count (sorted index): ")
df2 = pd.DataFrame({"Datetime_updated": sales_counts.index, 'Price_USD': sales_counts.values})
print(df2.head(10))

data = [
    dict(
        type="scatter",
        mode="lines",
        x=df2['Datetime_updated'],
        y=df2['Price_USD'],
        line=dict(shape="spline", smoothing=1.3, width=2, color="#59C3C3")
    )
]

fig = go.Figure(data=data)

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
fig.update_layout(font_family="Arial")
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#D8D8D8')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#D8D8D8')
fig.update_layout(xaxis_title="Dates", yaxis_title="Total Price USD")
fig.update_layout(xaxis = dict(tickfont = dict(size=16)), yaxis = dict(tickfont = dict(size=17)))
fig.update_layout(
    title={
        'text': "Sales Volume",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})


fig.show()




