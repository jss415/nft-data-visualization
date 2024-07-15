import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time as time
import numpy
import sys
from collections import Counter
from itertools import count


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

## Using chunking, find the 10-largest elements of Price_USD column for each chunk and combine the answers
df2 = pd.DataFrame()

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

n_largest_price_chunks = []
t2 = time.time()

for i in chunker(df, 200000):
    n_largest_price_chunks.append(i.nlargest(10, "Price_USD"))
t3 = time.time()

print('sort/head: {:0.2f}s'.format(t3 - t2))

df2 = pd.concat(n_largest_price_chunks, axis=0)
columns = ["Name", "Price_USD"]
df3 = df2.nlargest(10, "Price_USD").filter(columns)
print(df3)

c = Counter(df3["Name"].to_numpy())

# Avoid duplicate names
iters = {k: count(1) for k, v in c.items() if v > 1}
output_list = [x+str(next(iters[x])) if x in iters else x for x in df3["Name"].to_numpy()]

fig = go.Figure()
fig.add_bar(x=output_list, 
        y=df3["Price_USD"].to_numpy(),
        marker=dict(color="rgb(123, 199, 255)"))
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
fig.update_layout(font_family="Arial")
fig.update_layout(xaxis = dict(tickfont = dict(size=15)), yaxis = dict(tickfont = dict(size=13)))
fig.update_layout(xaxis_title="Names", yaxis_title="Price USD")
fig.update_layout(
    title={
        'text': "Top 10 Most Expensive NFTs",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
print(df3["Name"].to_numpy())
print(df3["Price_USD"].to_numpy())

fig.show()


