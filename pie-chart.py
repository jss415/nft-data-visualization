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

n_largest = None
flag = 0

t2 = time.time()
for i in chunker(df, 200000):
    chunk = i["Category"].value_counts()
    if flag == 0:
        n_largest = chunk
        flag = 1
    else:
        n_largest = n_largest.add(chunk, fill_value=0)

t3 = time.time()

print('find n_largest: {:0.2f}s'.format(t3 - t2))

print("answer: ")
print(n_largest.sort_values(ascending=False))

df2 = pd.DataFrame({"Category": n_largest.index, 'Count': n_largest.values})


# To see how colors were generated, go to https://www.w3schools.com/w3css/w3css_color_generator.asp
# Set #176178, Hue: 194, Opacity: 0.68
colors = ['#ecf8fb', '#c1e8f3', '#84d0e7', '#46b9dc', '#2393b6', '#176178']

fig = go.Figure(data=[go.Pie(labels=df2['Category'], values=df2['Count'])])
fig.update_traces(hoverinfo='label+text+value+percent', 
            textinfo='label+percent', 
            textfont_size=20,
            marker=dict(colors=colors, line=dict(color='#000000', width=1)))

fig.update_layout(
    title={'text': "Top Categories of NFTs",},
    title_x=0.5,
    title_font_family = "Arial",
    title_font_size = 20,
    legend=dict(font=dict(color="black", size=20), bgcolor="rgba(0,0,0,0)")
)


fig.show()



