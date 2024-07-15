from datetime import date
import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from dash import Dash, dash_table
import plotly.express as px
import plotly.graph_objects as go
import time as time
from collections import Counter
from itertools import count

t0 = time.time()
df = pd.DataFrame()
chunks = []
df_iter = pd.read_csv('nft_data.csv', chunksize=200000, low_memory=False, iterator=True)
for iter_num, chunk in enumerate(df_iter, 1):
    print('Processing iteration {0}'.format(iter_num))
    #print(chunk)
    chunks.append(chunk)
df = pd.concat(chunks, axis=0)
t1 = time.time()

buyer_username_df = df["Buyer_username"].value_counts()
df_numbers = pd.DataFrame({"Buyer Username": buyer_username_df.index, 'Numbers of NFTs Bought': buyer_username_df.values})

buyer_price_df = df['Price_USD'].groupby(df['Buyer_username']).sum()
df_total = pd.DataFrame({"Buyer Username": buyer_price_df.index, 'Total USD Spent': buyer_price_df.values})

df_buyer = pd.merge(df_numbers, df_total, on="Buyer Username").head(100)

#df2 = pd.DataFrame({"Datetime_updated": trade_volume_counts.index, 'Count': trade_volume_counts.values})
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in df_buyer.columns
        ],
        data=df_buyer.to_dict('records'),
        fixed_rows={'headers': True},
        sort_action="native",
        sort_mode='multi',
        page_action='native',
        page_current= 0,
        page_size= 10,
        style_table={'height': '300px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold',
            'border': '1px solid black'
        }
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)