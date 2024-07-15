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
from dash.dash_table import FormatTemplate
import plotly.express as px
import plotly.graph_objects as go
import time as time
from collections import Counter
from itertools import count


## Do not uncomment this block of code below unless you are ready to test with the entire dataset
# df = pd.DataFrame()
# chunks = []
# df_iter = pd.read_csv('nft_data.csv', chunksize=100000, low_memory=False, iterator=True)
# for iter_num, chunk in enumerate(df_iter, 1):
#     print('Processing iteration {0}'.format(iter_num))
#     print(chunk)
#     chunks.append(chunk)
# df = pd.concat(chunks, axis=0)
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


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def load_pie_chart():
    flag = 0
    n_largest = None

    t2 = time.time()
    for i in chunker(df, 200000):
        chunk = i["Category"].value_counts()
        if flag == 0:
            n_largest = chunk
            flag = 1
        else:
            n_largest = n_largest.add(chunk, fill_value=0)

    df2 = pd.DataFrame({"Category": n_largest.index, 'Count': n_largest.values})
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
    t3 = time.time()
    print('pie_chart took: {:0.2f}s'.format(t3 - t2))

    return fig

def load_bar_chart():
    df2 = pd.DataFrame()
    n_largest_price_chunks = []
    t2 = time.time()

    for i in chunker(df, 200000):
        n_largest_price_chunks.append(i.nlargest(10, "Price_USD"))
    
    df2 = pd.concat(n_largest_price_chunks, axis=0)
    columns = ["Name", "Price_USD"]
    df3 = df2.nlargest(10, "Price_USD").filter(columns)
    c = Counter(df3["Name"].to_numpy())

    # Avoid duplicate names
    iters = {k: count(1) for k, v in c.items() if v > 1}
    output_list = [x+str(next(iters[x])) if x in iters else x for x in df3["Name"].to_numpy()]

    fig = go.Figure()
    fig.add_bar(x=output_list, 
            y=df3["Price_USD"].to_numpy(),
            marker=dict(color="rgb(123, 199, 255)"))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
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
    t3 = time.time()
    print('bar-chart took: {:0.2f}s'.format(t3 - t2))
    
    return fig

def load_transaction_chart():
    trade_volume_counts = None
    flag = 0

    t2 = time.time()
    for i in chunker(df, 200000):
        chunk = i["Datetime_updated"].value_counts()
        if flag == 0:
            trade_volume_counts = chunk
            flag = 1
        else:
            trade_volume_counts = trade_volume_counts.add(chunk, fill_value=0)
    
    df2 = pd.DataFrame({"Datetime_updated": trade_volume_counts.index, 'Count': trade_volume_counts.values})
    df2 = df2.sort_values(by=['Datetime_updated'])

    data = [
        dict(
            type="scatter",
            mode="lines",
            x=df2['Datetime_updated'],
            y=df2['Count'],
            line=dict(shape="spline", smoothing=1.3, width=2, color="#59C3C3")
        )
    ]

    fig = go.Figure(data=data)

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(font_family="Arial")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#D8D8D8')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#D8D8D8')
    fig.update_layout(xaxis_title="Dates", yaxis_title="Number of Transactions")
    fig.update_layout(xaxis = dict(tickfont = dict(size=16)), yaxis = dict(tickfont = dict(size=17)))
    fig.update_layout(
        title={
            'text': "Transaction Volume",
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    
    t3 = time.time()
    print('find trade_volume_counts: {:0.2f}s'.format(t3 - t2))

    return fig

def load_sales_chart():
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
    
    df2 = pd.DataFrame({"Datetime_updated": sales_counts.index, 'Price_USD': sales_counts.values})

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

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
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
    t3 = time.time()
    print('sales chart took: {:0.2f}s'.format(t3 - t2))

    return fig

def load_buyer_table():
    t2 = time.time()
    buyer_username_df = df["Buyer_username"].value_counts()
    df_numbers = pd.DataFrame({"Buyer Username": buyer_username_df.index, 'NFTs Bought': buyer_username_df.values})

    buyer_price_df = df['Price_USD'].groupby(df['Buyer_username']).sum()
    df_total = pd.DataFrame({"Buyer Username": buyer_price_df.index, 'Total USD Bought': buyer_price_df.values})
    df_buyer = pd.merge(df_numbers, df_total, on="Buyer Username").head(100)
    t3 = time.time()
    print('loading buyer table: {:0.2f}s'.format(t3 - t2))
    df_buyer['Total USD Bought'] = df_buyer['Total USD Bought'].round(decimals = 2)
    return df_buyer

def load_seller_table():
    t2 = time.time()
    seller_username_df = df["Seller_username"].value_counts()
    df_numbers = pd.DataFrame({"Seller Username": seller_username_df.index, 'NFTs Sold': seller_username_df.values})

    seller_price_df = df['Price_USD'].groupby(df['Seller_username']).sum()
    df_total = pd.DataFrame({"Seller Username": seller_price_df.index, 'Total USD Sold': seller_price_df.values})

    df_seller = pd.merge(df_numbers, df_total, on="Seller Username").head(100)
    t3 = time.time()
    print('loading seller table: {:0.2f}s'.format(t3 - t2))
    df_seller['Total USD Sold'] = df_seller['Total USD Sold'].round(decimals = 2)
    
    return df_seller



def make_empty_figure():
    fig = go.Figure()
    fig.layout.paper_bgcolor = '#E5ECF6'
    fig.layout.plot_bgcolor = '#E5ECF6'
    return fig

collections = ["alien.worlds", "Cryptokitties", "Gods-unchained", "stf.capcom", "kogsofficial", "mlb.topps", "Godsunchained", "Sorare", "Axie", "officialhero"]
original_start, original_end = "2017-11-23", "2021-04-27"

# ToDo: write functons to create these two dataframes (filter and get the first 100)
#       change app layout to hook these tables to the frontend
#       go to update_prices to filter based on attributes  
df_buyer = load_buyer_table()
df_seller = load_seller_table()

buyer_table = dash_table.DataTable(
    id='datatable-interactivity_1',
    #columns=[
        #{"name": i, "id": i, "deletable": False, "selectable": True} for i in df_buyer.columns
    #],
    columns=[
        dict(id='Buyer Username', name='Buyer Username', deletable=False, selectable=True),
        dict(id='NFTs Bought', name='NFTs Bought', type='numeric', deletable=False, selectable=True),
        dict(id='Total USD Bought', name='Total USD Bought', type='numeric', format=FormatTemplate.money(2), deletable=False, selectable=True)
    ],
    data=df_buyer.to_dict('records'),
    fixed_rows={'headers': True},
    sort_action="native",
    sort_mode='multi',
    page_action='native',
    page_current= 0,
    page_size= 13,
    style_table={'height': '400px', 'overflowY': 'auto'},
    style_cell={
        # all three widths are needed
        'minWidth': '200px', 'width': '200px', 'maxWidth': '114px',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
    },
    style_header={
        'backgroundColor': 'white',
        'fontWeight': 'bold',
        'border': '1px solid black'
    }
)

seller_table = dash_table.DataTable(
    id='datatable-interactivity_2',
    #columns=[
        #{"name": i, "id": i, "deletable": False, "selectable": True} for i in df_seller.columns
    #],
    columns=[
        dict(id='Seller Username', name='Seller Username', deletable=False, selectable=True),
        dict(id='NFTs Sold', name='NFTs Sold', type='numeric', deletable=False, selectable=True),
        dict(id='Total USD Sold', name='Total USD Sold', type='numeric', format=FormatTemplate.money(2), deletable=False, selectable=True)
    ],
    data=df_seller.to_dict('records'),
    fixed_rows={'headers': True},
    sort_action="native",
    sort_mode='multi',
    page_action='native',
    page_current= 0,
    page_size= 13,
    style_table={'height': '400px', 'overflowY': 'auto'},
    style_cell={
        # all three widths are needed
        'minWidth': '200px', 'width': '200px', 'maxWidth': '114px',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
    },
    style_header={
        'backgroundColor': 'white',
        'fontWeight': 'bold',
        'border': '1px solid black'
    }
)




app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "NFT"


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H3("Non-Fungible Tokens (NFTs) Dashboard", style={"margin-bottom": "0px"})
            ])
        ], id="title"),
        html.Div([
            html.A(html.Button("Learn More", id="learn-more-button"),
                href="https://plot.ly/dash/pricing/")
        ], className="one-third column", id="button")
    ], id="header", style={"margin-bottom": "25px"}),
    html.Div([
        html.Div([
            html.P("Filter by transaction dates: ", style={"padding-top": "5px"}),
            dcc.DatePickerRange(
                id='my-date-picker-range',
                min_date_allowed=date(2017, 11, 23),
                max_date_allowed=date(2021, 4, 27),
                start_date=date(2017, 11, 23),
                end_date=date(2021, 4, 27),
                className="dcc_control"
            ),
            html.P("Filter by USD prices ($ millions): ", style={"padding-top": "5px"}),
            dcc.RangeSlider(
                id='my-price-slider-range',
                min=0,
                max=8,
                included=True,
                marks={interval: {'label': str(interval) + 'm', 'style': {'fontSize': 12}} for interval in np.arange(0, 8, 0.5)},
                className="dcc-range-slider"
            ),
            html.P("Filter by Collections: ", style={"padding-top": "10px", "margin-bottom": "0px"}),
            dcc.Dropdown(
                id='my-collection-dropdown',
                multi=True,
                placeholder='Select one or more collections',
                value=collections,
                options=[{"label": collection_name, "value": collection_name} for collection_name in collections]
            ),
            html.Br(),
            dbc.Button("Submit", id="submit", size="sm", className="submit")
        ], className="pretty_container four columns"),
        html.Div([
            html.Div([
                html.Div([
                    html.H6("6,071,027", id="transaction_text"), html.P("No. of Transactions")
                ], className="mini_container"),
                html.Div([
                    html.H6("6283", id="collection_text"), html.P("No. of Collections")
                ], className="mini_container"),
                html.Div([
                    html.H6("$457,231,026", id="price_USD_text"), html.P("Total Sales (USD)")
                ], className="mini_container")
            ], className="flex-display"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id="pie-chart", figure=load_pie_chart())
                    ], lg=6),
                    dbc.Col([
                        dcc.Graph(id="bar-chart", figure=load_bar_chart())
                    ], lg=6)
                ])
            ])          
        ],  id="right-column", className="eight columns")
    ], className="flex-display"),
    html.Div([
        dbc.Row([
            dbc.Col(),
            dbc.Col([
                dcc.Graph(id="transaction-chart", figure=load_transaction_chart())
            ], lg=5),
            dbc.Col([
                dcc.Graph(id="sales-chart", figure=load_sales_chart())
            ], lg=5),
            dbc.Col()
        ], justify="evenly")
    ], style={"padding-top": "20px"}),
    html.Div([
        dbc.Row([
            dbc.Col(),
            dbc.Col([
                buyer_table
            ], lg=5),
            dbc.Col([
                seller_table
            ], lg=5),
            dbc.Col()
        ], justify="evenly")
    ], style={"padding-top": "20px"})    
])


def update_pie_chart(df_filtered):
    flag = 0
    n_largest = None

    t2 = time.time()
    for i in chunker(df_filtered, 200000):
        chunk = i["Category"].value_counts()
        if flag == 0:
            n_largest = chunk
            flag = 1
        else:
            n_largest = n_largest.add(chunk, fill_value=0)

    df2 = pd.DataFrame({"Category": n_largest.index, 'Count': n_largest.values})
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
    t3 = time.time()
    print('updating pie_chart took: {:0.2f}s'.format(t3 - t2))
    return fig

def update_bar_chart(df_filtered):
    df2 = pd.DataFrame()
    n_largest_price_chunks = []
    t2 = time.time()

    for i in chunker(df_filtered, 200000):
        n_largest_price_chunks.append(i.nlargest(10, "Price_USD"))
    
    df2 = pd.concat(n_largest_price_chunks, axis=0)
    columns = ["Name", "Price_USD"]
    df3 = df2.nlargest(10, "Price_USD").filter(columns)
    c = Counter(df3["Name"].to_numpy())

    # Avoid duplicate names
    iters = {k: count(1) for k, v in c.items() if v > 1}
    output_list = [str(x)+str(next(iters[x])) if x in iters else x for x in df3["Name"].to_numpy()]

    fig = go.Figure()
    fig.add_bar(x=output_list, 
            y=df3["Price_USD"].to_numpy(),
            marker=dict(color="rgb(123, 199, 255)"))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
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
    t3 = time.time()
    print('updating bar-chart took: {:0.2f}s'.format(t3 - t2))
    
    return fig

def update_transaction_chart(df_filtered):
    trade_volume_counts = None
    flag = 0

    t2 = time.time()
    for i in chunker(df_filtered, 200000):
        chunk = i["Datetime_updated"].value_counts()
        if flag == 0:
            trade_volume_counts = chunk
            flag = 1
        else:
            trade_volume_counts = trade_volume_counts.add(chunk, fill_value=0)
    df2 = pd.DataFrame({"Datetime_updated": trade_volume_counts.index, 'Count': trade_volume_counts.values})
    df2 = df2.sort_values(by=['Datetime_updated'])

    data = [
        dict(
            type="scatter",
            mode="lines",
            x=df2['Datetime_updated'],
            y=df2['Count'],
            line=dict(shape="spline", smoothing=1.3, width=2, color="#59C3C3")
        )
    ]

    fig = go.Figure(data=data)

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(font_family="Arial")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#D8D8D8')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#D8D8D8')
    fig.update_layout(xaxis_title="Dates", yaxis_title="Number of Transactions")
    fig.update_layout(xaxis = dict(tickfont = dict(size=16)), yaxis = dict(tickfont = dict(size=17)))
    fig.update_layout(
        title={
            'text': "Transaction Volume",
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    t3 = time.time()
    print('updating trade_volume_counts: {:0.2f}s'.format(t3 - t2))
    return fig

def update_sales_chart(df_filtered):
    sales_counts = None
    flag = 0

    t2 = time.time()
    for i in chunker(df_filtered, 200000):
        chunk = i['Price_USD'].groupby(i['Datetime_updated']).sum()
        if flag == 0:
            sales_counts = chunk
            flag = 1
        else:
            sales_counts = sales_counts.add(chunk, fill_value=0)

    df2 = pd.DataFrame({"Datetime_updated": sales_counts.index, 'Price_USD': sales_counts.values})

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

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
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

    t3 = time.time()
    print('updating trade_volume_counts: {:0.2f}s'.format(t3 - t2))

    return fig

def update_table_1(df_filtered):
    t2 = time.time()
    buyer_username_df = df_filtered["Buyer_username"].value_counts()
    df_numbers = pd.DataFrame({"Buyer Username": buyer_username_df.index, 'NFTs Bought': buyer_username_df.values})

    buyer_price_df = df_filtered['Price_USD'].groupby(df_filtered['Buyer_username']).sum()
    df_total = pd.DataFrame({"Buyer Username": buyer_price_df.index, 'Total USD Bought': buyer_price_df.values})
    df_buyer = pd.merge(df_numbers, df_total, on="Buyer Username").head(100)
    t3 = time.time()
    print('updating table_1: {:0.2f}s'.format(t3 - t2))
    df_buyer['Total USD Bought'] = df_buyer['Total USD Bought'].round(decimals = 2)

    return df_buyer

def update_table_2(df_filtered):
    t2 = time.time()
    seller_username_df = df_filtered["Seller_username"].value_counts()
    df_numbers = pd.DataFrame({"Seller Username": seller_username_df.index, 'NFTs Sold': seller_username_df.values})

    seller_price_df = df_filtered['Price_USD'].groupby(df['Seller_username']).sum()
    df_total = pd.DataFrame({"Seller Username": seller_price_df.index, 'Total USD Sold': seller_price_df.values})

    df_seller = pd.merge(df_numbers, df_total, on="Seller Username").head(100)
    t3 = time.time()
    print('updating table_2: {:0.2f}s'.format(t3 - t2))
    df_seller['Total USD Sold'] = df_seller['Total USD Sold'].round(decimals = 2)

    return df_seller

@app.callback(Output("pie-chart", "figure"),
            Output("bar-chart", "figure"),
            Output("transaction-chart", "figure"),
            Output("sales-chart", "figure"),
            Output("datatable-interactivity_1", "data"),
            Output("datatable-interactivity_2", "data"),
            Input("submit", "n_clicks"),
            State("my-price-slider-range", "value"),
            State("my-date-picker-range", "start_date"),
            State('my-date-picker-range', 'end_date')
)
def update_price(n_clicks, price, start_date, end_date):
    if n_clicks is None:
        raise PreventUpdate
    if price is None and start_date is None and end_date is None:
        raise PreventUpdate
    df_filtered = pd.DataFrame()

    start_date_string = ""
    end_date_string = ""

    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%Y-%m-%d')
        print(start_date_string)
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%Y-%m-%d')
        print(end_date_string)

    if price is not None:
        price_min = price[0]
        price_max = price[1]
        price_min = price_min * 1000000
        price_max = price_max * 1000000
        print(f"price_min: {price_min}, price_max: {price_max}")
        df_filtered = df.loc[(df['Price_USD'] >= price_min) & (df['Price_USD'] <= price_max)]
        df_filtered = df_filtered[(df_filtered['Datetime_updated'] > start_date_string) & (df_filtered['Datetime_updated'] < end_date_string)]
    else:
        df_filtered = df[(df['Datetime_updated'] > start_date_string) & (df['Datetime_updated'] < end_date_string)]

    fig1 = update_pie_chart(df_filtered)
    fig2 = update_bar_chart(df_filtered)
    fig3 = update_transaction_chart(df_filtered)
    fig4 = update_sales_chart(df_filtered)
    table_1 = update_table_1(df_filtered)
    table_2 = update_table_2(df_filtered)


    return fig1, fig2, fig3, fig4, table_1.to_dict('records'), table_2.to_dict('records')

if __name__ == "__main__":
    app.run_server(debug=True)