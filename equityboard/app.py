import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from capm import capm, efficient_frontier, profit, stock_prices, daily_return
import numpy as np
from datetime import date


def chart(df: pd.DataFrame, tickers: list[str]) -> None:
    fig = px.line(df, y=tickers,
                  labels={'value': 'Return (%)'},
                  )
    fig.update_layout(
        # plot_bgcolor=colors['background'],
        # paper_bgcolor=colors['background'],
        font_color=colors['text'],
    )
    return fig


def capm_scatter(
        df: pd.DataFrame, df_daily,
        from_day, to_day, tickers: list[str]) -> None:
    df_capm_filt = capm(df[tickers], from_day, to_day)
    res = None

    if len(tickers) > 1:
        profits = profit(df[tickers], from_day, to_day).iloc[-1].to_numpy()

        n_year = (date.fromisoformat(to_day) - date.fromisoformat(
            from_day)).days / 365.
        profits_ann = np.abs(profits) ** (1. / n_year) * np.sign(profits)

        target_profits = np.linspace(max(0., profits_ann.min()),
                                     profits_ann.max(), 10)

        res = efficient_frontier(
            df_daily[tickers], profits, target_profits, from_day, to_day)

    if res is not None:
        weights, risks, profits = res

    fig = px.scatter(
        df_capm_filt, x='risk', y='return',
        color=df_capm_filt.index.tolist(),
        size=np.ones(len(tickers)),
        size_max=10,
        labels={'risk': 'Annualized risk',
                'return': 'Annualized return (%)',
                'color': 'Symbol'},
        hover_name=df_capm_filt.index.tolist(),
        hover_data={'return': True,
                     'risk': True,
                     'Name': True,
                     # 'Symbol': False,
                     },

    )
    # fig.add_traces(
    #     go.Scatter(
    #         x=df_capm['risk'], y=df_capm['return'],
    #         mode='markers',
    #         marker={
    #             'size': [1 for i in range(len(df_capm))],
    #             'color': ['black' for i in range(len(df_capm))]
    #         },
    #     )
    # )
    if res is not None:
        fig.add_traces(
            go.Scatter(x=risks, y=profits, mode='lines',
                       name='eff. front.')
        )
    fig.update_layout(
        # plot_bgcolor=colors['background'],
        # paper_bgcolor=colors['background'],
        font_color=colors['text'],
    )
    return fig


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    # 'background': '#111111',
    'background': '#FFFFFF',
    'text': 'black',
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

df = stock_prices()
iso_fmt = '%Y-%m-%d'
df_daily = daily_return(df,
                        df.index[1].strftime(iso_fmt),
                        df.index[-1].strftime(iso_fmt))
# df_daily = daily_return(df, '1990-01-03', '2021-04-01')

range_init = ['2020-01-01', '2021-04-09']
# tickers_init = ['AAPL', 'AAL', 'ABBV']
tickers_init = ['AAPL']


app.layout = html.Div(children=[
    dcc.Markdown('''
        # Capital Asset Pricing Model (CAPM)
    '''),

    html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='dd-chart-ticker',
                    options=[
                        {'label': t, 'value': t}
                        for t in df.columns
                    ],
                    value=tickers_init,
                    multi=True,
                ),
                ], style={'width': '100%', 'float': 'left',
                          'display': 'inline-block'}
            ),
            # html.Div([
            dcc.Input(
                id='in-text-from',
                type='text',
                value=range_init[0],
                placeholder='From: 2020-01-01'
            ),
            dcc.Input(
                id='in-text-to',
                type='text',
                value=range_init[1],
                placeholder='To: 2021-01-01'
            ),
            html.Button(
                id='bt-range',
                n_clicks=0,
                children='apply'
            )
            ], style={'width': '49%', 'float': 'left',
                      'display': 'inline-block'}
        )
            # ], style={'width': '49%', 'float': 'left',
            #           'display': 'inline-block'}
        # )
    ], style={'width': '100%', 'display': 'inline-block'}
    ),

    html.Div([
        dcc.Graph(
            id='g-chart',
            figure=chart(profit(df, range_init[0], range_init[1]),
                         tickers_init)
        ),
    ], style={'width': '49%', 'float': 'left', 'display': 'inline-block'}
    ),

    html.Div([
        dcc.Graph(
            id='g-capm',
            # figure=capm_scatter(df, range_init[0],
            #                     range_init[1], tickers_init)
        ),
    ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
    )
])

@app.callback(
    dash.dependencies.Output('g-chart', 'figure'),
    dash.dependencies.Input('dd-chart-ticker', 'value'),
    dash.dependencies.Input('bt-range', 'n_clicks'),
    dash.dependencies.State('in-text-from', 'value'),
    dash.dependencies.State('in-text-to', 'value'),
)
def update_output(tickers, n_clicks, from_day, to_day):
    df_profit = profit(df, from_day, to_day)
    return chart(df_profit, tickers)

@app.callback(
    dash.dependencies.Output('g-capm', 'figure'),
    dash.dependencies.Input('dd-chart-ticker', 'value'),
    dash.dependencies.Input('bt-range', 'n_clicks'),
    dash.dependencies.State('in-text-from', 'value'),
    dash.dependencies.State('in-text-to', 'value'),
)
def update_output(tickers, n_clicks, from_day, to_day):
    return capm_scatter(df, df_daily, from_day, to_day, tickers)


if __name__ == '__main__':
    app.run_server(debug=True)

