from capm import capm, efficient_frontier, profit, stock_prices, daily_return
from capm import risk, _n_year, result_df
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pkg_resources import resource_stream
import pickle


def chart(df_profit: pd.DataFrame, tickers: list[str]) -> None:
    fig = px.line(df_profit, y=tickers,
                  labels={'value': 'Return (%)',
                          'variable': 'Symbol'},
                  # hover_name='variable',
                  custom_data=['variable', 'value'],
                  )
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        hovermode='x unified',
    )
    fig.update_traces(
        hovertemplate=''.join([
            '<b>%{customdata[0]}</b> %{customdata[1]:.2f}%',
            # 'date: %{x}',
            # 'return: %{customdata[1]:.2f}%',
            '<extra></extra>'
        ])
    )
    return fig


def capm_scatter(
        df_daily: pd.DataFrame, risk: pd.Series, profit: pd.Series,
        from_day, to_day, tickers: list[str]) -> None:
    df_capm = capm(risk, profit)
    res = None

    if len(tickers) > 1:
        target_profits = np.linspace(max(0., profit.min()),
                                     profit.max(), n_eff)
        res = efficient_frontier(
            df_daily[tickers], profit, target_profits, from_day, to_day)

    if res is not None:
        weights, risks, profits = res

    df_capm.reset_index(inplace=True)
    fig = px.scatter(
        df_capm[df_capm.Symbol.isin(tickers)], x='risk', y='return',
        color='Symbol',
        # size=np.ones(len(tickers)),
        # size_max=10,
        labels={'risk': 'Standard dev. of daily returns (%)',
                'return': 'Compound annual return (%)',
                'color': 'Symbol'},
        # hover_name='Symbol',
        custom_data=['Symbol', 'Name'],

    )
    fig.update_traces(
        marker={
                    'size': [15 for i in range(len(df_capm))],
                },
        hovertemplate="<br>".join([
            "<b>%{customdata[0]}</b>",
            "risk: %{x:.2f}%",
            "return: %{y:.2f}%",
            # "name: %{customdata[1]}",
            "<extra></extra>"
        ]),
        selector=dict(type='scatter'),
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
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        # height=600,
        # width=650,
    )
    return fig, res


def generate_table(df: pd.DataFrame, row_index, maxcol=12) -> html.Table:
    i = min(row_index, len(df)-1)
    ncol = min(len(df.columns), maxcol)
    return html.Table(children=[
        html.Thead(
            html.Tr([html.Th(df.columns[j]) for j in range(ncol)])
        ),
        html.Tbody([
            html.Tr([
                html.Td(f'{df.iloc[i][df.columns[j]]:.2f}')
                for j in range(ncol)
            ]),
        ])
    ],
        id='tab-res',
        style={'color': 'white'},
    )


external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]

colors = {
    'background': '#111111',
    # 'background': '#FFFFFF',
    'text': 'white',
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# global variables
iso_fmt = '%Y-%m-%d'
all_tickers = pickle.load(resource_stream('resources', 'tickers.pkl'))
n_eff = 20

# initial state
range_init = ['2020-01-01', '2021-04-09']
tickers_init = ['AAPL']

app.layout = html.Div(
    [
        dcc.Markdown('''
            # Modern Portfolio Theory (MPT)
        ''',
            className='twelve columns title'
        ),

        html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id='dd-chart-ticker',
                            options=[
                                {'label': t, 'value': t}
                                for t in all_tickers
                            ],
                            value=tickers_init,
                            multi=True,
                        ),
                    ],
                ),
                dcc.Input(
                    id='in-text-from',
                    type='text',
                    value=range_init[0],
                    placeholder='From: 2020-01-01',
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
            ],
            className='twelve columns cell'
        ),

        html.Div(
            [
                dcc.Graph(
                    id='g-chart',
                ),
            ],
            className='twelve columns cell'
        ),

        html.Div(
            [
                dcc.Graph(
                    id='g-capm',
                ),

                html.Div('Portfolio weights',
                        style={'background-color': 'black',
                               'color': 'white',
                               'font-weight': 'bold',
                               'font-size': 'normal',
                               'padding-left': '50px',
                               'margin-bottom': '-5px',
                               'font-size': 'larger'}),
                html.Div(
                    [
                        html.Table(id='tab-res'),
                        dcc.Slider(
                                    id='slider-res',
                                    min=1,
                                    max=n_eff,
                                    marks={i: str(i) for i in range(1, n_eff)},
                                    value=1,
                                )
                    ],
                    id='tab-weights',
                    style={'background-color': 'black', 'padding-left': '50px'}
                ),
            ],
            className='twelve columns cell'
        ),

        html.Div(id='store-weights', style={'display': 'none'})
    ],
    className="container",
)

@app.callback(
    dash.dependencies.Output('g-chart', 'figure'),
    dash.dependencies.Output('g-capm', 'figure'),
    dash.dependencies.Output('tab-weights', 'children'),
    dash.dependencies.Output('store-weights', 'children'),
    dash.dependencies.Input('dd-chart-ticker', 'value'),
    dash.dependencies.Input('bt-range', 'n_clicks'),
    dash.dependencies.State('in-text-from', 'value'),
    dash.dependencies.State('in-text-to', 'value'),
)
def update_output(tickers, n_clicks, from_day, to_day):
    df = stock_prices(tickers)
    df_daily = daily_return(df,
                            df.index[1].strftime(iso_fmt),
                            df.index[-1].strftime(iso_fmt))
    df_profit = profit(df, from_day, to_day)
    ser_risk = risk(df_daily, from_day, to_day)

    ser_profit = df_profit.iloc[-1]
    n_year = _n_year(from_day, to_day)
    ser_profit = (
            (1 + ser_profit/100.).pow(1. / n_year)
            - 1) * 100
    ser_profit.name = 'return'

    fig_capm, res = capm_scatter(
        df_daily, ser_risk, ser_profit,
        from_day, to_day, tickers)

    if res is not None:
        res_df = result_df(*res, tickers)
    else:
        weights = np.ones((1, 1), dtype='float')
        res_df = result_df(weights,
                           ser_risk.to_numpy(),
                           ser_profit.to_numpy(),
                           tickers)
    tab_weights = generate_table(res_df, 0)

    slider = html.Div([
        dcc.Slider(
            id='slider-res',
            min=1,
            max=n_eff,
            marks={i: str(i) for i in [1, 5, 10, 15, 20]},
            value=1,
        )],
        # className='twelve columns',
        style={
            'width': '34%', 'padding-bottom': '12px',
            'margin-top': '20px', 'margin-left': '-20px'}
    )

    return [
        chart(df_profit, tickers),
        fig_capm,
        [
            tab_weights,
            slider,
        ],
        res_df.to_json(date_format='iso', orient='split'),
    ]

@app.callback(
    dash.dependencies.Output('tab-res', 'children'),
    dash.dependencies.Input('slider-res', 'value'),
    dash.dependencies.Input('store-weights', 'children')
)
def update_output(slider, jsonified_res_df):
    res_df = pd.read_json(jsonified_res_df, orient='split')
    return generate_table(res_df, slider-1)


if __name__ == '__main__':
    app.run_server(debug=True)

