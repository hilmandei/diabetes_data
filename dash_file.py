import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash.dependencies import Output, Input, State
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.figure_factory as ff
from scipy.stats import ttest_ind
import dash_table
import pickle


model = pickle.load(open('randomforest_final', 'rb'))
df = pd.read_csv("diabetes_data_upload.csv")
col_predictor = list(df.columns)
col_predictor.remove('class')
col0 = list(df.columns)
col0.remove('Age')
col1 = col0.copy()
col1.remove('class')
col2 = col1.copy()
col2.remove('Gender')

df_heatmap = pd.read_csv('relationship_plot.csv')
df_heatmap.set_index('Unnamed: 0', inplace=True)

df_roc = pd.read_csv('df_roc.csv')
df_perf = pd.read_csv('model_perf.csv')

# https://www.bootstrapcdn.com/bootswatch/
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])
app.config['suppress_callback_exceptions'] = True

relation_text = """Since this is not Continous variable, we will try to find the assosiation with CramerV approach, 
Association is similarly to correlation, the output is in the range of [0,1], where 0 means no association and 1 is full association. 
(Unlike correlation, there are no negative values, as there’s no such thing as a negative association. Either there is, or there isn’t)
You can find out on this good articel as my reference below.
"""

about_data = """This Data has 570 entries, there is no missing value, so we dont need to impute any value.
                And it contains 16 Categorical Feature, and 1 Continues Feature (Age). Categorical Feature includes Gender and 15 Symtomps,
                based on this Feature we would like to predict the about Class feature (Positive or Negative Diabetes).
                You can download dataset below.
                """

model_text = "RandomForest Model gives us the best result, we will use it as our Classifier to make a prediction about Diabetes or Not"
link = "https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9"
link_download = "https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset."

first_card = dbc.Card(dbc.CardBody([html.P(relation_text, style={"font-size": "11pt"}), html.A('Clink Here', href=link, target="_blank")]))
second_card = dbc.Card(dbc.CardBody(html.P(model_text, style={"font-size": "11pt"}), style={'padding': '5px', 'margin': '0px'}))


def plot_paretochart(df, listcolomn, gender):
    if gender is None:
        gender = 'All Gender'

    text_title = 'Positive Class vs Yes-Symtomps in ' + gender
    df_pareto = pd.DataFrame({'Symtomp': listcolomn})

    if gender.upper() == "ALL GENDER":
        condition = (df['class'] == 'Positive')

    elif gender.upper() == 'MALE':
        condition = (df['class'] == 'Positive') & (df.Gender == 'Male')

    else:
        condition = (df['class'] == 'Positive') & (df.Gender == 'Female')

    df_pareto['Yes_case'] = df_pareto.Symtomp.apply(lambda x: df[x][(df[x] == 'Yes') & condition].count())
    df_pareto.sort_values(by='Yes_case', ascending=False, inplace=True)
    df_pareto['Pct_cumsum'] = round(df_pareto['Yes_case'].cumsum() / sum(df_pareto['Yes_case']) * 100)

    barplot = go.Bar(
        x=df_pareto['Symtomp'],
        y=df_pareto['Yes_case'],
        name='Symtomps',
        marker=dict(color='rgb(34,163,192)'))

    lineplot = go.Scatter(
        x=df_pareto['Symtomp'],
        y=df_pareto['Pct_cumsum'],
        name='Cumulative Percentage',
        yaxis='y2')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(barplot)
    fig.add_trace(lineplot, secondary_y=True)
    fig.update_layout({"margin": dict(l=0, r=0, t=30, b=0, pad=0),
                    "showlegend": False,
                    "plot_bgcolor": "#f2f2f2",
                    "paper_bgcolor" : "rgb(255,255,255)",
                    "font": {"color": "black"},
                    "titlefont": {"size": 13},
                    'title': {
                         'text': text_title.upper(),
                         'y': 0.985,
                         'x': 0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'},
                        "autosize": True, "width":480, "height": 320,
                       'xaxis': {'tickangle': -90}})


    return fig

def plot_piechart(df, listcolumn):
    list_pie_chart = []
    a = 0
    b = 1
    for i in range(8):
        plot = dbc.Row([
            dcc.Graph(figure=px.pie(data_frame=df, names=listcolumn[a], template='seaborn', title=listcolumn[a]).update_layout(
                    {"margin": dict(l=10, r=0, t=0, b=0, pad=0),
                    "showlegend": True,
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "font": {"color": "black" },
                     "titlefont": {
                         "size": 8},
                    'title': {
                         'text': col0[a].upper(),
                         'y': 0.9,
                         'x': 0.32,
                         'xanchor': 'center',
                         'yanchor': 'top'},
                    "autosize": False,
                    "width": 215, "height": 215}
            ).update_traces(textposition='inside', texttemplate='%{percent:.0%f}', pull=[0.04, 0])),
            dcc.Graph(figure=px.pie(data_frame=df, names=listcolumn[b], template='seaborn', title=listcolumn[b]).update_layout(
                    {"margin": dict(l=0, r=0, t=0, b=0, pad=0),
                    "showlegend": True,
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "font": {"color": "black"},
                     "titlefont": {
                         "size": 8},
                     'title': {
                         'text': col0[b].upper(),
                         'y': 0.9,
                         'x': 0.32,
                         'xanchor': 'center',
                         'yanchor': 'top'},
                    "autosize": False, 'width': 215, "height": 215}
                 ).update_traces(textposition='inside', texttemplate='%{percent:.0%f}', pull=[0.04, 0]))],
                       style={'height': '200px'}, no_gutters=True, justify="between")

        list_pie_chart.append(plot)
        a += 2
        b += 2

    return list_pie_chart

def plot_dist(df, colname, condition):
    alpha = 0.05
    if colname == None:
        colname = 'Gender'

    unique_val = df[colname].unique()

    if condition.lower() == 'positive':
        list_value = [df.Age[(df['class'] == 'Positive') & (df[colname] == item)] for item in unique_val]
        text_title = f"Age's Dist for {colname} in <Positive Class>"

        _, pval = ttest_ind(list_value[0],list_value[1])

    else:
        list_value = [df.Age[df[colname] == item] for item in unique_val]
        text_title = f"Age's Dist for {colname} in <All Class>"

        _, pval = ttest_ind(list_value[0], list_value[1])

    if pval < alpha:
        text_hypotesis = "REJECT null Hypotesis, significant in average Age"
    else:
        text_hypotesis = "ACCEPT null Hypotesis, no significant in average Age"

    fig = ff.create_distplot(list_value, group_labels=unique_val, bin_size=10,
                             show_rug=False, curve_type='kde')
    fig.update_layout(
        {"margin": dict(l=0, r=0, t=30, b=35, pad=0),
         "template": 'plotly',
         "plot_bgcolor": "#f2f2f2",
         "paper_bgcolor": "rgb(255,255,255)",
         "showlegend": True,
         "font": {"color": "black"},
         "titlefont": {"size": 10},
         'title': {
             'text': text_title.upper(),
             'y': 0.97,
             'x': 0.53,
             'xanchor': 'center',
             'yanchor': 'top'},
         "autosize": True, "width": 395, "height": 260,
         }, legend = dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99))
    fig.add_annotation(text=text_hypotesis, xref="paper", yref="paper", x=0.08, y=-0.182, showarrow=False, font={'size': 11})

    return fig

def plot_heatmap(df_heatmap):
    z_text = np.around(df_heatmap.values, decimals=1)
    fig = ff.create_annotated_heatmap(df_heatmap.values, x=df_heatmap.index.tolist(), y=df_heatmap.index.tolist(),
                                      colorscale='Viridis', annotation_text=z_text)
    fig.update_layout(
        {"margin": dict(l=0, r=0, t=23, b=0, pad=0),
         "plot_bgcolor": "#f2f2f2",
         "paper_bgcolor": "rgb(255,255,255)",
         "template": 'plotly',
         "font": {"color": "black", 'size': 8},
         "autosize": True, "width": 515, "height": 450,
         "titlefont": {"size": 13},
         'title': {
             'text': "relationship between caterogical".upper(),
             'y': 0.99,
             'x': 0.56,
             'xanchor': 'center',
             'yanchor': 'top'}
         })
    fig.update_xaxes(side="bottom")

    return fig

def plot_hist(df, columnname):
    if columnname is None:
        columnname = 'Gender'

    text = f"Portion Class - {columnname}"
    fig = px.histogram(df, x="class", color=columnname, barmode='group')
    fig.update_layout({"margin": dict(l=0, r=0, t=30, b=0, pad=0),
                       "showlegend": False,
                       "plot_bgcolor": "#f2f2f2",
                       "paper_bgcolor": "rgb(255,255,255)",
                       "font": {"color": "black"},
                       "titlefont": {"size": 13},
                       'title': {
                           'text': text.upper(),
                           'y': 0.985,
                           'x': 0.5,
                           'xanchor': 'center',
                           'yanchor': 'top'},
                       "autosize": True, "width": 310, "height": 310
                       })
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    return fig

def plot_roc(df_roc):
    line_roc = px.line(df_roc, x='fpr', y='tpr', color='model')
    line_roc.add_scatter(x=[0, 1], y=[0, 1], showlegend=False, marker={"color": 'red'})
    line_roc.update_layout(
        {"margin": dict(l=0, r=0, t=23, b=0, pad=0),
         "plot_bgcolor": "#f2f2f2",
         "paper_bgcolor": "rgb(255,255,255)",
         "template": 'plotly',
         "font": {"color": "black", 'size': 11},
         "autosize": True, "width": 515, "height": 450,
         "titlefont": {"size": 13},
         'title': {
             'text': "model perfomance".upper(),
             'y': 0.98,
             'x': 0.56,
             'xanchor': 'center',
             'yanchor': 'top'}},
        legend=dict(
                yanchor="top",
                y=0.30,
                xanchor="right",
                x=0.97))

    line_roc.update_xaxes(range=(-0.03, 1.03))
    line_roc.update_yaxes(range=(-0.03, 1.03))

    for trace in line_roc.data:
        modelname = trace.name
        if modelname is not None:
            auc_score = round(df_perf.ROC[df_perf.Model == modelname].values[0],3)
            updated_name = modelname + f' (AUC:{auc_score})'
            trace.name = updated_name

    return line_roc

def plot_table(modelname):
    data_model = dict(df_perf.set_index('Model').T[modelname])
    table = dash_table.DataTable(
        id='table_plot',
        columns=[{
            'name': i,
            'id': i
        } for i in ('Matrix', 'Value')],
        data=[
            {i: str(j) if i == 'Matrix' else round(data_model[j], 3) for i in ('Matrix', 'Value')} for j in data_model
        ],
        editable=False,
        style_header={
            'backgroundColor': 'rgb(34,163,200)',
            'fontWeight': 'bold', 'font_size': '14px'},
        style_cell={'font_size': '12px'}
    )

    return table

def create_input(features):
    list_button = []
    list_state = []
    list_output = []
    for ft in features:
        if ft == 'Age':
            id_data = "input_{}".format(ft)
            type_val = 'number'
            data = dcc.Input(
                    id=id_data,
                    type=type_val,
                    placeholder="{} (number)".format(ft), min=1,
                    style={'width': '200px', 'height': '30px'})

        else:
            unique_val = df[ft].unique()
            id_data = "input_{}".format(ft)
            data = dcc.Dropdown(
                id=id_data,
                options=[{'label': x, 'value': x} for x in unique_val],
                placeholder=ft,
                style={'width': '200px', 'height': '30px'},)

        state_data = State(id_data, 'value')
        output_ = Output(id_data, 'value')

        list_output.append(output_)
        list_state.append(state_data)
        list_button.append(data)

    return list_button, list_state, list_output


# Layout section: Bootstrap (https://hackerthemes.com/bootstrap-cheatsheet/)
# ************************************************************************

data_input, state_input, output_data = create_input(col_predictor)

tab_1_information = dbc.Row([
            # column 1
            dbc.Col(plot_piechart(df, col0), width=3.5, className="pretty_container_2"),

            # column 2
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dcc.RadioItems(id='my-dpdn', value='All Gender', inputStyle={"margin-left": "10px", 'margin-right': '3px'},
                                       labelStyle={'display': 'inline-block'},
                                       options=[{'label': x, 'value': x} for x in ('All Gender', 'Female', 'Male')]
                                     , style={"width": 480}),
                        dcc.Graph(id='pareto1', figure={})], className='pretty_container',
                        style={'margin-bottom': '10px'}),
                    dbc.Col([
                        dcc.Dropdown(id='my-dpdn3', multi=False, value='Gender',
                                     options=[{'label': x, 'value': x} for x in col1]
                                     , style={"width": 320}),
                        dcc.Graph(id='histplot', figure={})], className='pretty_container',
                        style={'margin-bottom': '10px', 'margin-left': '5px'})
                ], no_gutters=True, justify="between"),

                dbc.Col([
                    dcc.Dropdown(id='my-dpdn2', multi=False, value='Gender',
                                 options=[{'label': x, 'value': x} for x in col1], style={'margin-bottom': '5px'}),
                    dbc.Row([dcc.Graph(id='kdeplot', figure={}, style={'margin': '0px', 'padding': '0px'}),
                             dcc.Graph(id='kdeplot2', figure={}, style={'margin': '0px', 'padding': '0px'})],
                            style={'margin': '0px', 'padding': '0px'}, justify="between")],
                    className='pretty_container', style={'margin-bottom': '10px'}),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=plot_heatmap(df_heatmap)), width=530),
                    dbc.Col(first_card, style={'width': 250, 'height': 450, 'margin': '5px', 'padding': '0px'})],
                    style={'height': 455, 'margin-bottom': '10px'}, no_gutters=True, justify="between",
                    className='pretty_container'),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=plot_roc(df_roc)), width=530),
                    dbc.Col([
                        html.P('Select Model to See Evaluation Matrix', style={'font-size': 14},
                               className='pretty_container'),
                        dcc.Dropdown(id='my-dpdn4', multi=False, value='RandomForestClassifier',
                                     options=[{'label': x, 'value': x} for x in df_perf.Model],
                                     style={'margin-bottom': '5px'}),
                        html.Div(id='table_div'),
                        dbc.Col(second_card, style={'margin-top': '10px', 'padding': '0px'})],
                        style={'width': 250, 'height': 450, 'margin': '5px', 'padding': '0px'}, )],
                    style={'height': 455}, no_gutters=True, justify="between", className='pretty_container')
            ],
                width='auto')

        ], no_gutters=True, justify='center')

tab_2_prediction = html.Div([
                dbc.Row(html.A(html.Img(src="assets/diabetes.png", alt='check', width="500", height="260", style={"border": "solid", "margin-top":'15px', 'margin-bottom':'5px'},
                                        className ='pretty_container'),
                               href="https://jnyh.medium.com/building-a-machine-learning-classifier-model-for-diabetes-4fca624daed0", target='_blank'),
                        justify="center"),

                dbc.Row([dbc.Col(data_input[:8], width=1.9, className="pretty_container"), dbc.Col(data_input[8:], width=1.9,
                        style={'margin-left': '17px'}, className="pretty_container")], style={'margin-top': '5px', 'margin-bottom': '10px'}, justify="center", no_gutters=True),

                dbc.Row([html.Button('Predict', id='submit-val',  n_clicks=0, className="btn btn-primary", style={'margin': '5px'}),
                        html.Button('Reset', id='reset-val', n_clicks=0, className="btn btn-info", style={'margin': '5px'})], justify='center', no_gutters=True),

                dbc.Row(html.Div(id='container-button-basic', className='pretty_container', style={'width': '250px', 'border': 'solid', 'margin': '5px', 'text-align':'center'}), justify='center', no_gutters=True),
                            ])


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Tabs(id='tabs-example', value='tab-2', children=[
                dcc.Tab(label='Information', value='tab-1', style={'padding': '10pt', 'fontWeight': 'bold'}, className='pretty_container'),
                dcc.Tab(label='Prediction', value='tab-2', style={'padding': '10pt', 'fontWeight': 'bold'}, className='pretty_container')],
                         style={'height': 50, 'font-size': '10pt', 'margin-left': '18px'}), width=1.7),
        dbc.Col(id='text_header'),
        dbc.Button(
            "About Data", id="popover-bottom-target", color="secondary",
            style={'height': 40, 'font-size': '10pt', 'fontWeight': 'bold', 'margin-right': '22px'}),
        dbc.Popover(
            [
                dbc.PopoverHeader("All About Data :"),
                dbc.PopoverBody([about_data, html.Br(), html.A('Download', href=link_download, target="_blank")])
            ],
            id="popover",
            target="popover-bottom-target",  # needs to be the same as dbc.Button id
            placement="left",
            is_open=False,

        )
    ]),
    html.Div(id='tabs-content')],
    fluid=True, style={'margin': '0px', 'padding': '0px'})

# Callback section: connecting the components
# ************************************************************************

@app.callback([Output('tabs-content','children'),
               Output('text_header', 'children')],
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        text_header = "DIABETES DATA"
        div = html.H1(text_header,  className='text-center text-primary mb-1')

        return tab_1_information, div

    elif tab == 'tab-2':
        text_header = "DATA PREDICTION"
        div = html.H1(text_header, style={'margin-right': '13%'}, className='text-center text-primary mb-1')

        return tab_2_prediction, div

@app.callback(
    Output("popover", "is_open"),
    [Input("popover-bottom-target", "n_clicks")],
    [State("popover", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    [Output('container-button-basic', 'children')] + output_data,
    [Input('submit-val', 'n_clicks'), Input('reset-val', 'n_clicks')],
    state_input)
def update_output(n_clicks, x_click, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    update_blank = [None for i in range(len(col_predictor))]
    list_value = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16]

    dict_data = dict(zip(col_predictor, list_value))
    dict_predict = {1: 'POSITIVE', 0: 'NEGATIVE'}

    if 'submit-val' in changed_id or changed_id == '.':
        if n_clicks == 0:
            text_output = ["Press Predict to Check Result"]
            return text_output + update_blank

        else:
            if None in list_value:
                text_output = ["Please Fill all Input"]
                return text_output + list_value
            else:
                predict_data = pd.DataFrame(data=dict_data, index=[0])
                pred = model.predict(predict_data)[0]

                return [dict_predict[pred]] + update_blank
    else:
        text_output = ["Press Predict to Check Result"]
        return text_output + update_blank



@app.callback(
    [Output('pareto1', 'figure'), Output('kdeplot', 'figure'), Output('kdeplot2', 'figure'), Output('histplot', 'figure'),
     Output('table_div', 'children')],
    [ Input('my-dpdn', 'value'), Input('my-dpdn2', 'value'), Input('my-dpdn3', 'value'), Input('my-dpdn4', 'value')])
def update_graph(gender_val, colname, colhist, modelname):
    pareto = plot_paretochart(df, col2, gender=gender_val)
    kdeplot = plot_dist(df, colname, condition='positive')
    kdeplot2 = plot_dist(df, colname, condition='all')
    histplot = plot_hist(df, colhist)

    if modelname is None:
        modelname = 'RandomForestClassifier'

    table = plot_table(modelname)

    return pareto, kdeplot2, kdeplot, histplot, table


if __name__ == '__main__':
    app.run_server(debug=True, port=8100)

    
# https://youtu.be/0mfIK8zxUds
