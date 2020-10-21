# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


#############################################
# Constants and useful values

LABELS_EXAMPLE = ['New York City',u'Montréal','San Francisco']
VALUES_EXAMPLE = ['NYC','MTL','SF']
VALUE_STRING_EXAMPLE = 'MTL'
VALUE_LIST_EXAMPLE = ['MTL','SF']
LIMITS_EXAMPLE = [0, 9]

COLORS_DEFAULT = {'background': '#01203A',
                  'text': '#FFFFFF',
                  'emphasis': '#00E4A2'}
TRANSITION_DEFAULT = 500
MARKDOWN_DEFAULT = '''
### Dash and Markdown
Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''

#############################################
# Reusable functions to generate HTML

def generate_h1(text='Interactive Dashboard'):
    """
    Return h1 from text
    Pure HTML
    """
    return html.H1(children=text,
                   style={'textAlign': 'center'})
    
def generate_div(text,colors=COLORS_DEFAULT):
    """
    Return div from text
    Pure HTML
    """
    return html.Div(children=text,
                    style={'textAlign': 'center', 'backgroundColor': colors['background'],
                           'fontSize':24, 'color': colors['emphasis']})

def generate_label(id='mylabel',text='Text missing'):
    """
    Return label from text
    Pure HTML
    """
    return html.Label(id=id,
                      children=text,
                      style={'marginTop':10, 'fontWeight':'bold'})

def generate_table(dataframe, max_rows=10):
    """
    Return table from Pandas dataframe
    Pure HTML
    """
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def generate_markdown(text=MARKDOWN_DEFAULT):
    """
    Dash syntax for Markdown text
    """
    return dcc.Markdown(children=text)

def generate_chart(fig,id='mychart'):
    """
    Return chart
    Interactive component generated with Javascript, HTML, CSS and React.js
    """
    return dcc.Graph(id=id, figure=fig)

def generate_fig(df,type_chart='bar',type_update='colors',
                 colors=COLORS_DEFAULT,transition=TRANSITION_DEFAULT):
    """
    Return a figure usable for chart
    """
    # Create figure
    if type_chart == 'bar':
        fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
        
    elif type_chart == 'scatter_df3':
        fig = px.scatter(df, x="gdp per capita", y="life expectancy",
                         size="population", color="continent", hover_name="country",
                         log_x=True, size_max=60)
    
    elif type_chart == 'scatter_df4':
        fig = px.scatter(df, x="gdpPercap", y="lifeExp", 
                         size="pop", color="continent", hover_name="country", 
                         log_x=True, size_max=55)
        
    # Update layout
    if type_update == 'colors':
        fig.update_layout(plot_bgcolor=colors['background'],
                          paper_bgcolor=colors['background'],
                          font_color=colors['text'])
    
    elif type_update == 'transition':
        fig.update_layout(transition_duration=transition) # smooth transition from previous figure

    return fig

def generate_slider(id='myslider',limits=LIMITS_EXAMPLE,marks=None,step=None):
    """
    Inputs:
        - limits: list of two int values
    """
    if marks is None:
        marks = {i: str(i) for i in range(limits[0],limits[1]+1)}
    return dcc.Slider(id=id,min=limits[0],max=limits[1],value=limits[0],
                      marks=marks,step=step)

def generate_widget(id='mywidget',labels=LABELS_EXAMPLE,values=VALUES_EXAMPLE,
                    value_string=VALUE_STRING_EXAMPLE,value_list=VALUE_LIST_EXAMPLE,
                    type_widget='dropdown',type_input='text'):
    """
    Inputs:
        - labels: list
        - values: list
        - value_default: str (among values) - can be a list if bool_multi is True
        - bool_multi: multi-select options
    Variables:
        - options: a list of dictionaries (keys are 'label' are 'value')
    """
    options = []
    for i in range(len(labels)):
        options.append({'label':labels[i], 'value':values[i]})
        
    if type_widget == 'dropdown':
        return dcc.Dropdown(id=id,options=options,value=value_string,multi=False)
    elif type_widget == 'dropdown-multi':
        return dcc.Dropdown(id=id,options=options,value=value_list,multi=True)
    elif type_widget == 'radio':
        return dcc.RadioItems(id=id,options=options,value=value_string)
    elif type_widget == 'checkboxes':
        return dcc.Checklist(id=id,options=options,value=value_list)
    elif type_widget == 'input':
        return dcc.Input(id=id,value=value_string,type=type_input)


#############################################
# Create own variables
df1 = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

#############################################
# Download variables from external files
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df2 = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')
df3 = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')
df4 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

#############################################
# Define app that generates HTML code

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(style={'fontFamily':'Courier', 'marginBottom': 50},
                      children=[
    generate_h1(),
    generate_div(['***',html.Br(),'Bar chart']),
    generate_chart(fig=generate_fig(df=df1,type_chart='bar',type_update='colors'),id='mybar'),
    generate_markdown(),
    generate_div('Scatter chart'),
    generate_chart(fig=generate_fig(df=df3,type_chart='scatter_df3',type_update=None),id='myscatter'),
    generate_div('Table'),
    generate_table(df2),
    generate_div('Static widgets'),
    html.Div(style={'columnCount': 2},children=[
        generate_label(id='label-mydropdown',text='Dropdown'),
        generate_widget(id='mydropdown',type_widget='dropdown'),
        generate_label(id='label-mydropdown-multi',text='Multi-Selection Dropdown'),
        generate_widget(id='mydropdown-multi',type_widget='dropdown-multi'),
        generate_label(id='label-myradio',text='Radio'),
        generate_widget(id='myradio',type_widget='radio'),
        generate_label(id='label-mycheckboxes',text='Checkboxes'),
        generate_widget(id='mycheckboxes',type_widget='checkboxes'),
        generate_label(id='label-mytextinput',text='Text Input'),
        generate_widget(id='myinput',type_widget='input'),
        generate_label(id='label-myslider',text='Slider'),
        generate_slider(id='myslider')
    ]),
    generate_div('Interactive widgets'),
    html.Div(style={'columnCount': 1},children=[
        html.H6('Change the value in the text input:'),
        generate_widget(id='interactive-input-1',value_string='Default',type_widget='input'),
        html.Br(),
        generate_label(id='interactive-output-1',text=''),
        html.H6('Move the slider to update the graph:'),
        generate_slider(id='interactive-input-2',limits=[df4['year'].min(), df4['year'].max()],
                        marks={str(year): str(year) for year in df4['year'].unique()}),
        generate_chart(fig=generate_fig(df=df4,type_chart='scatter_df4',type_update='transition'),id='interactive-output-2')
    ]),
    generate_div(['***',html.Br(),'Just an empty footer'])
])


#############################################
# Define app callbacks

@app.callback(
    Output(component_id='interactive-output-1', component_property='children'),
    [Input(component_id='interactive-input-1', component_property='value')]
) # function is called when app starts (overwrite specified values if any)
def update(value): # function and argument can have any name
    return f'Output: {value}'

@app.callback(
    Output('interactive-output-2', 'figure'),
    [Input('interactive-input-2', 'value')])
def update_figure(selected_year):
    df4_filt = df4[df4.year == selected_year]
    return generate_fig(df=df4_filt,type_chart='scatter_df4',type_update='transition')

if __name__ == '__main__':
    # app.run_server(dev_tools_hot_reload=False) to de-active hot-reloading
    app.run_server(debug=True)

