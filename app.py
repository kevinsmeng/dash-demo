# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:36:55 2020

@author: ksmeng
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np


#############################################
# Constants and useful values
FONT_DEFAULT = 'Arial'
COLORS = {'background': '#01203A', 'text': '#FFFFFF', 'emphasis': '#00E4A2'}
LAYOUT_STYLE = {'fontFamily':FONT_DEFAULT}
TITLE_STYLE = {'margin':'10px', 'padding': '10px','color':COLORS['emphasis'],
               'backgroundColor':COLORS['background']}
MARKDOWN_STYLE = {'lineHeight':'80%', 'marginBottom':'50px'}
DD_STYLE = {'margin':'10px', 'padding': '10px', 'border':'2px solid #01203A'}
MARKER_SIZE_DEFAULT = 20 # Use it if data is None

SUCCESS_LABELS = ['Success','Stable (SF)','Stable','Failure','Missing']
SUCCESS_COLORS = ['green','blue','orange','red','grey']
DATE_MIN = pd.Timestamp(2017,11,1)
DATE_MAX = pd.Timestamp(2020,10,23)


#############################################

def get_data(file):
    """
    Input:
        - file: excel
    Returns pandas dataframe
    """
    df_data = pd.read_excel(file)
    df_data.fillna(-1, inplace=True)
    return df_data

def get_info(file):
    """
    Input:
        - file: csv or txt, delimited by semicolon
    Returns pandas dataframe
    """
    df_info = pd.read_csv(file,delimiter=';')
    df_info.fillna('', inplace=True)
    return df_info

def get_legends():
    """
    Input:
        - path: can include *
    Return dictionary of pandas dataframes
    """
    dict_legends = {}
    files = ['https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_demo_handedness.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_demo_institution.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_demo_intellectual_disability.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_demo_psychiatric_diagnosis.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_demo_psychiatric_history.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_demo_state.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_study_censor.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_study_change_aeds.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_study_effect_aeds.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_study_frequency_seizures.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_epilepsy.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_focal_aetiology.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_focal_lesional.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_generalised_aetiology.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_generalised_seizures.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_multifocal.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_refractory.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_refractory_onset.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_surgery_details.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_surgery_history.txt', 'https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/legends/legends_type_vns_history.txt']
    for file in files:
        key = file.split('/')[-1][8:].split('.')[0]
        dict_legends[key] = pd.read_csv(file,delimiter=';')
        dict_legends[key].fillna('', inplace=True)
    return dict_legends

def print_entry(df,n):
    for i in range(df.shape[1]):
        key = list(df.keys())[i]
        print(key+': '+str(df[key][n]))

def get_dropdown_value_options(type_filter, key):
    """
    Input:
        - type_filter: 'demo', 'type', 'drug'
        - key: column name in dataframe, e.g. 'demo_handedness'
    """
    if type_filter in ['demo','type']:
        value = ['-1']
        options = [{'label':'Data Missing', 'value':'-1'}]
        idx_key = dict_dds[type_filter].index(key)
        for i in range(1,dict_legends[dict_dds[type_filter][idx_key]].shape[0]):
            value.append(dict_legends[dict_dds[type_filter][idx_key]]['code'][i])
            options.append({'label':dict_legends[dict_dds[type_filter][idx_key]]['legend'][i],
                            'value':dict_legends[dict_dds[type_filter][idx_key]]['code'][i]})
    elif type_filter in ['drug']:
        value = key+'_changes'
        options = []
        for i in ['_cessation','_changes']:
            options.append({'label':key+i, 'value':key+i})
    elif type_filter in ['drug-multi']:
        value = ['Data Missing']
        options = [{'label':'Data Missing', 'value':'-1'}]
        legend_key = 'study_effect_aeds' if '_cessation' in key else 'study_change_aeds'
        for i in dict_legends[legend_key]['legend'][1:].tolist():
            value.append(i)
            options.append({'label':i, 'value':i})
#    print(value)
    return value, options

def generate_table(dataframe, max_rows=20):
    """
    Return table from Pandas dataframe
    """
    return html.Table(style={'width':'100%'}, children=[
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

#############################################

df_data = get_data('https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/data/data_clean.xlsx')
df_aeds = get_info('https://raw.githubusercontent.com/kevinsmeng/dash-demo/main/info/info_aeds.txt')
dict_legends = get_legends()
dict_dds = {} # dictionary of lists for dropdown options
for key in ['demo','type','frequ']:
    dict_dds[key] = [i for i in df_data if key in i]
id_default = 'SVHM064'
df_hoverData_default = {'points':[{'customdata':[id_default]}]} # default


#############################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(style=LAYOUT_STYLE,children=[

    # Left panel
    html.Div(style={'width':'20%', 'float':'left', 'display':'inline-block'},children=[  
        # Title label
        html.H4(style=TITLE_STYLE, children=['KeyLead Demo']),
        
        # Pie chart for success
        html.Div(style=DD_STYLE, children=[
            html.H4(children='Treatment success'),
            dcc.Dropdown(id='success-choices',multi=True,value=SUCCESS_LABELS,
                         options=[{'label':i, 'value':i} for i in SUCCESS_LABELS]),
            dcc.Graph(id='bar-success', style={'height': '30vh'})
        ]),
    
        # Dropdowns for demographics
        html.Div(style=DD_STYLE,children=[
            html.H4(children='Demographic filters'),
            dcc.Dropdown(id='demo-filters',value=dict_dds['demo'][1],
                         options=[{'label':i, 'value':i} for i in dict_dds['demo']]),
            dcc.Dropdown(id='demo-choices',multi=True,value=dict_legends[dict_dds['demo'][0]]['code'][1:].tolist(),
                         options=[{'label':dict_legends[dict_dds['demo'][0]]['legend'][i], 'value':dict_legends[dict_dds['demo'][0]]['code'][i]} for i in range(1,dict_legends[dict_dds['demo'][0]].shape[0])]),
        ]),
        # Dropdowns for epilepsy
        html.Div(style=DD_STYLE,children=[
            html.H4(children='Epilepsy filters'),
            dcc.Dropdown(id='epilepsy-filters',value=dict_dds['type'][0],
                         options=[{'label':i, 'value':i} for i in dict_dds['type']]),
            dcc.Dropdown(id='epilepsy-choices',multi=True)
        ]),
        # Dropdowns for treatment
        html.Div(style=DD_STYLE,children=[
            html.H4(children='Treatment filters'),
            dcc.Dropdown(id='treatment-drugs',value='brv',
                         options=[{'label':df_aeds['chemical name'][i]+' ('+df_aeds['abbreviation'][i]+')', 'value':df_aeds['abbreviation'][i]} for i in range(1,df_aeds.shape[0]-1)]),
            dcc.Dropdown(id='treatment-filters',value=dict_dds['type'][0],
                         options=[{'label':i, 'value':i} for i in dict_dds['type']]),
            dcc.Dropdown(id='treatment-choices',multi=True),
        ])
    ]),
            
    # Central panel
    html.Div(style={'width': '45%', 'float':'left', 'display':'inline-block'},children=[
        # Population scatter
        html.Div(style=DD_STYLE, children=[
            # Title for selected population
            dcc.Markdown(id='label-population-scatter',style=MARKDOWN_STYLE,children=['']),
            
            # Dropdown for data size and color
            html.Div(style={'verticalAlign':'middle', 'width':'50%', 'display': 'inline-block'},children=[
                html.Div(style={'verticalAlign':'middle', 'width':'30%', 'display': 'inline-block'},children=['Marker size:']),
                html.Div(style={'verticalAlign':'middle', 'width':'60%', 'display': 'inline-block'},children=[
                    dcc.Dropdown(id='data-size-choices',value='brv_dose',
                                 options=[{'label':i, 'value':i} for i in ['brv_dose']])
                ]),
                html.Div(style={'verticalAlign':'middle', 'width':'30%', 'display': 'inline-block'},children=['Marker color:']),
                html.Div(style={'verticalAlign':'middle', 'width':'60%', 'display': 'inline-block'},children=[
                    dcc.Dropdown(id='data-color-choices',value=dict_dds['frequ'][2],
                                 options=[{'label':i, 'value':i} for i in dict_dds['frequ']])
                ]),
            ]),
            # Slider for time points
            html.Div(style={'verticalAlign':'middle', 'width':'50%', 'display': 'inline-block'},children=[
                html.Div(style={'textAlign':'center', 'verticalAlign':'top'},children=['Time point (months):']),
                html.Div(children=[
                    dcc.Slider(id='study-slider',min=0,max=24,value=0,
                               marks={i: str(i) for i in range(0,25,3)},step=1)
                ])
            ]),
            # Graph for population scatter
            dcc.Graph(id='scatter-main',hoverData=df_hoverData_default)
        ]),
        # Population stats
        html.Div(style=DD_STYLE, children=[
            # Title for population stats
            html.H4(children='Population stats'),
            # Pie charts for demo stats
            html.Div(style={'width':'50%', 'display': 'inline-block'},children=[
                dcc.Graph(id='demo-chart')
            ]),
            html.Div(style={'width':'50%', 'display': 'inline-block'},children=[
                dcc.Graph(id='epilepsy-chart')
            ])
        ])
    ]),
    
    
    html.Div(style={'width': '30%', 'float':'left', 'display': 'inline-block'},children=[
        # Participant seizure plot
        html.Div(style=DD_STYLE, children=[
            # Title for selected participant
            dcc.Markdown(id='label2',style=MARKDOWN_STYLE,children=['']),
            # Text input for participant ID
            html.Div(style={'verticalAlign':'middle', 'width':'30%', 'display': 'inline-block'},children=['Participant ID:']),
            html.Div(style={'verticalAlign':'middle', 'width':'60%', 'display': 'inline-block'},children=[
                dcc.Input(id='participant-textinput',value=id_default,type='text')
            ]),
            # Dropdown for seizure type
            html.Div(style={'verticalAlign':'middle', 'width':'30%', 'display': 'inline-block'},children=['Seizure type:']),
            html.Div(style={'verticalAlign':'middle', 'width':'60%', 'display': 'inline-block'},children=[
                dcc.Dropdown(id='seizure-filter',value=dict_dds['frequ'][2],
                             options=[{'label':i, 'value':i} for i in dict_dds['frequ']])
            ]),
            # Line plot for seizure frequency evolution over time
            dcc.Graph(id='seizure-evolution')
        ]),
        # Participant info
        html.Div(style=DD_STYLE, children=[
            # Title for participant info
            html.H4(children='Participant info'),
            # HTML table summarising participant details
            html.Div(id='participant-table', children=[''])
        ])
    ]),
    
])
        
    
#############################################
        
@app.callback(
    [Output('demo-choices', 'value'),
     Output('demo-choices', 'options')],
    [Input('demo-filters', 'value')])
def update_demo_choices(key):
    if key is None:
        return None, []
    else:
        return get_dropdown_value_options('demo',key)

@app.callback(
    [Output('epilepsy-choices', 'value'),
     Output('epilepsy-choices', 'options')],
    [Input('epilepsy-filters', 'value')])
def update_epilepsy_choices(key):
    if key is None:
        return None, []
    else:
        return get_dropdown_value_options('type',key)

@app.callback(
    [Output('treatment-filters', 'value'),
     Output('treatment-filters', 'options')],
    [Input('treatment-drugs', 'value')])
def update_treatment_filters(key):
    if key is None:
        return None, []
    else:
        return get_dropdown_value_options('drug',key)
    
@app.callback(
    [Output('treatment-choices', 'value'),
     Output('treatment-choices', 'options')],
    [Input('treatment-filters', 'value')])
def update_treatment_choices(key):
#    print(key)
    if key is None:
        return None, []
    else:
        return get_dropdown_value_options('drug-multi',key)


def get_population_stats(key, list_pie):
    """
    This function is called in 'update_population_plots()' callback
    """
    # Convert list to pandas dataframe
    df_pie = pd.DataFrame({})
    df_pie[key] = pd.Series(list_pie)
    
    # Plot pie chart
    fig = px.pie(df_pie,names=key,title=key)
    fig.layout.font.family = FONT_DEFAULT
    fig.update_layout(showlegend=False)
    fig.update_traces(textinfo='percent') # possible to do textposition='inside'

    # Custom hover
    for ser in fig['data']:
        ser['hovertemplate']='''<b>%{label}</b><br>
%{percent} (%{value} participants)'''
    
    return fig


def get_success_color(label):
    """
    Returns color corresponding to label
    """
    idx = SUCCESS_LABELS.index(label)
    return SUCCESS_COLORS[idx]
    
def get_success_stats(data, list_success_values):
    """
    This function is called in 'update_population_plots()' callback
    """
    counts = []
    labels = list_success_values if list_success_values else SUCCESS_LABELS
    colors = [get_success_color(i) for i in labels]
    for i in labels:
        counts.append(0)
    for i in data:
        if i in labels:
            counts[labels.index(i)] += 1
    ncounts = sum(counts)
    if ncounts > 0:
        for i in range(len(counts)):
            counts[i] = round(counts[i]/ncounts*100, 2)
    
    # Plot bar chart of success
    fig = px.bar(x=labels,y=counts)
    fig.layout.font.family = FONT_DEFAULT
    fig.update_xaxes(title='')
    fig.update_yaxes(title='Success rate (%)', range=[0,100])
    fig.update_traces(marker={'color':colors})
    fig.update_layout(margin=dict(t=20,l=20,b=0,r=0), showlegend=True,
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    
    # Custom hover
    for ser in fig['data']:
        ser['hovertemplate']='''<b>%{x}</b><br>
Ratio: %{y}%'''

    return fig


def determine_success(freq, freq_ref, idx, list_idx, bool_color=True, threshold='50'):
    """
    This function determines whether treatment is a success
    Inputs:
        - freq: seizure frequency (categorical label) at time point of interest
        - freq_ref: seizure frequency (categorical label) at time point of reference
        - idx: index at time point of interest
        - list_idx: list of indices for participant of interest
        - threshold: value (percent) that determines whether treatment is a success or not
    """
    
    # Determine if treatment is a success at selected time point (for categorical labels 1-9)
    if freq in range(1,10) and freq_ref in range(1,10):
        # Success: seizure frequency decreased wrt start visit
        if freq > freq_ref:
            return 'green', 'Success'
        # Failure: seizure frequency increased wrt start visit
        elif freq < freq_ref:
            return 'red', 'Failure'
        # Stable (2 types available): seizure frequency same as start visit
        else:
            if freq == 9: # rare seizures at start visit and seizure free at follow-up
                return 'blue', 'Stable (SF)'
            else: # stable frequency, but not seizure free
                return 'orange', 'Stable'
            
    # Categorical labels outside 1-9 (-1: missing, 10: better, 11: worse)
    else:
        # Success: Categorical label 10 means better, but no frequency was reported (if it is a follow-up visit)
        if list_idx.index(idx) > 0 and freq == 10:
            return 'green', 'Success'
        # Failure: Categorical label 11 means worse, but no frequency was reported (if it is a follow-up visit)
        elif list_idx.index(idx) > 0 and freq == 11:
            return 'red', 'Failure'
        # Missing data: Other categorical labels not be considered (-1 represents missing data)
        else:
            return 'grey', 'Missing'

@app.callback(
    [Output('scatter-main', 'figure'),
     Output('label-population-scatter', 'children'),
     Output('demo-chart', 'figure'),
     Output('epilepsy-chart', 'figure'),
     Output('bar-success', 'figure')],
    [Input('study-slider', 'value'),
     Input('data-color-choices', 'value'),
     Input('success-choices', 'value'),
     Input('demo-filters', 'value'),
     Input('demo-choices', 'value'),
     Input('demo-choices', 'options'),
     Input('epilepsy-filters', 'value'),
     Input('epilepsy-choices', 'value'),
     Input('epilepsy-choices', 'options')])
def update_population_plots(nmonths, seizureType, list_success_values,
                            key_demo, list_demo_user_options, list_demo_possible_options,
                            key_type, list_type_user_options, list_type_possible_options):
    
    # Initialise a pandas series of bool (for filtering purpose)
    x = []; y = []; size = []; color = []; success_data = []; customdata = pd.DataFrame({})
    list_id_participant = []; list_idx_visit = []; list_nvisits = []; list_start_date = []; list_ndays = []
    if seizureType is None:
        seizureType = 'frequ_fbtcs'
    
    # Prepare list of participants to analyse (default exclusions)
    list_participants = list(df_data['id_participant'].unique())
    list_excluded_participants = ['SVHM057','SVHM109'] # based on start date, too early
    for id_participant in list_excluded_participants:
        list_participants.remove(id_participant)
        
    # Prepare filtering based on demographics dropdowns (user selection)
    list_demo_included_labels = []; list_demo_included_values = []; list_demo_pie = []
    if key_demo is None:
        key_demo = 'demo_handedness'
    for i in list_demo_possible_options:
        if i['value'] in list_demo_user_options:
            list_demo_included_labels.append(i['label'])
            list_demo_included_values.append(int(i['value'])) # be careful: int/str conversion
    
    # Prepare filtering based on epilepsy dropdowns (user selection)
    list_type_included_labels = []; list_type_included_values = []; list_type_pie = []
    if key_type is None:
        key_type = 'type_epilepsy'
    for i in list_type_possible_options:
        if i['value'] in list_type_user_options:
            list_type_included_labels.append(i['label'])
            list_type_included_values.append(int(i['value'])) # be careful: int/str conversion
            
    # Loop through participants to get last visit before slider mark (study date)
    for id_participant in list_participants:
        
        # Get indices of all rows corresponding to current participant
        list_idx = list(df_data[df_data['id_participant']==id_participant].index)
        
        # Boolean variables to determine whether to plot participant (based on dropdown filters)
        bool_demo_appear = df_data[key_demo][list_idx[0]] in list_demo_included_values # based on demographic filters
        bool_type_appear = df_data[key_type][list_idx[0]] in list_type_included_values # based on epilepsy filters
        bool_drug_appear = True #df_data[key_drug][list_idx[0]] in list_drug_included_values
        
        # Dataframe of bool variables to determine whether to plot participant (based on time point slider)
        df_ndays = df_data['date_of_visit'][list_idx] - df_data['date_of_visit'][list_idx[0]] # difference of pd.Timestamps is pd.Timedelta
        df_bool_appear = df_ndays > pd.Timedelta((nmonths-1)*30,'days') # based on time point and visit dates (appear if at least one visit after [nmonths-1])
        df_bool_plot = df_ndays <= pd.Timedelta(nmonths*30,'days') # this is to determine which time point to plot (plot last visit before [nmonths])
        
        # Append actual values to lists (for plotting and hovering purposes)
        if bool_demo_appear and bool_type_appear and bool_drug_appear and list(df_bool_appear[df_bool_appear].index):
            
            # demographic pie chart
            idx = list_demo_included_values.index(df_data[key_demo][list_idx[0]])
            list_demo_pie.append(list_demo_included_labels[idx])
            
            # epilepsy pie chart
            idx = list_type_included_values.index(df_data[key_type][list_idx[0]])
            list_type_pie.append(list_type_included_labels[idx])
            
            # main scatter plot
            if list(df_bool_plot[df_bool_plot].index):
                
                # Calculate visit number and number of days since enrolment
                idx = list(df_bool_plot[df_bool_plot].index)[-1] # get index of last visit before nmonths
                x.append(df_data['date_of_visit'][idx] if df_data['date_of_visit'][idx] < DATE_MAX else None)
                list_idx_visit.append(list_idx.index(idx)+1)
                list_ndays.append(df_ndays[idx].days)
                
                # Determine if treatment is a success at selected time point (see function)
                success_color, success_label = determine_success(df_data[seizureType][idx], df_data[seizureType][list_idx[0]], idx, list_idx)
                color.append(success_color)
                success_data.append(success_label) # this list is not for plotting purposes, therefore shorter than color
                
            # Print potential errors (it should not be the case)
            else:
                print(str(id_participant)+': bool_appear is True, but bool_plot is False')
                x.append(None)
                list_idx_visit.append(-1)
                list_ndays.append(-1)
                color.append('black')
                
        # Append None or -1 to lists (if data does not appear)
        else:
            x.append(None)
            list_idx_visit.append(-1)
            list_ndays.append(-1)
            color.append('grey')
        
        # Append actual values that do not depend on whether data appears or not
        y.append(df_data['age_of_onset'][list_idx[0]] if df_data['age_of_onset'][list_idx[0]] >= 0 else -1)
        size.append(df_data['brv_dose'][list_idx[0]] if df_data['brv_dose'][list_idx[0]] > 0 else MARKER_SIZE_DEFAULT)
        list_id_participant.append(id_participant)
        list_start_date.append(df_data['date_of_visit'][list_idx[0]].strftime('%Y-%m-%d') if df_data['date_of_visit'][list_idx[0]] < DATE_MAX else '')
        list_nvisits.append(len(list_idx))
    
    # Create pie charts from lists generated above
    fig_demo = get_population_stats(key_demo, list_demo_pie)
    fig_type = get_population_stats(key_type, list_type_pie)
    fig_success = get_success_stats(success_data, list_success_values)
    
    # Custom data for hover template: Column keys must be integers (not string)
    customdata[0] = pd.Series(list_id_participant)
    customdata[1] = pd.Series(list_idx_visit)
    customdata[2] = pd.Series(list_nvisits)
    customdata[3] = pd.Series(list_start_date)
    customdata[4] = pd.Series(list_ndays)
    customdata[5] = pd.Series(size)
    customdata[6] = pd.Series([d.strftime('%Y-%m-%d') if not d is None else '' for d in x])
    
    # Plot scatter with selected population
    fig = px.scatter(x=x,y=y,size=size,labels={'x':'Date of visit', 'y':'Age of onset'})
    fig.layout.font.family = FONT_DEFAULT
    fig.update_traces(marker={'color':color},customdata=customdata)
    fig.update_layout(hovermode='closest', margin=dict(t=20,l=20,b=0,r=0))
    fig.update_xaxes(range=[DATE_MIN, DATE_MAX])

    # Custom hover template (check %{x} for raw dates of visit; weird behavior sometimes happens)
    for ser in fig['data']:
        ser['hovertemplate']='''<b>%{customdata[0]}</b><br>
Age of onset: %{y} y.o.<br>
Brv dose: %{customdata[5]} mg<br>
Date of start: %{customdata[3]}<br>
Date of visit: %{customdata[6]}<br>
Time point: %{customdata[4]} days<br>
Visit number: %{customdata[1]}/%{customdata[2]}'''
    
    # Update population info
    plural1 = 's' if len(x)-sum(pd.Series(x).isnull()) > 1 else ''
    plural2 = 's' if nmonths > 1 else ''
    text_scatter = f'''
### Selected population\n
Population size: **{len(x)-sum(pd.Series(x).isnull())} participant{plural1}** (out of {len(x)} available)\n
Time point: **{nmonths} month{plural2}** from enrolment date
'''
    # Return fig to update in scatter graph
    return fig, text_scatter, fig_demo, fig_type, fig_success


@app.callback(
    Output('participant-textinput','value'),
    [Input('scatter-main', 'hoverData')])
def update_participant_textinput(hoverData):
    id_participant = hoverData['points'][0]['customdata'][0]
    return id_participant


@app.callback(
    [Output('seizure-evolution', 'figure'),
     Output('label2','children'),
     Output('participant-table', 'children')],
    [Input('seizure-filter', 'value'),
     Input('participant-textinput', 'value')])
def update_participant_plot(seizureType, id_participant):
    
    # Check if participant exists (if text input modified manually by user)
    # Participant will always exist if hovering (mouse over data point on main scatter)
    x = []; y = []; fs_text = []; customdata = pd.DataFrame({})
    if id_participant in df_data['id_participant'].unique():
        list_idx = list(df_data[df_data['id_participant']==id_participant].index)
    else:
        list_idx = []; x = [np.nan]; y = [np.nan]; fs_text = ['']
        
    # Loop through each row corresponding to selected participant
    for idx in list_idx:
        x.append(df_data['date_of_visit'][idx] if df_data['date_of_visit'][idx] < DATE_MAX else None)
        y.append(df_data[seizureType][idx] if not seizureType is None else np.nan)
        if list_idx.index(idx) == 0: # start visits (main legend)
            fs_text.append(dict_legends['study_frequency_seizures']['legend'][y[-1]] if y[-1] > 0 else 'Data missing')
        else: # follow-up visits (alternative legend)
            fs_text.append(dict_legends['study_frequency_seizures']['legend_alt'][y[-1]] if y[-1] > 0 else 'Data missing')
    y = [i if i in range(1,12) else np.nan for i in y] # -1: missing data, 12: not documented (if follow-up visits)
    
    # Returns a color that represents success at last visit available (for plotting purposes)
    color, _ = determine_success(df_data[seizureType][list_idx[-1]],df_data[seizureType][list_idx[0]],list_idx[-1],list_idx)
    
    # Custom data for hover template (seizure frequency labels are too long to show on y-axis)
    customdata[0] = pd.Series(fs_text) # fs_text: legend for seizure frequency
    
    # Plot line plot with selected participant
    fig = px.line(x=x,y=y,labels={'x':'Date of visit', 'y':'Seizure frequency'})
    fig.layout.font.family = FONT_DEFAULT
    fig.update_traces(line={'color':color}, marker={'color':color, 'size':MARKER_SIZE_DEFAULT}, mode='lines+markers', customdata=customdata)
    fig.update_yaxes(range=[0,12])
    fig.update_layout(margin=dict(t=20,l=20,b=0,r=0))
    
    # Custom hover template
    for ser in fig['data']:
        ser['hovertemplate']='''
Date of visit: %{x}<br>
Seizure frequency: %{customdata[0]}'''

    # Update participant info (label)
    id_participant_text = id_participant if id_participant in df_data['id_participant'].unique() else 'Not valid'
    plural1 = 's' if len(x) > 1 else ''
    text1 = f'''
### Selected participant\n
Participant ID: **{id_participant_text}**\n
Number of visits: **{len(x)} visit{plural1}**
'''

    # Update participant info (table)
    # Create list if the key matches available legends
    keys = df_data.keys().tolist()
    values = df_data.iloc[list_idx[0]].tolist()
    list_keys = []; list_values = []
    for i in range(len(keys)):
        if keys[i] in dict_legends:
            if values[i] == -1:
                list_keys.append(keys[i])
                list_values.append('Data Missing')
            else:
                a = dict_legends[keys[i]]['code']==str(values[i])
                idx = a[a].index
                list_keys.append(keys[i])
                list_values.append(dict_legends[keys[i]]['legend'][idx])
        elif keys[i] in ['age_of_onset','brv_dose']:
            list_keys.append(keys[i])
            list_values.append(str(values[i]))
    
    # Build dataframe to generate HTML table
    df_table = pd.DataFrame({'key':list_keys, 'value':list_values})
    df_table = df_table.sort_values('key', ascending=True)
    table = generate_table(df_table)
    
    return fig, text1, table


#############################################
        
if __name__ == '__main__':
    app.run_server(debug=True)
