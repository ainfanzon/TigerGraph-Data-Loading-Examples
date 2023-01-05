from IPython.display import display, Markdown, clear_output, HTML
from PIL import Image
import plotly.express as px
import ipywidgets as widgets
import psycopg2
import pandas as pd
import socket
import json
import getpass
import random
import dash
import pyTigerGraph as tg
import re
import matplotlib.pyplot as plt
#import dash_cytoscape as cyto
from dash.dependencies import Input, Output
from dash import dcc, html
#from jupyter_dash import JupyterDash
#from dash import dcc
#from dash import html
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.graph_objects as go
import networkx as nx

# Initialize variables
pd.set_option('display.max_colwidth', None)

t01 = widgets.Text(value='', placeholder='Enter Client ID', description='Client ID:', disabled=False)
t02=widgets.Select(options=['MutualFund', 'Stock', 'ETF'],value='MutualFund',rows=3, description='Investment:',disabled=False)
b01 = widgets.Button(description='Search')
b02 = widgets.Button(description='Search')

out = widgets.Output()

# Get IP address

def what_is_my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
    return s.getsockname()[0]

# Connect to the TigerGraph database
def connect(h,g,u,p):
    return tg.TigerGraphConnection(
          host=h
        , graphname=g
        , username=u
        , password=p) 

def pg_connect(h,prt,d,u,pw):
    return psycopg2.connect(
          host=h
        , port=prt 
        , database=d
        , user=u
        , password=pw) 

def showCustomer(df):
    html = '''
         <table class="container">
        <thead>
          <tr>
          <th><h1>''' + df['first'].to_string(index=False) + ' ' + df['middle'].to_string(index=False) +  ' ' + df['last'].to_string(index=False) +  '''</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Address: ''' + df['address_1'].to_string(index=False) + '''<br>
                ''' + df['address_2'].to_string(index=False) + '''<br>
                ''' + df['city'].to_string(index=False) + ' ' + df['state'].to_string(index=False) + ' ' + df['zip'].to_string(index=False) + '''<br>
                Telephone: ''' + df['phone'].to_string(index=False) + '''</td>
            <td>    </td>
            <td>Birthdate: ''' + df['fulldate'].to_string(index=False) + '''</br>
                Age ''' + df['age'].to_string(index=False) + '''<br>
                    ''' + df['maritalStatus'].to_string(index=False) + '''<br>
                    ''' + df['sex'].to_string(index=False) + '''</td>
          </tr>
          <tr>
            <td>Title: ''' + df['Employment'].to_string(index=False) + '''<br>
                Dependent(s): ''' + df['kids'].to_string(index=False) + '''</td>
            <td>   </td>
            <td>Interests: ''' + df['Interest'].to_string(index=False) + '''<br>
                Social Media: ''' + df['socialPlatform'].to_string(index=False) + '''<br>
                Social Handle: ''' + df['socialhandle'].to_string(index=False) + '''<br>
                Hobbies: ''' + df['Hobbies'].to_string(index=False) + '''</td>
          </tr>
        </tbody>
      </table>
      '''
    display(widgets.HTML(value=html))
        
def on_button_clicked(_):
    # "linking function with output"
    gsql='SELECT * FROM Client WHERE primary_id == "' + t01.value + '"'
    df1=pd.DataFrame.from_records(flatten1(json.loads(conn.gsql(gsql))))
    with out:
        # what happens when we press the button
        clear_output()
        showCustomer(df1)
        
# END OF WIDGETS DEFINITION

# Function to parse TG query output
def flatten(obj):
    output = []
    for e in obj:
        element = {}
        element["v_id"] = int(e["v_id"])
        element["v_type"] = e["v_type"]
        for k in e["attributes"]:
            element[k] = e["attributes"][k]
        output.append(element)
    return output
# END parsing JSON output

def flatten1(obj):
    output = []
    for e in obj:
        element = {}
        element["v_id"] = e["v_id"]
        element["v_type"] = e["v_type"]
        for k in e["attributes"]:
            element[k] = e["attributes"][k]
        output.append(element)
    return output

def re_graph(df4):
    
    for index, row in df4.iterrows():
        df5 = df4.assign(source = 'C00000002', target = df4['v_id'])

    A = list(df5["source"].unique())
    B = list(df5["target"].unique())

    node_list = list(set(A+B))
    
    # add all the nodes into the graph using the add_node method. 
    G = nx.Graph()
    for i in node_list:
        G.add_node(i)
    
    # add edges to the graph
    for i,j in df5.iterrows():
        G.add_edges_from([(j["source"],j["target"])])
    
    # assign some position to each node for plotting
    pos = nx.spring_layout(G, iterations=1)
    for n, p in pos.items():
        G.nodes[n]['pos'] = p
    
    # plot using plotly
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.75,color='#888'),
        hoverinfo='none',
        mode='lines')
    
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))
    
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
    
    # assign annotations to the nodes and edges
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])
    
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
    
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hovertext=node_list,
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Customer Relationships',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Connections",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def re_graph(df5):
    
    for index, row in df4.iterrows():
        df5 = df4.assign(source = 'from_id', target = df4['to_id'])

    A = list(df5["source"].unique())
    B = list(df5["target"].unique())

    node_list = list(set(A+B))
    
    # add all the nodes into the graph using the add_node method. 
    G = nx.Graph()
    for i in node_list:
        G.add_node(i)
    
    # add edges to the graph
    for i,j in df5.iterrows():
        G.add_edges_from([(j["source"],j["target"])])
    
    # assign some position to each node for plotting
    pos = nx.spring_layout(G, iterations=1)
    for n, p in pos.items():
        G.nodes[n]['pos'] = p
    
    # plot using plotly
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.75,color='#888'),
        hoverinfo='none',
        mode='lines')
    
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))
    
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
    
    # assign annotations to the nodes and edges
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])
    
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
    
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hovertext=node_list,
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Related Categories/Attributes',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Connections",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig