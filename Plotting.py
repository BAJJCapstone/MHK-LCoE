
# coding: utf-8

# In[2]:

import plotly.plotly as py
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

from NOAAStations import TidalStation

test_station = TidalStation(8447191)

time, height = test_station.predictWaterLevels(24*30)


# In[12]:


trace = go.Scatter(
    x = time/24,
    y = height,
    mode = 'lines',
    name = 'lines',
    line = dict(color = 'rgb(52, 165, 218)')
)

layout = go.Layout(
    title = 'Water Level Height | Bournedale, Cape Cod Canal',
    titlefont = dict(
        size = 26,
        color = 'rgb(131, 135, 135)'),
    xaxis = dict(title = 'Time (Days)',
        titlefont = dict(
        size = 20,
        color = 'rgb(131, 135, 135)'),
        tickfont=dict(
            size=16,
            color='rgb(131, 135, 135)'
        )),
    yaxis = dict(title = 'Height from MLLW (Meters)',
        titlefont = dict(
        size = 20,
        color = 'rgb(131, 135, 135)'),
        tickfont=dict(
            size=16,
            color='rgb(131, 135, 135)'
        )),
    paper_bgcolor='transparent',
    plot_bgcolor='transparent')

fig = go.Figure(data = [trace], layout=layout)
py.iplot(fig, filename='harmonicConstituent')


# In[ ]:



