import matplotlib.pyplot as plt
import numpy as np
import math


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logistic_sigmoid(x, k=5.0):
    if abs(x) >= 0.1:
        return 0
    return sigmoid(k * x) * (1.0 - sigmoid(k * x))


x = np.arange(-0.2, 0.2, 0.001)
y = []
for val in x:
    y.append(4 * logistic_sigmoid(val))
# print(x)
# plt.xlabel('SDF value')
# plt.ylabel('Opacity')
# plt.plot(x, y)

import plotly.express as px
import pandas as pd

# df = pd.DataFrame(
#     dict(
#         SDF_values=x,
#         Opacity=y,
#     )
# )
# fig = px.line(df, x="SDF_values", y="Opacity", title="Opacity-SDF Function")
# fig.update_layout(
#     font=dict(
#         # family="Times New Roman",
#         size=24,  # Set the font size here
#         # color="RebeccaPurple"
#     ),
#     # margin=dict(l=20, r=20, t=20, b=20),
#     # paper_bgcolor="LightSteelBlue",
#     xaxis=dict(titlefont=dict(size=36), title="<b>SDF</b>"),
#     yaxis=dict(titlefont=dict(size=36), title="<b>Opacity</b>"),
#     title="<b>Opacity-SDF Function</b>",
#     title_x=0.5,
#     # line=dict(width=10),
#     width=10,
# )
# fig.show()

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y, name="High 2014", line=dict(width=10)))
fig.update_layout(
    font=dict(
        # family="Times New Roman",
        size=24,  # Set the font size here
        # color="RebeccaPurple"
    ),
    # margin=dict(l=20, r=20, t=20, b=20),
    # paper_bgcolor="LightSteelBlue",
    xaxis=dict(titlefont=dict(size=36), title="<b>SDF</b>"),
    yaxis=dict(titlefont=dict(size=36), title="<b>Opacity</b>"),
    title="<b>Opacity-SDF Function</b>",
    title_x=0.5,
    # line=dict(width=10),
    # width=10,
)

fig.show()
