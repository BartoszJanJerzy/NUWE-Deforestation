import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List


def load_history_graph(
    acc: List[float],
    val_acc: List[float],
    loss: List[float],
    val_loss: List[float],
    title='History',
    width: int=1300,
    height: int=600
) -> go.Figure:
    x = [i+1 for i in range(len(acc))]
    
    fig = make_subplots(
        cols=2, rows=1,
        subplot_titles=('Accuracy', 'Loss')
    )
    
    for y, name, col, color in zip(
        [acc, val_acc, loss, val_loss],
        ['Accuracy', 'Val Accuracy', 'Loss', 'Val Loss'],
        [1, 1, 2, 2],
        ['blue', 'orange', 'blue', 'orange']
    ):
        fig.add_trace(
            trace=go.Scatter(
                name=name,
                x=x,
                y=y,
                mode='lines+markers',
                marker_color=color
            ),
            col=col, row=1
        )
    
    fig.update_yaxes(range=(0, max(acc)+0.1), col=1)
    fig.update_yaxes(range=(0, max(val_loss)+0.5), col=2)
    
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_title='epoch'
    )
    
    return fig


def load_models_history_graph(
    acc_list: list,
    labels: list,
    name: str,
    marker_color: str,
    acc_list_2: list,
    name_2: str,
    marker_color_2: str,
    title: str = 'Models training history',
    width: int = 700,
    height: int = 500
) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=name,
        orientation='h',
        y=labels,
        x=acc_list,
        text=acc_list,
        marker_color=marker_color
    ))
    fig.add_trace(go.Bar(
        name=name_2,
        orientation='h',
        y=labels,
        x=acc_list_2,
        text=acc_list_2,
        marker_color=marker_color_2
    ))
    
    fig.update_xaxes(title='Accuracy', range=(0, 1.1))
    
    fig.update_layout(
        title=title,
        width=width,
        height=height
    )
    
    return fig