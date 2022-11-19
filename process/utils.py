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