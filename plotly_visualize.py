from plotly.graph_objs import *
from plotly.offline import plot as offpy
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.graph_objects as go


def reformat_graph_layout(G, layout):
    '''
    this method provide positions based on layout algorithm
    :param G:
    :param layout:
    :return:
    '''
    if layout == "graphviz":
        positions = graphviz_layout(G)
    elif layout == "spring":
        positions = nx.fruchterman_reingold_layout(G, k=0.5, iterations=1000)
    elif layout == "spectral":
        positions = nx.spectral_layout(G, scale=0.1)
    elif layout=="random":
        positions = nx.random_layout(G)
    else:
        raise Exception("please specify the layout from graphviz, spring, spectral or random")

    return positions


def visualize_graph(G, edge_curvatures=None, node_labels=None, node_size=None, layout="graphviz", pos=None, filename="networkx.html", title=""):
    if pos:
        positions = pos
    else:
        # Generate positions for the nodes
        positions = reformat_graph_layout(G, layout)  # If you have a custom layout function

    # Check if edge_curvatures is provided and not empty
    use_edge_curvatures = bool(edge_curvatures)

    if use_edge_curvatures:
        # Normalize the curvature values for color mapping
        curvatures = list(edge_curvatures.values())
        min_c = min(curvatures)
        max_c = max(curvatures)
        norm = mcolors.Normalize(vmin=min_c, vmax=max_c)
        cmap = cm.get_cmap('coolwarm')
    else:
        # Set a default edge color
        default_edge_color = 'gray'

    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]

        if use_edge_curvatures and edge in edge_curvatures:
            curvature = edge_curvatures[edge]
            # Map curvature to color
            color = mcolors.rgb2hex(cmap(norm(curvature)))
            hover_text = f'Curvature: {curvature:.2f}'
        else:
            color = default_edge_color  # Default edge color
            hover_text = ''  # No hover info

        # Create an edge trace
        edge_trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=2, color=color),
            hoverinfo='text',
            text=hover_text,
            mode='lines'
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes_list = []

    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node_labels[node] if node_labels and node in node_labels else str(node))
        node_colors.append(len(list(G.neighbors(node))))  # Color based on degree
        node_sizes_list.append(node_size if node_size else 10)  # Default size is 10

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_sizes_list,
            color=node_colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                thickness=15,
                title='Node Degree',
                xanchor='left',
                titleside='right'
            )
        )
    )

    # Create a colorbar for edge curvatures if available
    if use_edge_curvatures:
        colorbar_trace = go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                colorscale='Jet',
                showscale=True,
                cmin=min_c,
                cmax=max_c,
                colorbar=dict(
                    thickness=15,
                    title='Edge Curvature',
                    xanchor='right',
                    titleside='right'
                )
            ),
            hoverinfo='none'
        )
    else:
        colorbar_trace = None

    # Assemble the figure
    data = edge_traces + [node_trace]
    if colorbar_trace:
        data.append(colorbar_trace)

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title=title,
            titlefont=dict(size=16),
            showlegend=False,
            width=800,
            height=600,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    # Save and display the figure
    fig.write_html(filename, auto_open=True)






def visualize_graph_3d(G, node_labels, node_sizes, filename, title="3d"):
    edge_trace = Scatter3d(x=[],
                       y=[],
                       z=[],
                       mode='lines',
                       line=Line(color='rgba(136, 136, 136, .8)', width=1),
                       hoverinfo='none'
                       )


    node_trace = Scatter3d(x=[],
                       y=[],
                       z=[],
                       mode='markers',
                       #name='actors',
                       marker=Marker(symbol='dot',
                                     size=[],
                                     color=[],
                                     colorscale='Jet',#'Viridis',
                                     colorbar=dict(
                                         thickness=15,
                                         title='Node Connections',
                                         xanchor='left',
                                         titleside='right'
                                     ),
                                     line=Line(color='rgb(50,50,50)', width=0.5)
                                     ),
                       text=[],
                       hoverinfo='text'
                       )

    positions = nx.fruchterman_reingold_layout(G, dim=3, k=0.5, iterations=1000)



    for edge in G.edges():
        x0, y0, z0 = positions[edge[0]]
        x1, y1, z1 = positions[edge[1]]
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]
        edge_trace['z'] += [z0, z1, None]


    for node in G.nodes():
        x, y, z = positions[node]
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        node_trace['z'].append(z)


    for adjacencies in G.adjacency_list():
        node_trace['marker']['color'].append(len(adjacencies))

    for size in node_sizes:
        node_trace['marker']['size'].append(size)


    for node in node_labels:
        node_trace['text'].append(node)

    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    layout = Layout(
        title=title,
        width=1000,
        height=1000,
        showlegend=False,
        scene=Scene(
            xaxis=XAxis(axis),
            yaxis=YAxis(axis),
            zaxis=ZAxis(axis),
        ),
        margin=Margin(
            t=100
        ),
        hovermode='closest',
        annotations=Annotations([
            Annotation(
                showarrow=False,
                text="",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=Font(
                    size=14
                )
            )
        ]), )

    data = Data([node_trace, edge_trace])
    fig = Figure(data=data, layout=layout)

    offpy(fig, filename=filename, auto_open=True, show_link=False)



