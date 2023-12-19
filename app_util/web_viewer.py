import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np

# Create a Dash web application
app = dash.Dash(__name__)

# Generate initial data
np.random.seed(42)
num_points = 100
initial_data = pd.DataFrame({
    'X': np.random.rand(num_points),
    'Y': np.random.rand(num_points),
    'Z': np.random.rand(num_points),
})

# Initial camera position
initial_camera = dict(eye=dict(x=1.25, y=1.25, z=1.25))

# Layout of the application
app.layout = html.Div([
    dcc.Store(id='camera-store', data=initial_camera),
    dcc.Graph(id='scatter-plot', style={'height': '80vh'}),
    dcc.Interval(
        id='update-interval',
        interval=1000,  # update every 1 second
        n_intervals=0
    )
])

# Callback to update the scatter plot and camera perspective in real-time
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('camera-store', 'data')],
    [Input('update-interval', 'n_intervals')],
    [State('camera-store', 'data')]
)
def update_plot_and_camera(n_intervals, stored_camera_data):
    # Simulate real-time data update
    new_data = pd.DataFrame({
        'X': np.random.rand(num_points),
        'Y': np.random.rand(num_points),
        'Z': np.random.rand(num_points),
    })

    # Get the current camera perspective
    camera = stored_camera_data['eye']

    # Update the camera perspective dynamically
    new_camera = dict(eye=dict(x=camera['x'] + np.random.uniform(-0.1, 0.1), 
                               y=camera['y'] + np.random.uniform(-0.1, 0.1), 
                               z=camera['z'] + np.random.uniform(-0.1, 0.1)))

    # Update the camera data
    stored_camera_data['eye'] = new_camera['eye']

    # Update the figure with a 3D scatter plot
    fig = px.scatter_3d(
        new_data,
        x='X',
        y='Y',
        z='Z',
        title='Real-time 3D Scatter Plot Animation',
        labels={'X': 'X-axis', 'Y': 'Y-axis', 'Z': 'Z-axis'},
        size_max=3  # Adjust the maximum marker size
    )

    # Set the camera perspective
    fig.update_layout(scene_camera=dict(eye=new_camera['eye']))

    return fig, stored_camera_data

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)




'''
from time import sleep
from dash import Dash, html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def hello_world(app):

    app.layout = html.Div(
        [html.Div(children="Hello World!")]
    )


def main():

    #app = Dash(__name__)
    #hello_world(app)
    #app.run(debug=True)

    df = px.data.iris()

    arr = df.to_numpy()
    print(arr)

    df = pd.DataFrame(arr, columns=[
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species', 'species_id'
        ]
    )
    print(df)

    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')
    fig.show()


if __name__ == "__main__":
    main()
'''