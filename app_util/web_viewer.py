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