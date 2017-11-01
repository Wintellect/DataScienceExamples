import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.externals import joblib
import plotly.graph_objs as go

app = dash.Dash()

training_data = joblib.load("./training_data.pkl")
training_labels = joblib.load("./training_labels.pkl")

app.layout = html.Div(children=[
    html.H1(children='Simple Linear Regression', style={'textAlign': 'center'}),

    html.Div(children=[
        html.Label('Enter years of experience: '),
        dcc.Input(id='years-of-experience', placeholder='Years of experience', type='text'),
        html.Div(id='result')
    ], style={'textAlign': 'center'}),

    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=training_data['YearsExperience'],
                    y=training_labels,
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                )
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'Years of Experience'},
                yaxis={'title': 'Salary'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest'
            )
        }
    )
])


@app.callback(
    Output(component_id='result', component_property='children'),
    [Input(component_id='years-of-experience', component_property='value')])
def update_years_of_experience_input(years_of_experience):
    if years_of_experience is not None and years_of_experience is not '':
        try:
            salary = model.predict(float(years_of_experience))[0]
            return 'With {} years of experience you should earn a salary of ${:,.2f}'.\
                format(years_of_experience, salary, 2)
        except ValueError:
            return 'Unable to give years of experience'


if __name__ == '__main__':
    model = joblib.load("./linear_regression_model.pkl")
    app.run_server(debug=True)
