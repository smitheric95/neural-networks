import csv
import plotly.plotly as py
import plotly.graph_objs as go


def main():

    game = "MountainCar-v0"
    dirichlet1_data = []
    dirichlet1_avg_data = []
    x = []
    with open('Data/rewards_'+game+'_DQN_Guided_Exploration_dirichlet_1.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        d1sum = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                x.append(int(row[0]))
                d1sum += float(row[1])
                dirichlet1_data.append(float(row[1]))
                dirichlet1_avg_data.append(d1sum / line_count)
                line_count += 1

    dirichlet2_data = []
    dirichlet2_avg_data = []
    with open('Data/rewards_'+game+'_DQN_Guided_Exploration_dirichlet_2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        d2sum = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                d2sum += float(row[1])
                dirichlet2_data.append(float(row[1]))
                dirichlet2_avg_data.append(d2sum / line_count)
                line_count += 1

                
    dirichlet3_data = []
    dirichlet3_avg_data = []
    with open('Data/rewards_'+game+'_DQN_Guided_Exploration_dirichlet_3.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        d3sum = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                d3sum += float(row[1])
                dirichlet3_data.append(float(row[1]))
                dirichlet3_avg_data.append(d3sum / line_count)
                line_count += 1
                
    
    mvn_data = []
    mvn_avg_data = []
    with open('Data/rewards_'+game+'_DQN_Guided_Exploration_multivariate_normal.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        mvn_sum = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                mvn_sum += float(row[1])
                mvn_data.append(float(row[1])) 
                mvn_avg_data.append(mvn_sum / line_count)
                line_count += 1

    dqn_data = []
    dqn_avg_data = []
    with open('Data/rewards_'+game+'_DQN_dirichlet_1.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        dqn_sum = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                dqn_sum += float(row[1])
                dqn_data.append(float(row[1]))    
                dqn_avg_data.append(dqn_sum / line_count)
                line_count += 1

    font_size = 48
    trace1 = go.Scatter(
        x = x,
        y = dirichlet1_avg_data,
        name='Dirichlet 1',
        textfont=dict(
            size=font_size
        )
    )
    trace2 = go.Scatter(
        x = x,
        y = dirichlet2_avg_data,
        name = 'Dirichlet 2',
        textfont=dict(
            size=font_size
        )
    )

    trace3 = go.Scatter(
        x = x,
        y = dirichlet3_avg_data,
        name = 'Dirichlet 3',
        textfont=dict(
            size=font_size
        )
    )
    trace4 = go.Scatter(
        x = x,
        y = mvn_avg_data,
        name = 'Gaussian',
        textfont=dict(
            size=font_size
        )
    )
    trace5 = go.Scatter(
        x = x,
        y = dqn_avg_data,
        name = 'DQN',
        textfont=dict(
            size=font_size
        )
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        legend=dict(
            font=dict(
                size=24
            )
        ),
        title=game,
        titlefont=dict(size=32),
        xaxis = dict(title = 'Episode', titlefont=dict(size=24), tickfont=dict(size=24)),
        yaxis = dict(title = 'Avg Reward', titlefont=dict(size=24), tickfont=dict(size=24)),
    )
    
    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='Reinforcement Learning - '+game)        

    print("Hello World")

if __name__=="__main__":main()