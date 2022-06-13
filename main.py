import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

graph_dots = 100

flow_pos = []
flow_neg = []
person_per_tick = []
with open("./results.txt", "r") as results_file:
    for line in results_file:
        person_per_tick_, flow_pos_, flow_neg_ = line.replace("\n", "").split(" ")
        person_per_tick.append(float(person_per_tick_))
        flow_pos.append(float(flow_pos_))
        flow_neg.append(float(flow_neg_))


##regression analyse
x = np.array(flow_pos).reshape((-1, 1))
y = np.array(flow_neg)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)


min_flow_pos = min(flow_pos)
max_flow_pos = max(flow_pos)
step = (max_flow_pos - min_flow_pos) / graph_dots
x_graph_array = np.arange(min_flow_pos, max_flow_pos, step)
x_input = x_graph_array.reshape((-1, 1))
y_graph_array = model.predict(x_input)

#show_graph
plt.plot(flow_pos, flow_neg, 'ro')
plt.plot(x_graph_array, y_graph_array)
plt.show()
