import numpy as np
import pandas as pd
import datetime
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import Curve_Fitting_Network as CFN
import Curve_Fitting_Network_2 as CFN2

def rmse(estimated, actual):
    n = len(estimated)
    return (1/n)*np.sqrt(np.sum((estimated-actual)*(estimated-actual)))

def calculate_observable(data, par0, par1, par2):
    x, x_b, t, Q = data
    M_p = 0.938 #GeV

    #-1/(par[3]*par[3]*par[4]*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5]))*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5])))*(par[0]
    #+ par[1]*cos(x[0]) + par[2]*cos(x[0]*x[0]));

    estimate = -1/(x_b*x_b*t*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q))*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q)))*(par0 + par1*np.cos(x) + par2*np.cos(x*x))
    return estimate

def uniform_sample(val, error):
    return np.random.uniform(low=val-error, high=val+error)

def get_graph_arrays(line_value, x_axis, model):
        line1 = line_value
        #x_axis = np.linspace(0, 6, num=100)
        x_b1 = np.zeros((len(x_axis))) + x_b[line1*7]
        t_1 = np.zeros((len(x_axis))) + t[line1*7]
        Q_1 = np.zeros((len(x_axis))) + Q[line1*7]
        data1 = (x_axis, x_b1, t_1, Q_1)

        model_curve1 = []
        for i in range(len(x_axis)):
            params_tmp = model.feedforward(np.array([[x_b1[i]], [t_1[i]], [Q_1[i]], [x_axis[i]], [np.cos(x_axis[i])], [np.cos(x_axis[i]*x_axis[i])]]))

            data1_tmp = (x_axis[i],x_b1[i], t_1[i], Q_1[i])
            model_curve1.append(calculate_observable(data1_tmp, params_tmp[0][0], params_tmp[1][0], params_tmp[2][0]))
            if i==0:
                print(params_tmp[0][0],' ', params_tmp[1][0], ' ', params_tmp[2][0])

        return data1, model_curve1

def get_mean_and_std(values):
    # sum_tmp = 0.0
    # n = len(values)
    # for i in range(n):
    #     sum_tmp+=values[i]
    # mean = sum_tmp/n
    # std_sum_tmp = 0.0
    # for i in range(n):
    #     std_sum_tmp += (values[i]-mean)*(values[i]-mean)
    # std_dev = np.sqrt(std_sum_tmp/n)

    mean = np.mean(values)
    std_dev = np.std(values)
    return mean, std_dev

data = pd.read_csv('/Users/yeshwanthsomu/Documents/Fermilab/NeuralNetsPyC/Compton_FF_Code/data_ff.csv')
attributes =['X', 'X_b', 'Q', 't', 'F']

x_b = np.array(data['X_b'])
Q = np.array(data['Q'])
t = np.array(data['t'])
X = np.array(data['X'])

axis = np.arange(len(X))

F = np.array(data['F'])
errF = np.array(data['errF'])


tot_num = len(X)
train_num = 400
test_num = tot_num-train_num

#h = (X, x_b, t, Q)
y_data = F
X_data = []
for i in range(len(x_b)):
    X_data.append([x_b[i], t[i], Q[i]])
X_data = np.array(X_data)
#np.random.shuffle(X_data)




h = (X, x_b, t, Q)
y = F

# some initial parameter values, can be set to specific value or sign and constrained or set to 1.0 as default
initialParameters = np.array([1.0, 1.0, 1.0])
constraints = ((-np.inf, -np.inf, -np.inf), # Parameter Lower Bounds
               ( np.inf,  np.inf,  np.inf)) # Parameter upper bounds

y_w_error = []
for i in range(len(y)):
    y_w_error.append(uniform_sample(F[i], errF[i]))
y_w_error = np.array(y_w_error)

# Actually Very Real Parameters -------------------------------------------------- ##
actualParameters, pcov = curve_fit(calculate_observable, h, y, initialParameters)#, bounds=constraints)


# Parameters of Equation factoring in error of F
fittedParameters, pcov = curve_fit(calculate_observable, h, y_w_error, initialParameters)#, bounds=constraints)



# getting parameters for each curve
curve_fit_parameters = []
curve_fit_pcov = []

param0_list=[]
param1_list=[]
param2_list=[]
params = []

for i in range(0, len(X), 7):
    h=(X[i:(i+7)], x_b[i:(i+7)], t[i:(i+7)], Q[i:(i+7)])
    y = []
    for j in range(7):
        y.append(uniform_sample(F[i+j], errF[i+j]))
    y = np.array(y)
    t_p, p_cov = curve_fit(calculate_observable, h, y, initialParameters)#, bounds=constraints)
    curve_fit_parameters.append(t_p)
    curve_fit_pcov.append(p_cov)

    param0_list.append(t_p[0])
    param1_list.append(t_p[1])
    param2_list.append(t_p[2])
    params.append(t_p)

    # print(t_p)
params = np.array(params)
print(fittedParameters)


par0_m, par0_std = get_mean_and_std(param0_list)
print('Parameter 0 mean: {0} std dev: {1}'.format(par0_m, par0_std))
par1_m, par1_std = get_mean_and_std(param1_list)
print('Parameter 1 mean: {0} std dev: {1}'.format(par1_m, par1_std))
par2_m, par2_std = get_mean_and_std(param2_list)
print('Parameter 2 mean: {0} std dev: {1}'.format(par2_m, par2_std))

# plt.subplot(3,1,1)
# plt.hist(param0_list, label='Parameter 0')
# #fit_0 = stats.norm.pdf(param0_list, par0_m, par0_std)
# #plt.plot(param0_list, fit_0)
# plt.title('Parameters Curve_fit Histogram')
# plt.legend()
#
# plt.subplot(3,1,2)
# plt.hist(param1_list, label='Parameter 1')
# plt.legend()
# #
# plt.subplot(3,1,3)
# plt.hist(param2_list, label='Parameter 2')
# plt.legend()
# plt.show()

p0,p1,p2 = fittedParameters
output = calculate_observable(h, p0, p1, p2)



print()
print('Curve fit:')
print('Fitted ', fittedParameters)
print('Actual: ? ')
print('RMSE: ', rmse(y, output))


## Creating the Neural Net and Plotting the Lines

##Initialization

num_inputs = 6
num_outputs = 3
learning_rate = 0.025
regularization_rate = 0.12
F_error_scaling = np.array([[0.03], [0.03], [0.09]])
iterations = 10
batch_size = 4
layers = [num_inputs, 12, num_outputs]

X_data = [] # Input data that is being fed into ANN (x_b, t, Q, X, np.)
y_data = []
for i in range(len(x_b)):
    X_data.append([x_b[i], t[i], Q[i], X[i], np.cos(X[i]), np.cos(X[i]*X[i])])
    y_data.append([uniform_sample(F[i], errF[i])])
X_data = np.array(X_data)
y_data = np.array(y_data)


X_train = X_data[:train_num]
y_train = y_data[:train_num]

X_test = X_data[train_num:]
y_test = y_data[train_num:]

print(np.shape(y_train))
print(np.shape(X_train))

training_data = []
for i in range(len(X_train)):
    training_data.append((np.reshape(X_train[i],(num_inputs,1)), np.reshape(y_train[i],(1))))

test_eval_data = []
for i in range(len(X_test)):
    test_eval_data.append((np.reshape(X_test[i],(num_inputs,1)), np.reshape(y_test[i],(1))))


X_data = []
y_data = []
for i in range(len(x_b)):
    X_data.append([x_b[i], t[i], Q[i], X[i], np.cos(X[i]), np.cos(X[i]*X[i])])
    y_data.append([uniform_sample(F[i], errF[i])])
X_data = np.array(X_data)
y_data = np.array(y_data)

X_train = X_data[:train_num]
y_train = y_data[:train_num]

X_test = X_data[train_num:]
y_test = y_data[train_num:]
#np.random.shuffle(X_data)

print(np.shape(y_train))
print(np.shape(X_train))

training_data = []
for i in range(len(X_train)):
    training_data.append((np.reshape(X_train[i],(num_inputs,1)), np.reshape(y_train[i],(1))))

test_eval_data = []
for i in range(len(X_test)):
    test_eval_data.append((np.reshape(X_test[i],(num_inputs,1)), np.reshape(y_test[i],(1))))


print(type(test_eval_data))
print("")
print("")

print(datetime.datetime.now())
print("")
model_deep_network = CFN.CurveFittingNetwork(layers)
eval_cost, eval_acc, train_cost, train_acc = model_deep_network.SGD(training_data, iterations, batch_size,
                                                                    learning_rate,
                                                                    lmbda=regularization_rate,
                                                                    scaling_value=F_error_scaling,
                                                                    shrinking_learn_rate=True,
                                                                    evaluation_data=test_eval_data,
                                                                    monitor_training_accuracy=True,
                                                                    monitor_training_cost=True,
                                                                    monitor_evaluation_accuracy=True,
                                                                    monitor_evaluation_cost=True)
predicted_dnn =[]
actual_dnn = []

for (x,y) in test_eval_data:
    out = model_deep_network.feedforward(x)
    h_tmp = (x[3], x[0], x[1], x[2])
    predicted_dnn.append(calculate_observable(h, out[0], out[1], out[2]))
    actual_dnn.append(y)
predicted_dnn=np.array(predicted_dnn)
actual_dnn=np.array(actual_dnn)

print(datetime.datetime.now())





## Sectioning of data
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

secX = np.array(chunkIt(X, 7))
secX = np.transpose(secX)

