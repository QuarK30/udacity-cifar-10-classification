import numpy as np
import random

def sigmoid (z):
    return 1/(1+np.exp(-z))

def model(w, b, X):
    return sigmoid(np.dot(X, w)+b)

def loss_function(x_list,y_list,w,b):
    average_loss = 0.0
    for i in range(len(x_list)):
        y_pred = model(w,b,x_list[i])
        average_loss += np.multiply(-y_list[i],np.log(y_pred)) + np.multiply(-(1-y_list[i]),np.log(1-y_pred))
    average_loss /= len(x_list)
    return average_loss

def gradient(y_pred,y_real,x):
    diff = y_pred - y_real
    dw = diff * x
    db = diff

    return dw,db

def step_gradient(batch_x_list, batch_y_list,lr,w,b):
    aver_dw, aver_db = 0,0
    for i in range(len(batch_x_list)):
        pred_y = model(w,b,batch_x_list[i])
        dw,db = gradient(pred_y,batch_y_list[i],batch_x_list[i])
        aver_dw += dw
        aver_db += db
    aver_dw /= len(batch_x_list)
    aver_db /= len(batch_y_list)

    w -= lr * aver_dw
    b -= lr * aver_db

    return w,b

def train(x_list,real_y_list,batch_size,lr,max_iter):
    w,b =0,0
    loss_list = []
    for i in range(max_iter):
        batch_idex = np.random.choice(len(x_list),batch_size)
        batch_x_list = [x_list[i] for i in batch_idex]
        batch_real_y_list = [real_y_list[i] for i in batch_idex]
        w, b = step_gradient(batch_x_list,batch_real_y_list,lr,w,b)
        loss = loss_function(batch_x_list,batch_real_y_list,w,b)
        print('w:{0},b:{1}'.format(w,b))
        print('loss:{0}'.format(loss))
        loss_list.append(loss)
    return loss_list

def gan_sample_data():
    w = random.randint(0, 15) + random.random()
    b = random.randint(0, 10) + random.random()
    num_sample = 100
    x_list = []
    real_y_list = []
    for i in range(num_sample):
        x = random.randint(0, 100) * random.random()
        y = w * x + b
        y = 1 / (1 + np.exp(-y)) + random.random() * random.randint(-1, 1)
        x_list.append(x)
        real_y_list.append(y)
    return x_list, real_y_list, w, b

def plot_fig(loss_list):
    x_axis = range(0, 10000)
    fig = plt.figure()
    fig.set_title = ('loss')
    fig.set_xlabel = ('loss number')
    fig.set_ylabel = ('loss vlue')
    plt.plot(x_axis, loss_list)

    plt.show()

def run():
    x_list,real_y_list,w,b = gan_sample_data()
    loss_list = train(x_list,real_y_list,50,0.001,10000)
    plot_fig(loss_list)


if __name__ == '__main__':
    run()
    