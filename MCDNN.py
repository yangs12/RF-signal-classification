#   The codes are for
#   S. Yang et al. "RF Signal-Based UAV Detection and Mode Classification: A
#   Joint Feature Engineering Generator and Multi-Channel Deep
#   Neural Network Approach".
import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Activation
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import keras.backend as K


# Parameters
np.random.seed(1)
learningrate = 0.001    # Start learning rate
number_category = 10

print("Load Data")
data = np.loadtxt("E:\\Data\\RF_Datasmooth.csv", delimiter=",")
x = np.transpose(data[0:2048,:])
print('x shape'+str(x.shape))
label = np.transpose(data[2050:2051,:])
y = to_categorical(label)   # One-hot encoding

total_acc = []
count = 0

# stratified split dataset
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
epoch_num = 300

# learning rate decay
def scheduler(epoch):
    reducelearningrate=1/2*(1+np.cos(epoch*np.pi/epoch_num))*learningrate
    K.set_value(model.optimizer.lr, reducelearningrate)
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

print(label)
for train, test in kfold.split(x, label):
    # Cross-validation number
    count = count + 1
    print(count)

    # prepare/split traning and test sets
    xtrain = x[train]
    xtest = x[test]
    ytrain = y[train]
    ytest = y[test]

    x1train = xtrain[:, 0:1024]
    x2train = xtrain[:, 1024:2048]
    x1test = xtest[:, 0:1024]
    x2test = xtest[:, 1024:2048]

    # build network structure
    model = Sequential()

    # Input1 - low frequency, Input2 - high frequency
    input_x = Input(shape=(1024,), name='input_x')
    x1 = Dense(units=128, activation='relu')(input_x)
    input_y = Input(shape=(1024,), name='input_y')
    x2 = Dense(units=128, activation='relu')(input_y)

    m = keras.layers.concatenate([x1, x2])      # Concatenate two inputs
    m = Dense(units=128, activation='relu')(m)
    m = Dense(units=128, activation='relu')(m)
    m = Dense(units=128, activation='relu')(m)
    output = Dense(units=y.shape[1], activation='softmax', name='output')(m)

    model = Model(inputs=[input_x,input_y], outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # model.summary()   # Print network structure

    hist=model.fit(x=[x1train,x2train], y=y[train], epochs=epoch_num, batch_size = 32, verbose = 0,callbacks=[reduce_lr])
    acc = model.evaluate(x=[x1test,x2test], y=y[test], verbose = 1)
    print('Accuracy'+str(acc[1]*100))
    total_acc.append(acc[1]*100)
    y_pred = model.predict([x1test,x2test])

    # Save output data
    np.savetxt("Outputs%s.csv" % count, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')