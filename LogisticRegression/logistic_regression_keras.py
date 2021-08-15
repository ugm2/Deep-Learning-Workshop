from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

class LogisticRegressionKeras(Sequential):
    def __init__(self, input_dim):
        super(LogisticRegressionKeras, self).__init__()
        self.add(Dense(1, input_dim=input_dim, activation='sigmoid'))
        opt = SGD(learning_rate=0.01)
        self.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])