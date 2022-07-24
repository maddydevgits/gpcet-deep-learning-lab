import numpy as np
from keras.layers import Dense, Dropout, Activation
from keras.layers import PReLU
from keras.models import Sequential

seed = 7
np.random.seed(seed)

num_digits = 14  # binary encode numbers
max_number = 2 ** num_digits


def prime_list():
    counter = 0
    primes = [2, 3]

    for n in range(5, max_number, 2):
        is_prime = True
        for i in range(1, len(primes)):
            counter += 1
            if primes[i] ** 2 > n:
                break
            counter += 1
            if n % primes[i] == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


primes = prime_list()
#print(primes)

def prime_encode(i):
    if i in primes:
        return 1
    else:
        return 0


def bin_encode(i):
    return [i >> d & 1 for d in range(num_digits)]


def create_dataset():
    x, y = [], []
    for i in range(102, max_number):
        x.append(bin_encode(i))
        y.append(prime_encode(i))
    return np.array(x), y


x_train, y_train = create_dataset()
print(x_train)
print(y_train)
x_train=np.array(x_train,dtype='int32')
y_train=np.array(y_train,dtype='int32')

model = Sequential()
model.add(Dense(units=100, input_dim=num_digits))
model.add(PReLU())
model.add(Dropout(rate=0.2))
model.add(Dense(units=50))
model.add(PReLU())
model.add(Dropout(rate=0.2))
model.add(Dense(units=25))
model.add(PReLU())
model.add(Dropout(rate=0.2))
model.add(Dense(units=1))
model.add(Activation("sigmoid"))

model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=128,
                    validation_split=0.1)

# predict
errors, correct = 0, 0
tp, fn, fp = 0, 0, 0

for i in range(2, 101):
    x = bin_encode(i)
    y = model.predict(np.array(x).reshape(-1, num_digits))
    if y[0][0] >= 0.5:
        pred = 1
    else:
        pred = 0
    obs = prime_encode(i)
    #print(i, obs, pred, y[0][0])
    if pred == obs:
        correct += 1
    else:
        errors += 1
    if obs == 1 and pred == 1:
        tp += 1
    if obs == 1 and pred == 0:
        fn += 1
    if obs == 0 and pred == 1:
        fp += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f_score = 2 * precision * recall / (precision + recall)

print("Errors :", errors, " Correct :", correct, "F score :", f_score)
