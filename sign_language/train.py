from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from utils import *
from datetime import datetime
from apply_augmentation import augmentation

TRAIN_SPLIT = 0.70
TEST_SPLIT = 1 - TRAIN_SPLIT
VAL_SPLIT = 0.25

actions = get_actions()
sequences, labels = augmentation(amount=0)

added_sequences, added_labels = load_numpy(DATA_PATH)
sequences += added_sequences
labels += added_labels

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode="min", min_delta=0.001)
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../data/', histogram_freq=0, write_graph=True, write_images=True)

model = Sequential()
model.add(LSTM(256, return_sequences=True, activation='selu', kernel_initializer='lecun_normal', input_shape=(20, 244)))
model.add(Dropout(0.4))
model.add(LSTM(128, return_sequences=False, activation='selu', kernel_initializer='lecun_normal'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='selu', kernel_initializer='lecun_normal'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='selu', kernel_initializer='lecun_normal'))
model.add(Dense(actions.shape[0], activation='softmax'))

opt = SGD(lr=0.0001)

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(X_train, y_train, epochs=800, callbacks=[early_stopping],
          verbose=1, validation_split=VAL_SPLIT)

y_pred = model.predict(X_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

acc = (cm.diagonal().sum() / cm.sum()) * 100
precision = np.diag(cm) / np.sum(cm, axis=0)
recall = np.diag(cm) / np.sum(cm, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy:  {acc}%")
print(f"Precision: {precision}")
print(f"Recall:    {recall}")
print(f"F1 score:  {f1_score}")

date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
model.save('../data/model.h5')
model.save(f'../data/{date}.h5')

