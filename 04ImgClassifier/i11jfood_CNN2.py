from  tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import utils
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import utils
import numpy as np

categories = ['Gyudon', 'Ramen','Sushi','Okonomiyaki','Karaage']
nb_classes = len(categories)
image_size = 100

# 데이터 로드 및 전처리 수행
def main():
    # 증감된 NumPy 데이터 파일 불러오기
    data = np.load("./saveFiles/japanese_food_aug.npz")
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_test = data['Y_test']

    X_train = X_train.astype('float') / 256
    X_test = X_test.astype('float') / 256

    Y_train = utils.to_categorical(Y_train, nb_classes)
    Y_test = utils.to_categorical(Y_test, nb_classes)

    model = model_train(X_train, Y_train)
    model_eval(model, X_test, Y_test)

def build_model(in_shape):
    model = Sequential()

    # 입력층 - 합성곱 층
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=in_shape))
    model.add(Activation('relu'))  # 활성화 함수 ReLU 적용(비선형성 추가)
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 2*2최대 풀링 적용
    model.add(Dropout(0.25))  # 과적합 방지를 위한 드롭아웃 적용(25% 뉴런 비활성화)

    # 은닉층1 - 합성곱 층
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    # 은닉층2 - 합성곱 층
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 은닉층3 - 완전 연결층
    model.add(Flatten())  # 2D 특징맵을 1D 벡터로 변환
    model.add(Dense(512)) # 완전연결층(512개의 뉴런)
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 출력층 : 클래스의 수만큼 뉴런 생성
    model.add(Dense(nb_classes))
    # 다중 클래스 분류를 위한 소프트맥스 활성화 함수 지정
    model.add(Activation('softmax'))

    # 모델 컴파일 : 손실 함수 , 옵티마이저, 평가 지표
    model.compile(loss='binary_crossentropy',  # 손실함수 지정
                  optimizer='rmsprop',  # RMSprop 옵티마이저 사용
                  metrics=['accuracy'])  # 평가 지표로 정확도 사용

    return model

def model_train(X, Y):
    model = build_model(X.shape[1:])

    model.fit(X,Y, batch_size=32, epochs=30)

    hdf5_file = "./saveFiles/japanese_food_aug_model.hdf5"
    model.save_weights(hdf5_file)
    return model

def model_eval(model, X, Y):
    score = model.evaluate(X, Y)
    print('loss =', score[0])
    print('accuracy =', score[1])

if __name__ == '__main__':
    main()