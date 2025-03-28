# AI-Library  
필요한 사전 지식  
  
인공신경망  
CNN  
ResNet  
RNN  
U-Net  
Auto Encoder  
LSTM  
Attention Mechanism  
CRNN  
GAN  
  
## 2주차 인공신경망(ANN - artifitial neural network)
더 좋은 결과물을 위해서...  
1. 적절한 최적화 함수 선택하기
2. 학습률 변경하기
3. epoch 변경하기

batch (데이터 묶음 단위)
iteration(하나의 epoch를 달성하기 위해서 반복하는 횟수)
epoch(학습의 반복 횟수)

1000 개의 data를 batch_size 100으로 학습하면 iteration이 10번이 되고 10번을 하면 epoch 한번이다.   

### 사인함수 예측하기  
y = torch.sin(x)를 통해서 삼각함수를 그리고 계수를 구하여 함수 예측하기  
  
1.모델 정의 -> 2.모델 순전파 -> 3.오차 계산 -> 4.오차 역전파(가중치 업데이트) -> 2~4 반복(epoch 학습횟수) -> 학습 종료  
  
오차 = (예상y - y) 제곱의 합 

1. import 
2. 사인 함수 그리기 (-PI부터 +PI 까지 1000 개 구간으로 나누어서 x에 넣은 값을 사인 함수로)
3. 계수가 난수로 된 변수(a, b, c, d)를 가진 3차 함수 그리기  
4. matplotlib을 이용한 표 시각화  
5. 학습률 설정 (learning_rate 낮은 수인 1e-6 = 1 X 10<sup>-6</sup>)  
6. epoch 수 만큼 학습 실행
7.  손실 정의 loss = (y_pred - y).pow(2).sum().item()  
평균제곱오차를 구하는 방식과 달리 평균을 구하지 않고 손실 정의를 함, .item()은 pyTorch 텐서에서 단일값을 python의 데이터 타입으로 변환하는 함수  
8. 기울기 대한 미분(순간기울기)를 통해 
9. 기울기를 학습률*미분계수만큼 업데이트
10. 사인함수와 예상함수, 난수 함수를 비교하는 표 작성  

### 보스턴 집값 예측하기 (회귀)

1.모델 정의 -> 2.데이터 불러오기 -> 3.손실 계산 -> 4.오차 역전파(가중치 업데이트) 및 최적화 -> 2~4 반복(epoch 학습횟수) -> 학습 종료  

  
### MNIST 손글씨 분류하기 
28 X 28 pixel 784 차원으로 만든 후 사용  
입력층    은닉    출력  
784       64      10  

1.모델 정의 -> 2.데이터 불러오기(train data와 test data) -> 3.손실 계산 -> 4.오차 역전파(가중치 업데이트) 및 최적화 -> 2~4 반복(epoch 학습횟수) -> 학습 종료  

## 3주차 CNN(Convolutional Neural Network) - VGG16
  
CNN  
합성곱을 사용하여 이미지의 특징을 추출하는 신경망  
ann의 가중치 업데이트와 달리 필터를 자동으로 구성(어떻게 특징을 뽑을 지)하는 신경망  

32X32 이미지 -> 16X16이미지 출력(특징추출-feature map) 기본블록(코드 재사용을 위해) -> 8X8 이미지 출력 -> 4X4 이미지 출력(핵심특징) -> 평탄화(flatten)  -> MLP -> 예측 확률  
                (convolution... Max Pooling)  
MLP (Multi Layer Perceptron)  
이미지 크기가 커짐에 따라 계산량이 커짐 → 비효율적!  
엄청난 수의 가중치(파라미터) 필요!  
픽셀 단위로 학습해서 공간적 정보(위치, 패턴) 손실  
VS  
CNN  
큰 이미지도 작은 필터로 처리가능하여 작은 파라미터로 학습 가능함  
이미지 속에서 사물의 위치나 크기가 달라져도 분류할 수 있음  
  
### CNN 구성요소 
합성곱
작은 필터를 이용해 이미지로부터 특징을 뽑아내는 알고리즘
커널과 필터(Kernel & Filter)
RGB 각각 R채널, G채널, B채널 마다 특징을 추출하는 커널과 각각의 색을 추출하는 커널을 모으면 필터   
Stride 커널의 위치 이동 거리  
특징맵(2X2), 3X3 커널, 5X5 이미지 위에서 이동할 때| stride가 2만큼 이동하기 때문에 2X2 특징 맵이 생성 됨. 1이었으면 총 9번 움직이는 3X3이 되었을 것  
  
특징맵(Feature Map) 커널과 연산을 통해 나타나는 특징  

padding 점점 이미지가 학습할 수록 줄어드는 걸 방지하기 위해서  
ex) 3X3 이미지에 3X3 커널을 학습하면 1X1 짜리 특징맵만 생성됨.  
convolution 후에도 입력 크기를 유지하기 위해 연산 전에 이미지 주변을 다른 값으로 채움  
zero padding( 0으로 밖에 두르기 - 연산량 감소(학습 시간 작다-자연스럽지 않다.)  
reflection padding(복사, 가까운 걸 복사해서 사용함)
replication padding(안쪽에 있는 요소 복사, 2번째 있는 가로,세로를 사용|좀 더 자연스러움)  


Max pooling 가장 중요한 값만 가져오기 위해서 사용  
convolution 연산을 통해 나온 특징맵을 다운샘플링하여 크기를 줄이고, 중요한 정보(최댓값)만 남김  
장점: 특징맵 크기 감소에 따른 연산량 감소, 중요 정보 손실 최소화, 과적합(overfitting) 방지 가능, 객체가 이미지 상의 위치에 상관없이 잘 인식할 수 있음.  

### CNN의 구조(VGG16)

기본적으로 3*3 필터를 가장 많이 사용하고, 필요하면 5*5를 사용하기도 함(중앙 파악을 위해서 홀수 지정)  

요즘: CNN을 대체할 수 있는 ViT(vision transformer) 등장
### 실습 CIFAR - 10 데이터셋 
데이터셋 이미지 증감 -> 이미지 정규화(RGB 평균 및 표준편차로 정규화 진행) ->기본 블록 정의(Conv2d → ReLU → Conv2d → ReLU → MaxPool) ->  CNN으로 이미지 분류(입력 이미지 → 기본블록 1 → 기본블록 2 → 기본블록 3 → Flatten → FC1 (Fully Connected Layer) → ReLU → FC2 → ReLU → FC3 → 최종 클래스 예측) -> 정확도 평가  

### 실습: 전이학습모델 VGG로 분류하기

전이 학습(transfer learning)   
전이 학습  
다른 데이터를 이용해 이미 학습된 모델을 가져와 현재 나의 데이터에 최적화시키는 학습 방법.  
드롭아웃(Dropout)  
 과적합을 피하기 위해 일부 뉴런을 랜덤하게 비활성화하는 기법(epoch마다 랜덤 비활성화)  

## 4주차 ResNet (Residual Network)
Residual 잔여, 잔차  
### ResNet(Residual Network): Skip connection을 이용한 CNN 신경망  
CNN 중 가장 많이 사용하는 모델  
기존의 학습(VGG 모델)의 경우 층이 더해지면 어느 순간 기울기가 0에 가까워지는 기울기 소실 문제  
이를 해결하기위해서 skip connection 도입해 Residual Learning을 함 = ResNet  
예시로 VGG-16은 16층  
ResNet-18, ResNet-34, ... , ResNet-1202 등이 있음  
도표를 보면 어느 순간부턴 더 낮아지지 않는 걸 확인 가능   
Sigmoid 함수가 대부분 0에 가깝고 높은 값도 0.25정도고 층을 더 할 수록 줄어듦.  
skip connection: 블록을 통과하는 f(x)와 통과하지 않은 X를 더해줌  
그런 블록을 Residual Block이라고 부름  
- f(x) + X  
224 X 224 이미지
초반의 연산량을 줄이고 전체적인 파악을 위해 큰 Conv 7X7 사용 -> 112 X 112
특징맵 추출
이미지는 점점 작아지고  
필터 수는 점점 늘어남  

처음 image 3채널 (R, G, B)  
나올 땐 64 채널  
F(x)와 x를 더해주려면 채널의 수가 같아야 함. 이미지의 특징을 손상시키지 않으면서 채널의 수를 맞춰주는 작업 *다운샘플*을 진행시킴  
 
### Average Pooling  
2  2  7  3   
9  4  6  1  
8  5  2  4  
3  1  2  6    
Filter (2X2)  
Stride (2, 2)  
Max pool
9  7  
8  6  
Average Pooling 
4.25  4.25  
4.25  3.5  
  
### Batch Normalization(배치 정규화)
배치 단위를 정규화하는 기법이며, 각 배치의 출력의 분포를 일정하게 해줌.  
학습이 되는 부분으로 정규화 결과에 Scale(감마), Shift(베타)를 학습해 정규화 결과에 적용함  






























