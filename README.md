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

  



















