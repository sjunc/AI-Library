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

f(x) + X  
  
224 X 224 이미지
초반의 연산량을 줄이고 전체적인 파악을 위해 큰 Conv 7X7 사용 -> 112 X 112
특징맵 추출
이미지는 점점 작아지고 필터 수는 점점 늘어남  
  
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

## 5주차 RNN(Recurrent Neural Network)
: 시퀀스 데이터(순서가 있는 데이터)를 처리하기 위한 반복 구조 신경망 recurrent 반복되는  
시계열 데이터  
RNN = 순서를 '기억'하는 가장 기본적인 구조 (문장, 음성, 주가, 센서값(온도 등), 음악 등  
  
• output_t = f(input_t + hidden_{t-1})  
 t: 지금, t-1 예전 시점  
핵심**: 같은 가중치 를 모든 시점에서 공유함  
하나의 문장을 학습할 때 모두 같은 가중치를 기지고 학습하고 loss를 계산하고 가중치를 수정한다.  
  
나는, 너를, 좋아해  
나는 h₁ = f(W · x₁ + U · h₀) h₀:임의로 초기화할 것  
너를 h₂ = f(W · x₂ + U · h₁)  
좋아해 h₃ = f(W · x₃ + U · h₂)  
h: 각 입력을 통해 나오는 hidden state  

### 활용 분야  
• 자연어 처리: 문장 분류, 감정 분석, 번역  
• 음성 인식, 음악 생성  
• 시계열 예측(ex. 주가, 날씨 등 예측)    
### 한계  
• 기울기 소실 문제: 역전파시 기울기가 0에 가까워져 학습이 되지 않는 문제  
• 먼 과거의 정보를 기억하기 어려운 문제: 문장의 앞부분 내용을 뒤에서 활용하기 어려움(*장기 의존성 문제)    
### 대표 RNN  
• Vanilla RNN: 가장 단순한 RNN    
• LSTM(Long Short-Term Memory): 장기 기억을 할 수 있도록 보완함  
• GRU(Gated Recurrent Unit): LSTM보다 계산량이 적고 빠르게 학습됨  

입력 -> 은닉 -> 출력
     h₀  
x₁ → h₁ →  
↓  
x₂ → h₂ →  
↓  
x₃ → h₃ →  
  
시간 t = 1  
  
x₁ * [ W ]  
\  
[+] -> f -> h₁  
/  
h₀ * [ U ]   
  
W,U: 가중치  

f: 활성화 함수(ReLu, tanh)  

p11 벡터 단위로 표현한 RNN  
p12 뉴런 단위로 표현한 RNN  

하나의 문장 처리 중 W, U 등 가중치는 고정  

### RNN의 입출력

일 대 다 (ex]이미지 캡셔닝 이미지 문장 설명 출력 in: 이미지 1장 out: 이미지에 대한 설명 문장)
(one-to-many)  
다 대 일 (ex] 감정 분류, 스팸 메일 분류)  
(many-to-many)  
다 대 다 (ex] 품사 태깅, 언어 번역, 챗봇)  
(many-to-many)  

### RNN의 장기 의존성 문제 (->LSTM 으로 보완)
기울기 소실(Gradient Vanishing)  
: 딥러닝에서 역전파를 할 때, 오차를 전달하는 그래디언트가 반복적으로 곱해지며 점점 작아지는 현상  
장기 의존성 문제(Long-Term Dependency)   
: 긴 시퀀스에서 초반의 입력이 출력에 영향을 주지 못하게 됨  
기울기 손실(원인)  
장기 의존성 문제(결과)  
1. 시퀀스 데이터가 길어짐(문장이 길 때)  
↓  
2. 기울기 소실 발생  
↓  
3. 초반 입력이 학습되지 않음  
↓  
4. 장기 의존성 문제 발생 (과거 정보 상실)  

### 넷플릭스 주가 예측 
기존 모델과 비슷하고 초기 은닉 상태를 정의하는 부분만 다름 (h₀)  
(회귀 문제) 출력 모양을 맞춰주기 위해 마지막에 MLP(MultiLayer Perceptron) 사용  
1번째 날 ~ 30번째 날 학습 -> 31번째 날 예측  
2번째 날 ~ 31번째 날 학습 -> 32번째 날 예측  
3번째 날 ~ 32번째 날 학습 -> 33번째 날 예측  
...........................     
N-31번째 날 ~ N-1번째 날 학습 -> 예측  

## 6주차 LSTM
#### RNN의 한계
기울기 소실(원인)  
장기 의존성 문제 발생(결과)  
#### Long Short Term Memory 
장기 기억(cell state)와 단기 기억(hidden state) 담당이 존재  
Gate를 통해 연산을 진행  
기존 RNN의 문제를 해결  
기존 RNN의 hidden state(은닉 상태)에 cell state(셀 상태)를 추가  
• 필요한 정보는 기억하고 Input gate + candidate gate  
• 불필요한 정보는 약화시키고 Forget gate  
• 새로운 정보는 선택적으로 저장할 수 있음 Input gate + candidate gate   
내보낼 때 Output gate 총 4가지  

자연어 처리, 시계열 예측(주가), 음성 인식(STT, Speech-To-Text) 등에서 사용됨  
Long-Term 장기 기억 유지 과거의 데이터  
Short-Term 단기 정보 처리 새로 들어오는 데이터   
Memory 기억  
  
관련 개념들  
• Hidden State(hₜ, 또는 hsₜ)  
• Cell State(cₜ 또는 csₜ)  
• Forget Gate  
• Input Gate  
• Output Gate  
• Candidate Gate(ĉₜ)  
• Concatenate 연산 [hₜ₋₁, xₜ]  
• Element-wise multiplication  
• sigmoid와 tanh 활성화 함수  
  
### 연산
Concatenate 연산  
연결해서 붙인다.  
[1, 2, 3] [4, 5, 6] = [1, 2, 3, 4, 5, 6]
O 안에 ' 기호  
Element-wisemultiplication  
~를 기준으로(방향으로) 연산한다.  
O안에X 기호  

### 함수
Sigmoid  
0~1 사이의 값  
결과가 무조건 양수 값임.  
Tanh(탄젠트 H 혹은 하이퍼볼릭탄젠트)  
-1 ~ 1 사이의 값  
결과는 입력에 따라서 음수 -> 음수, 0 -> 0, 양수 -> 양수  
  
### State  
장기기억회로   Output gate와 현재 시점의 Cell State를 통해 Hidden State가 업데이트됨  
  
_>____________________>  cell state  
|                  |  
|__________________|__>  hidden state  
단기기억 회로    |    
               출력   
Forget gate(어떤 걸 잊을 지 결정) 현재 시점의 입력과 Input gate, Candidate gate를 통해 Cell State가 업데이트됨  

### gate
Forget Gate   
이전 cell state에서 어떤 정보를 잊을지 결정하는 gate   
장기기억 중에서, 중요하지 않은 과거의 정보를 제거함   
Candidate Gate(+Input)  
새로 들어올 정보의 값(내용)을 생성함  
(비유) 새롭게 들어온 입력으로 초안을 작성함  
Input Gate  
현재 시점의 입력에서 어떤 정보를 새로 기억할지 결정하는 gate  
(비유) candidate gate가 뽑아둔 초안 중 쓸만한 걸 선택함  
Output Gate  
최종적으로 현재 시점의 출력(hidden state)을 결정함  
여기서 만들어진 출력은 다음 시점으로 전달되면서 결과 예측에 사용될 것임.  
진행과정  
(hₜ₋₁ [예전의 hidden state], xₜ [지금의 입력])  
↓  
[Forget Gate] : 이전 기억 (cₜ₋₁ 예전의 cell state) 중에서 무엇을 얼마나 잊을지 결정  
[Input Gate + Candidate Gate] : 현재 들어온 입력 중에서 무엇을(candidate gate) 얼마나(input gate) 기억할지 결정  
↓  
Cell State 업데이트 (cₜ가 생성됨) 그 다음 LSTM,시점의 입력으로 가게 됨  
↓  
[Output Gate] : 무엇을 얼마나 출력할지 결정  
↓  
출력 (hₜ [t+1 시점의 입력]) 생성  
작동원리 그림  
Xt = xt(현재 들어온 입력) 와 Ht-1 concatenate 연산(단순 합치기) 백터임.   
현재 가중치와 
Xfₜ = σ(W_f · Xₜ + b_f) ← forget gate
  결과값은 0~1사이  
  0에 가깝게 나오면 기억을 잊고  
  1에 가까우면 기억을 보존함  
  
맨위 + 기호 있는 곳  아래
지금 온 과정: 잊을 것을 잊고 새로 기억해야할 것을 선정해서 Ct-1을 Ct로 업데이트 함.  
ĉₜ = tanh(W_c · Xₜ + b_c): candidate gate  
부호 유지하면서 새롭게 들어올 정보의 초안을 생성함  
iₜ = σ(W_i · Xₜ + b_i): input gate   
초안 중에서 결정함 Element-wisemultiplication   
oₜ = σ(W_o · Xₜ + b_o) ← output gate  
σ 0과 1사이  
아래 softmax를 거치는 과정은 ( 결과를 낼 때 사용)  
이 그래프에서 편차는 생략되어 있는 것이고 가중치는 학습하면서 전부 게이트 별로 따로 업데이트를 함  

## 7주차 GRU, CRNN
  
복습: LSTM 장기 의존성 문제 해결  
Cell state (장기) + hidden state(단기)  
Forget, Input+ candidate, Output gate 존재   
시그모이드와 tanh 사용의 차이  
### GRU
LSTM의 구조가 복습하여 학습 시간이 긺. 성능은 유지하면서 더 간단하고 학습이 빠른 구조  
GRU (Gated Reccurented Unit)  
두 개의 게이트  
Update gate 새 정보의 비율 조절  
Reset gate 적게 쓸지 조절  
하나의 State  
GRU는 Hidden state만 사용함   
•Update Gate(z_t)  
• Reset Gate(r_t)  
• Hidden State(h_t)  
• Candidate State(g_t)  
z_t: Update Gate  
r_t: Reset Gate  
σ: sigmoid 활성화 함수 -> (0~1) 부호 정보 없애고 0~1로 정규화  
tanh: hyperbolic tangent 활성화 함수 -> (-1 ~ 1)  

### GRU의 구조
z_t(Update Gate) (reset gate와 설명이 같음)  
: 무엇을 기억에서 지우고, 무엇을 유지할지 결정함  
: 이전 상태 h_(t-1)과 현재의 입력 x_t를 이용해 계산함  
zt = WxzXt + WhzHt-1 +bx
W의 첫 번째 첨자: 입력 소스  
W의 두 번재 첨자: 목적지 게이트  
  
g_t(Candidate State, )  
새로운 hidden state의 후보을 담고있음.    
gt = tanh(WxgXt + Whg(rt 0x ht-1) +bg)  
     부호유지     resetgate 결과물과 이전 hidden state의 값 연산  
       
h_t(최종 hidden state)  
: update gate(z_t)를 사용하여 예전의 기억과 새로운 정보를 혼합함  
gt(새로운 값)을 얼마나 남길지 정하고 예전정보를 얼마나 남길지 정해서 더한 값  

### CRNN
CNN (Convolutional Neural Network) + RNN(Recurrent Neural Network)   
이미지 특징 추출에 탁월                시간 순서(sequence data)가 있는 데이터 처리에 유용   
이미지에서 선, 모서리, 패턴을 뽑아낼 수 있음 (feature 추출)  
문장 번역, 음성 인식(STT), 이미지 캡셔닝  
다대다    ,  다대다      ,  일대다   
  
### CRNN 구조
CRNN = Input -> CNN Layer -> RNN Layer -> Output  
   손글씨 이미지 -> CNN feature(순차적인데이터) -> RNN -> 출력 (손글씨 인식 결과)  
   손글씨 이미지, 스펙트로그램, 영상 프레임 등의 순서가 있는 데이터들  
   왼쪽부터 오른쪽, 주파수이미지, 영상분석(비디오캡셔닝)     
즉 CRNN도 RNN처럼 순서가 있는 데이터로  

•Input  
예: 손글씨 이미지, 스펙트로그램, 영상 프레임 등의 순서가 있는 데이터들  
•CNN Layer  
•2D 이미지 입력을 여러 개의 feature map으로 변환하여 출력함  
•여기서 중요한 건 CNN이 시간축을 보존하도록 커널을 설계해야 함  
(예: 가로로 긴 글자 이미지 → CNN의 출력도 글자 순서를 시퀀스로 해석 가능한 형태여야 함)  
•RNN Layer  
•CNN의 출력(feature sequence)을 받아서 시퀀스를 분석함  
•예: Vanilla RNN, GRU나 LSTM 같은 셀을 사용하여 시퀀스 정보를 처리함  
•Output Layer  
•최종적으로 어떤 값이 출력되어야 할 지 계산함(예: 글자, 음소 등의 값)  
•일반적으로 CTC (Connectionist Temporal Classification)라는 손실함수를 사용함   
→ 글자의 위치를 정확히 몰라도 전체 문자열을 학습 가능함!  
### 활용분야  
• 문자 인식(OCR)  
• (예) 차량 번호판 인식, Captcha 해독, 문서의 디지털화  
• CRNN이 이미지 내의 연속된 글자를 순차적으로 인식함  
• 음성 인식 (Speech-to-Text)  
• 음성 -> 스펙트로그램(시간축에 따른 이미지) -> CRNN  
• CNN으로 주파수의 패턴을 추출하고 RNN으로 순차적으로 데이터를 처리함  
• (예) 자동 자막 생성, 자동 회의록 작성  
• 비디오 분석  
• 연속된 영상 프레임을 분석함  
• (예) CCTV 이상 행동 탐지, 스포츠 영상 분석  
• 악보 인식  
• 이미지로 된 악보 -> 음표 변환  
• (예) 악보의 디지털화,   
• 수어 인식  
• 비디오 분석과 비슷한 방식으로 CRNN 적용 가능  

### CTC Loss
**CTC(Connectionist Temporal Classification) Loss**
cnn에서 나온 feagure map과 rnn을 통해 나온 다대다 시퀀스를 정답인 hello 5글자만 나타야 할 때 같은 상황에서 손실을 구할 때 사용  
즉, 하나의 정답을 표현하는 방법이 여러가지 존재할 때, 가능한 모든 표현 방식을 고려하여 손실을 계산하도록 설계된 함수  
예측 시퀀스로부터 공백과 중복된 문자를 제거하여 최종 예측 결과를 구함  

a를 나타내는 표현으로는 aa, _a, a_ 
aa: 0.4 X 0.4 = 0.16
a_: 0.4 X 0.1 = 0.04
_a: 0.1 X 0.4 = 0.04
-> 0.24  
CTC loss 사용    
"dogs"로 해석될 수 있는 모든 시퀀스 경로들을 인정함.  
( d d o o o g s s뿐만이 아니라, d d o o g g s s, _ d _ o o _ g s, d _ _ o g g _ s, d d d o o g g s s 등도 dogs로 해석되는 여러가지 경로임.)  
CTC는 이 모든 경로의 확률을 더한 값을 정답 "dogs"의 확률로 간주함.  
모델은 이 확률(정답을 맞출 확률)이 최대한 높아지는 쪽으로 학습되고, 그 과정에서 계산되는 손실값이 CTC Loss임.  



### 실습 Captcha 이미지 인식  
Input -> CNN Layer -> RNN Layer -> Output  
이미지를 분해하지않고 통으로 넣은 다음 특징을 추출  

• 목표: CAPTCHA 이미지 속 문자를 자동으로 인식하는 신경망 구현  
• 네트워크 구조: CNN(ResNet) + RNN (GRU)  
• 새로운 개념: CTC Loss  
• 프로젝트의 입력값: 영어 소문자 및 숫자 포함된 CAPTCHA 이미지  
• 프로젝트의 출력값: 예측된 문자 시퀀스  
  
   
## 중간고사
30문제 : 코드 작성 제거 (코드 해설 필요) 구조 파악 필요 무슨 기능인지   
RNN 5주차 생각해보기 답 추가됨  
객관식12, 주관식11, 서술형7   


## 09주차 U-Net
U자형 Network 정보가 압축 되다가 점점 복윈되는 형태  
Image Segementation  
### 등장 배경
Image Classification 문제  
이미지 -> class CNN으로 일부 해소 가능  
이미지 분할(Image segmentation) 문제 픽셀당 분할  
이미지 -> segmentation map  

image Classfication  
object detection (localization)  
Image segmentation  

정밀함에 있어서 한계가 존재했음  
  
Image Classification 문제:   
일반 CNN (입력)이미지 -> Convolution layer -> Pooling -> Fully connected layer -> (출력)Class정보  
이미지 분할(Image segmentation) 문제:  
단순한 FCN(Fully Convolutional Network)을 사용하면,
(입력)이미지 -> Convolution layer -> Pooling -> Convolution layer -> (출력)Segmentation map  
-> 이미지의 기존 위치 정보 소실로, 모서리나 경계선이 뭉개진 저품질의 결과(ex:의료영상 성능 저하)  
-> (IDEA) 이미지의 위치 정보를 유지해줄 skip connection이 필요!  

### 특징 
Encoder-Decoder 구조와 Skip connection을 통해 공간 정보(모양 파악)와 추상 정보(의미)를 동시에 활용함  
(축소)   (복원)    
전체 구조가 대칭적인 U자 형태를 띰  
Encoder-Decoder 구조로 해상도 감소-복원 흐름을 따라감  
Fully Convolutional Network과 비교하면 U-Net은 소규모 데이터셋에서 높은 정확도를 보임  
그러한 특성으로 의료 영상(IRB)처럼 데이터 수집이 어려운 분야에서 많이 활용됨  

### Encoder-Decoder
Encoder: Convolution과 Pooling을 반복하면서 입력된 이미지를 점점 압축하고, 추상적인 특징을 추출해내는 역할을 함  
Decoder: Upsampling을 통해 추출된 특징(압축된 정보)을 점진적으로 복원하고, 원래의 이미지 크기와 유사하게 재구성하는 역할을 함  
### Upsampling
이미지에서 추출한 특징을 이용해 이미지를 복원하는 과정, 이미지의 해상도를 키우는 과정  
겹치는 부분을 더해 나감 2X2 -> 3X3   
### Downsampling = 해상도(H×W)를 줄이는 것  
U-Net에서는 인코더 내부에서 일어나는 해상도를 축소하는 연산 자체를 가리키는 용어(채널수는 상관없음)  
ex) ResNet에서의 다운샘플 -> 모양을 맞춰주기 위해 해상도를 줄이면서 채널수를 늘렸었음!: F(x)와 x를 더해주려면 채널의 수가 같아야 함.  

### Skip Connection
같은 깊이의 인코더 출력과 디코더의 입력을 직접 연결해주는 구조  
U-Net에서는 두 feature map을 concatenate(채널 추가)하거나, 값을 더해줌(다른 연산도 가능)  
64개 채널 U 64개 -> 더해서 128개의 채널
목적: 인코더 초반에서 추출한 공간 정보를 디코더에서 고수준의 의미 정보와 함께 사용해 정밀하게 이미지를 복원할 수 있게 만들기 위해 사용할 수 있음   
경계면 정보 손실을 완화시킴(의료영상 성능 향상)  

### 구조 

Encoder와 Decoder가 각각 여러 층으로 구성되며, 같은 깊이의 인코더와 디코더 층 사이에 skip connection이 있음  
Encoder 층 10개 -> Decoder 층 10개  
Encoder에서 추출한 featere는 decode에서 고해상도 출력을 복원할 때 직접 결합되어 사용됨  
마지막에는 1*1 convolution을 이용해 클래스 수만큼 채널을 줄이고, 이미지의 각 픽셀마다 분류 결과를 출력함  

### 활용 분야

Image Segmentation  
이미지 분할 사람, 배경, 도로 차량  
Image Restoration  
의료영상 노이즈 제거  

### 실습  


















