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

### 실습 동물 사진과 배경 분리하기

모델 정의 데이터 호출 및 전처리
Unet ->   resize 

기존의 3가지로 분리되어있는 데이터에서 경계와 동물을 합해서 이진분류로 변경시킴  
  
## 10주차 Auto-Encoder
오토인코더(AutoEncoder)는 입력 데이터를 압축하여 latent vector로 요약하고, 이를 바탕으로 입력을 복원하는 비지도 학습 신경망 구조   
  
• 흐름: 이미지 입력 -> 인코더 통과 -> 잠재 벡터(latent vector) 생성 -> 디코더 통과 -> 복원된 이미지 출력  
                                    잠재 벡터 = 이미지를 잘 요약한 작은 벡터  
  
• 출력은 입력과 가능한 한 유사해야 함  
• 손실함수는 주로 MSE(평균 제곱 오차)를 사용함  
*지도학습이 아니며, 입력을 복원하는 것이 목적임 (입력받는 이미지 자체가 정답이지만 비지도학습임)  
  
UNet에서 Skip Connection이 없는 느낌  
가장 깊은 층에서 latent vector가 작동  
### 왜 쓸까?
정답을 맞추는 것이 목적이 아니라, 데이터의 중요한 특성을 잘 요약하는 인코더(*latent vector*, 잠재 벡터)를 만들고, 그 정보를 기반으로 이미지를 잘 복원하는 디코더를 만들기 위함임  
• 학습 과정에서 모델은 입력의 구조적/의미적 핵심을 latent vector에 응축하고, 그 요약된 표현만으로원본 입력에 가깝게 복원할 수 있도록 학습됨  
• 이 과정은 데이터의 내재된 패턴을 잘 학습하도록 유도함  
• (예) 사람 얼굴 사진에서 주요 특징들을 아주 잘 뽑아내어서 그 특징만으로도 입력된 사람 얼굴과 최대한 비슷한 이미지를 복원할 수 있음  
즉 목적은 정확한 복사가 아닌 의미있는 요약후 의미 있는 복원  

### AutoEncoder 구조
AutoEncoder의 기본 구조 [Encoder] -> (latent vector) -> [Decoder]  
• Encoder: 입력데이터를 점점 압축하여 중요한 특징만 남김  
• Conv2D(이미지) 또는 Dense Layer 등을 사용할 수 있음  
• Latent Vector: 입력의 핵심 정보만을 담고 있는 압축된 표현  
• 즉, 입력을 요약한 정보  
• Decoder: Latent Vector로부터 입력과 가능한 한 비슷하게 복원함  
• ConvTranspose2D(Upsampling) 또는 Dense 등을 사용할 수 있음  
* 인코더와 디코더는 주로 대칭 구조를 이루며, 출력은 입력과 같은 크기와 형태임

다운샘플링 : 5X5 이미지 3X3 커널 stride = 1 -> 3X3 이미지  
대칭을 이루어야함  
업샘플링: 3X3 이미지 3X3 커널 stride = 1 -> 5X5 이미지  
    
### Latent Vector
• "Latent" = 잠재적인, 잠복해 있는  
• 입력의 핵심 정보를 인코더를 통해 압축해 표현한 벡터  
• 추상적인 정보만 포함하며, 이 벡터를 기반으로 복원을 수행함  
• 즉, latent vector는 원본 데이터의 아주 잘 정리된 요약본임!  
• AutoEncoder는 latent vector 하나로 원본 이미지처럼 복원해야 함  
(추가 설명) latent vector는 말 그대로 1D vector([128])일 수도 있지만, 이미지 데이터를 사용하는 AutoEncoder에서는 작은 크기의 3D tensor([8*8*64])일 수도 있음! - 원본 이미지의 요약 정보만 남긴 표현이라는 점에서 이 역시도 latent vector라고 부름  
feature map 과 다른 점은?  
여러개의 feature map  
하나 만으로 복윈하는 핵심 정보만 담은 latent vector  
기술적으로 비슷함  

### Convolutional AutoEncoder(CAE)
• 이미지 데이터를 다룰 때 적합한 합성곱 기반의 AutoEncoder  
• Dense Layer가 아닌 Conv2D, MaxPooling 등의 CNN의 연산을 사용하여 이미지의 공간적 구조(위치, 형태 등)를 잘 보존함  
• (예) 인코더 - Conv2D -> ReLU -> MaxPooling 등을 반복  • 디코더 - ConvTranspose2D(=Upsampling) -> ReLU -> UpSampling2D 등을 반복  
• 장점: 공간정보 유지, 이미지 데이터에 특화  
 ConvTranspose2D(=Upsampling unet 할 때 나왔던 업샘플링)   
 UpSampling2D (별도의 가중치 학습 없이 주변의 픽셀들을 복사하거나 짐작으로 보관으로 이미지 확대 업샘플링)  
 인코더와 디코더는 대칭 구조를 이룸  

### AutoEncoder 활용
노이즈 제거  
  
선명한 이미지(원본) -> 노이즈 추가 -> 인코다 -> latent -> 디코더 -> 출력(원본 이미지와 비교)  
  
이상 탐지  
  
Anomaly Detection: 재구성 오류로 이상 탐지에 활용함 -> latent vector로부터 복원하는 능력이 떨어지면, '이상'으로 판단  
AutoEncoder를 정상 데이터로만 학습하면 정상 데이터 입력에서는 복원 능력이 뛰어남.
그렇지만 학습된 것과 현저히 다른 이상 데이터가 입력되면, AutoEncoder는 정확하게 복원하지 못하고 오차가 커짐.
이 특성을 사용하여 anomaly detection을 할 수 있음 (스팸처리도 가능한가? 속도가 느려서 안되나?)  

차원 축소  
Demensionality Reduction: PCA처럼 사용 가능  
• latent vector를 차원 축소의 결과로 활용 가능(요약 특화)  

### UNet과 비교  
AutoEncoder와 U-Net은 모두 encoder-decoder 구조이지만, 복원의 목적과 방식이 서로 다름
• U-Net은 skip connection을 통해 인코딩 중에 나온 원본에 더 가까운 형태의 여러 feature map을 디코딩 단계에서 받아 데이터의 정밀한 복원에 사용하지만, AutoEncoder는 요약한 후 나온 latent vector 하나만으로 원본 데이터와 비슷하게 복원해 냄  
  
*정밀한 위치 정보나, 경계 보존이 중요한 경우(예 - 의료영상), skip connection을 사용하는 U-Net이 더 유리하고, 성능이 높음  
둘은 즉 목적 자체가 다름  

| 항목 | Autoencoder |U-Net |
|------|---|---|
|목적 | 입력을 그대로 복원(요약된 정보를 얻을때)| 픽셀 단위 예측|
|입력-출력 관계|입력과 동일한 해상도에서 목적에 맞는 값을 얻는 것이 목적 |입력-출력 관계 입력과 동일한 출력을 얻는 것이 목적|
|Skip connection|X| O (복원할 때 힌트가 됨)|
|측력 특징(이미지)|latent vector만을 기반으로 복원한 이미지|skip connection으로 정밀하고 정확하게 경계를 유지한 이미지|
|압축 강도| 높음 (latent vector로 압축)| 낮음 (정보를 많이 유지하려 함)|
|사용 분야| 노이즈 제거, 이상 탐지 등 |의료 영상, 이미지 분할(segmentation)등|


### AutoEncoder 실습
  
입력  
원본 이미지 -> [노이즈 추가] -> 노이즈 추가된 이미지 -> Encoder -> latent vector -> Decoder -> 복원된 이미지  

Mnist데이터 셋 -> 가우시안 노이즈 추가 -> 텐서 변환 및 정규화 -> CNN 블록과 ReLU 조합 Encoder -> CNN블록 + upsample Decoder -> 채널 수 맞추기 -> 학습 진행 -> 이미지 비교  
      
    
      

## 11주차 Let There Be Color 흑백이미지 채색
구조  
레벨 특징 추출기  
  
미들레벨  
  
글로벌 레벨 (broadcast 연산)  
  
가중치를 공유하며 다운스케일링  
  
전역(global) + 지역 정보 결합  
컬러라이제이션 네트워크를 통과하면서 입력 이미지(흑백)에 합해서 채색된 이미지 출력  

### 모델 설명 
• 흑백 이미지를 채색하는 모델  
• 기존 CNN으로도 채색하는 모델을 구현 가능하나, CNN만 사용해서는 전체적인 이미지의 문맥(global context)를 효율적으로 활용하는 데에는 한계가 있었음  
• -> Let There Be Color 모델은 이 한계를 극복하기 위해, 지역 정보와 전역 정보를 동시에 학습하고, 이미지 분류기 앞단의 연산을 병행하여 전역 정보의 품질을 향상시켰음  
  
이미지 분류의 앞단 연산은 왜 사용할까?  
• A: 이미지에 있는 사물이 무엇인지 문맥을 이해하면 더 적절한 생상을 예측할 수 있기 때문  
  
• (예) 하늘 /바다, 사과/토마토  
• 이미지 분류(task classification)의 앞단의 연산을 병행하여 학습해 문맥 정보(문맥을 담고 있는 벡터)를 강화함  
-> 즉, 분류기의 전역 특징 추출 구조는 활용하여 이 정보를 색상 복원에 사용하고, 분류 결과는 실제로 출력하지 않음.  

### 색 공간(Color Space)  
  
색공간: 색을 수치로 표현하기 위한 체계  
• 컴퓨터는 색을 숫자로 처리하므로, 용도에 따라 다양한 종류의 색공간이 있음  
   
종류:  
• RGB(Red, Green, Blue) - 화면 출력용  
• CMYK(Cyan, Magenta, Yellow, Black) - 인쇄용   
• HSV, HSL - 색조 기반 표현  
• Lab - 인간의 시각을 기준으로 정의된 색공간  
이것 사용 (L 밝기, 흑백 ab 색상)  

##### RGB: Red, Green, Blue의 조합으로 색을 표현함  
• 인간이 시각 지각 특성과는 거리가 있음  
• (=R, G, B 값을 각각 변화시킬때 인간이 직관적으로 어떤 색일지 예측하기 어려움)  
• 밝기와 색상 정보가 명확히 분리되어있지 않음.  
###### Lab: 국제조명위원회가 만든 색공간으로, 인간의 시각적 인식에 기반한 색 표현 체계임  
• L*: 밝기(Lightness), 0이면 검정, 100이면 흰색  
• a*: 녹색 <-> 빨강으로 이루어진 축-+  
• b*: 파랑 <-> 노랑으로 이루어진 축-+  
   
• 장점: 시각적으로 균등한 색공간임, 밝기(L)와 색상(ab)을 분리해서 조작할 수 있음  
• -> 색 비교, 색 보정, 채색 모델 등에 적합함  
  
### 모델 구조  
• 입력: 흑백 이미지의 L 채널      +  합해서 표시  
• 출력: 컬러 이미지의 a,b 채널    +   L + ab 채널  
• 구성요소:   
• Low-Level Feature Network  
• Mid-Level Feature Network  
• Global Feature Network  
• Fusion Layer  
• Colorization Network  
                                MID  
• Low-Level Feature Network  -<          +  Fusion Layer  -> Colorization Network  
                                GLOBAL  


##### Low-Level Feature Network
• 역할: 엣지, 질감 등 기초 시각 정보를 추출함  
• 구조: 여러 개의 합성곱 레이어와 활성화 함수로 구성됨(stride로 이미지 크기를 줄임,(stride=2면 1/2) 채널 수 증가)  
##### Mid-Level Feature Network
• 역할: 지역의 의미 정보를 추출하여 이미지의 세부 구조를 파악함  
• 추후 Fusion Layer에서 Global Feature와 합쳐짐.  
• 병합되기 전, 지역 정보 보존하여 갖고 있는 역할을 함.   
• 구조: 풀링없이 합성곱 레이어를 통해 해상도를 유지하며 특징을 추출함    
  
##### Global Feature Network
• 역할: 이미지 전체의 문맥 정보를 파악하여 색상 예측에 도움을 줌  
• 구조: 다운샘플링으로 전역 특징을 추출하고, 이미지 분류 앞단의 구조를 사용해 전역 정보를 학습함  
• (참고) ImageNet으로 학습된 분류기 네트워크의 중간 feature를 뽑는 층까지 가져와 활용할 수 있음!  
###### Fusion Layer
• 역할: 전역 정보와 지역 정보를 결합하여 더 정확한 색상 예측을 가능하게 함  
• 방법: 전역 특징을 지역 특징맵에 broadcast하여(=복제해서 크기를 맞춰주고) 채널 방향으로 결합함   

*broadcast: 낮은 차원의 벡터를 공간적으로 확장하는 연산  
   
##### Colorization Network
• 역할: 업샘플링(X8 8배)을 통해 원래 이미지 크기로 복원하고, 픽셀 단위로 a, b 채널을 예측함  
• 구조: 업샘플링 레이어와 합성곱 레이어를 통해 컬러 이미지를 생성함  

### 실습

캐글 데이터셋
층 구성 
층을 늘리면서 stride = 2 를 통해 이미지 절반으로 압축해서 최종 1/8 배  

글로벌 레벨  
이미지에 대한 문맥 벡터 추출   
  
최종 2채널(a/b) 출력  

rate = 0.001

## 12주차 Attention 알고리즘
Seq2Seq 모델  
(Sequence to Sequence)  
입력 시퀀스를 받아 출력 시퀀스로 변환하는 모델  
핵심 구조는 인코더와 디코더!  
Input Sequence -> Encoder -> Context Vector -> Decoder -> Output Sequence  
Encoder - 입력 시퀀스를 하나의 벡터로 압축  
Decoder - 그 벡터를 바탕으로 출력 시퀀스 생성  
### Seq2Seq 모델의 한계
- Seq2Seq는 모든 입력 시퀀스를 받아 고정된 크기의 벡터로 압축함  
- 디코더는 입력 전체를 직접 보지 못하고, 압축된 벡터만으로 출력 시퀀스를 만들어야 함  
- 즉, 시퀀스(문장)가 길어질수록 중요한 정보가 희석되고 사라질 수밖에 없음  
"나는 어제 친구랑 도서관에서 책을 읽고나서 카페에 가서 아아를 마시고 왔어" -> Seq2Seq: ???  
solution: 출력 시퀀스를 하나씩 만들 때, 입력 시퀀스 중 어느 단어가 중요한지 직접 살피며 생성함  
### Attention Mechanism: 현재 시점에서 중요한 정보에 집중하는 매커니즘
• (idea) 사람이 책을 읽을 때 중요한 단어에 집중하듯, 기계도 마찬가지로 디코딩할 때 그 시점에서 중요한 입력 단어를 직접 골라 보면 더 좋은 출력을 낼 수 있음  
• 디코더에서 문장을 생성할 때, 압축된 벡터뿐만 아니라 전체 입력 시퀀스를 보고, 각 단어의 중요도를 계산하여 가중치를 줌  
• 현재 시점에서 가장 중요한 단어에 더 많이 집중(Attention)하도록 학습함  
### Seq2Seq + Attention
기존 Seq2Seq는 인코더 마지막의 출력 하나만 사용했지만, Attention은 인코더의 모든 출력(hidden state)을 활용해 디코더가 중요한 단어를 동적으로 선택할 수 있도록 함  
(예) "나는 사과를 먹었다"  
  
[입력 문장]     
나는 → 사과를 → 먹었다   
↓         ↓       ↓   
[h₁]    [h₂]     [h₃]  ← 인코더의 각 단어별 hidden state ht      
↓         ↓       ↓   
디코더에서 시점 t마다 attention score 계산 (h₁, h₂, h₃ 각각과 유사도)    
↓  
context vector 생성(h1, h2, h3과 attention score간의 가중합을 구하여 생성)  
↓  
st + ct → 다음 단어 예측  
- st: 지금까지 디코더가 생성한 문맥 정보를 담고 있는 상태 벡터  
- ct: 현 시점에서 입력 중 가장 중요한 정보를 담고 있는 context vector  
  
### 계산 방법
[입력 문장]  
나는 → 사과를 → 먹었다  
↓ ↓ ↓  
[h₁] [h₂] [h₃] ← 인코더의 각 단어별 hidden state ht와, 디코더가 지금까지 생성한 단어들로부터 얻은 hidden state st  
↓ ↓ ↓  
디코더에서 시점 t마다 attention score 계산 (h₁, h₂, h₃과 st와의 유사도)  
↓  
context vector 생성(h1, h2, h3과 attention weight간의 가중합을 구하여 생성)  
↓  
st + context vector  
↓  
다음 단어 예측  
*attention weight: attention score들을 softmax로 정규화하여 나온 가중치 (전체 합 = 1)  
(1) 벡터 h₁ = 나는의 정보  
(2) 벡터 h₂ = 사과를의 정보  
(3) 벡터 h₃ = 먹었다의 정보  
(4) 지금까지 생성한 단어들로 부터 나온 hidden state 벡터 st  
이 St를 기준으로 인코더의 각 hi와 유사도(score)를 계산함.  
score들을 softmax에 넣어 attention score(attention weight)를 계산함  
attention weight와 각 시점의 hidden state를 통해 context vector를 생성함  
현재의 context vector ct와 현재의 hidden state인 st를 활용해 출력(단어 예측 결과)을 생성함  
st: 지금까지 디코더가 생성한 문맥 정보를 담고 있음  
ct(현 시점의 context vector): 입력한 문장에서 참고해야 할 정보  

OUTPUTt = softmax(Wo ' [St;Ct] + b)  
--
|벡터 | 값 |
|------|---|
|h1| [1, 0, 0]|
|h2| [0, 1, 1]|
|h3| [1, 1, 0]|
|st| [2, 1, 1]|


#### 1. Attention Score 계산  
score1 = St ' h1 = [2, 1, 1] ' [1, 0, 0] = 2  
score2 = St ' h2 = [2, 1, 1] ' [0, 1, 1] = 1+1 = 2  
score3 = St ' h3 = [2, 1, 1] ' [1, 1, 0] = 2 +1 =3  
#### 2. Attention Weight 계산 : score의 softmax 정규화 값  
attention weights ~~ [0.212, 0.212, 0.576]  
#### 3. Context Vector 계산  
0.212 ' h1 = [0.212, 0, 0]  
0.212 ' h2 = [0, 0.212, 0.212]  
0.576 ' h3 = [0.576, 0.576, 0.0]   
  
context ~~ [0.212 + 0.576, 0.212 + 0.576, 0.212] = [0.788, 0.788, 0.212]  
#### 4. st와 ct를 사용해 출력 시퀀스 생성   
• 즉, Seq2Seq처럼 벡터 하나만을 사용하지 않고, 매 디코딩 시점마다 동적으로 생성되는 고정된 크기의 context vector를 사용함  
• context vector의 크기는 고정되어 있지만, 시점마다 내용이 다름!  
• 입력 시퀀스의 모든 단어를 부분적으로 반영하면서도 중요한 단어에 집중할 수 있음!  

### 실습 기계번역
GRU + Attention  
LSTM + Attention  
## 13주차 Transformer 모델
• 논문 'Attention Is All You Need'(2017)에서 소개된 딥러닝 모델  
• 순차적 처리 없이도 입력 시퀀스의 관계를 효과적으로 학습할 수 있도록 설계되었음  
**주요 특징**  
• Self-Attention Mechanism: (발전-> Multi-Head Attention Mechanism) 입력 시퀀스 내의 모든 단어 간의 관계를 동적으로 학습함  
• Encoder-Decoder 구조 사용  
• **병렬 처리**: 전체 시퀀스를 동시에 처리할 수 있어 RNN(순차적으로 처리해야 함)보다 학습속도가 빠름  
• 다양한 활용 분야: 자연어 처리, 컴퓨터 비전(ViT), 음성 처리 등 다양한 분야에 활용됨  

### 구조(복잡)

(입력 데이터) -> 토크나이저 -> 임베딩 레이어 -> 인코더 N개 -> 디코더 N개 -> 출력층 -> (출력 결과)  
  
*디코더는 순차적인 생성 작업을 할 때 필요하며, 단순 분류 목적(예: BERT | 감정분류, 감성분석 등)일 경우에는 필요하지 않음.  

• 토크나이저(Tokenizer): 입력 텍스트를 토큰화하고 정수 ID 시퀀스로 인코딩함   
• 임베딩 레이어(Embedding layer): 정수 ID 시퀀스를 각 ID에 해당하는 고차원 벡터(임베딩)로 변환하고, 위치 정보를 더함<-- (전체를 받기 때문에 위치정보 따로 Postional Encoding)  
• Encoder: 전체 시퀀스를 받아 각 단어가 문장 내 다른 모든 단어들과 어떤 관계를 맺고 있는지 학습함.  
단어의 문맥적 의미를 보다 풍부하게 파악할 수 있음. 여러 인코더 층이 쌓여 있음 Attention  
• Decoder: 이전에 생성한 단어들(이전 출력)과 인코더의 출력을 입력받아(Attention) 다음 단어를 순차적으로 예측함. 여러 디코더 층이 쌓여 있음  
• 출력층(Output Layer): 문제 유형에 맞는 최종 결과를 생성함  


왼쪽 - 인코더 스택(n회 반복)  
• Input Embedding + Positional Encoding  
:입력 단어들을 임베딩하고, 위치 정보를 더함  
• Multi-Head Self-Attention  
:입력 문장 안의 단어들끼리 서로 어떤 관계인지 학습함  
• Feed Forward + Add & Norm  
:비선형 변환과 정규화  

오른쪽 - 디코더 스택(n회 반복)  
• Output Embedding + Positional Encoding  
: 이전에 생성된 출력 단어들을 임베딩하고, **위치 정보**를 더함  
• Masked Multi-Head Self-Attention(부정 행위 방지)  
:디코더 내부에서 일어나는 Self-Attention  
:현재 시점 이전의 단어까지만 참조할 수 있도록 masking 적용  
• Linear + Softmax -> Output Probabilities
:(Linear) 마지막 출력 벡터를 단어장의 크기의 벡터로 변환함
:(Softmax) 각 단어가 나올 확률로 변환하고  
:(Output) 가장 확률이 높은 단어를 출력함  


#### Input Embedding + Positional Encoding
토크나이저(Tokenizer):
["the", "cat", "sit", "##s", "."]
→ [101, 1996, 4937, 3523, 2015, 1012, 102] (문장의 끝과 시작 인덱스도 들어가서 2개가 더 많음)  
※ 숫자는 단어에 대응되는 고정된 정수 ID
Input Embedding(각 단어를 벡터로 변환)
[101] → [0.12, -0.38, 0.45, 0.88]
[1996] → [-0.21, 0.13, 0.76, -0.45] 
  
#### Positional Encoding
위 임베딩만으로는 위치정보가 없음. 그래서 Positional Encoding을 더해줘야 함.  
Word: "cat"   
Embedding: [0.10, 0.40, -0.50, 0.30]  
Position (index=1): PE[1] = [0.00, 0.84, 0.91, 0.14]  

- Postional Encoding에서 더해주는 값 PE는 학습을 통해 생성된 값이 아닌 공식에 의해 계산되는 값임  
- "cat" + 1번째 PE -> 문장 중간에 있는 cat이다.  
- "cat" + 10번째 PE => 문장 끝에 있는 cat이다.
  
   
#### Multi-Head Self-Attention
Self-Attention
입력 시퀀스 내의 모든 단어들끼리의 **관계(의존성)** 를 동적으로 파악하는 매커니즘  
단어마다 문맥을 반영한 표현을 만듦  
!!전체 토큰간의 관계를 동시에 계산함(병렬 계산)  
"이 단어가 문장 내에 있는 다른 어떤 단어들과 얼마나 관련이 있을까?"를 계산하는 것  
"The animal didn't cross the street because it was too tired."  
단어 'it'이 가리키는 것이 무엇일까  
?-> Self-Attention은 it이 street보다는 animal과 더 높은 관련성이 있다고 판단하도록 학습함  

Multi-Head Self-Attention
Self-Attention을 다양한 방식으로 여러번 실행해서 정보의 시야를 넓힘(다양한 시각으로 문장을 바라봄)
(예) head1 - 문법적 관계
(예) head2 - 의미적 유사성
(예) head3 - 위치 정보
...
Multi-Head Self-Attention은 self-attention을 여러번하여, 서로 다른 가중치를 병렬적으로 계산함

#### Feed Forward + Add & Norm 층
Feed Forward Network(FFN)
• Self-Attention 결과를 입력받음
• 두 선형 레이어 사이에 ReLU 등의 활성화 함수를 사용함
• 학습하며 비선형성을 추가하는 역할을
함
Add & LayerNorm
• Add: FFN 출력과 원래 입력을 잔차 연결함(Residual Connection)
-> 기울기 소실 방지, 원본 정보 유지
• LayerNorm(Layer Normalization): 잔차가 더해진 결과를 Layer Normalization으로 정규화
-> 학습 안정화, 수렴 속도 개선

#### Output Embedding + Positional Encoding
디코더에 들어가는 입력 문장을 숫자 벡터로 바꾸고, 그 단어가 문장 속 몇 번째인지 위치 정보도 함께 더하는 과정  
Output Embedding: 단어 하나하나를 컴퓨터가 이해할 수 있도록 벡터로 바꿈  
Positional Encoding: 벡터에 단어의 위치 정보를 더해줌  
Q. 이 값이 디코더의 입력으로 사용되는 이유는?  
지금까지 디코더에서 출력했던 시퀀스들을 다시 입력으로 넣어야 다음 단어를 생성할 수 있다.  


#### Masked Multi-Head Self-Attention
디코더의 Self-Attention에서 아직 생성하지 않은 미래의 단어를 참조하지 않도록 Attention Score를 마스킹하여(안 보이게 하여) 계산하는 방식  
예측 시점에 미래의 (정답) 단어를 참조하면 부정행위가 되어버림.  
이를 방지하기 위해 Mask를 사용해 이후 단어에 대한 attention을 0으로 제한해야 함.  

#### Linear, Softmax -> Output  
Linear Layer(선형 변환)  
입력은 디코더의 마지막 출력  
선형 변환을 수행함  
출력은 단어 집합의 차원으로 매핑된 벡터임  
Softmax  
전체 단어 중에서 다음에 나올 가능성이 높은 단어를 예측함  
OUTPUT  

### 다양한 transformer 기반 모델
• BERT(Bidirectional Encoder Representations from Transformers) -> 분류, 분석  
• 트랜스포머 기반 양방향 인코더 표현 모델  

• GPT(Generative Pre-trained Transformer)  
• 생성형 사전학습 트랜스포머 모델   

• T5(Text-to-Text Transfer Transformer)
• 텍스트-투-텍스트 전이 학습 트랜스포머 모델

• BART(Bidirectional and Auto-Regressive Transformer)
• 양방향 자기회귀 트랜스포머 모델

|사용목적 | 인코더 사용 | 디코더 사용 | 대표 모델
|------|---|------|---|
|분류/분석|O|X|BERT|
|텍스트생성/번역/요약|O|O|T5, BART, Transformer|
|텍스트 생성(GPT 계열)|X|O| GPT, GPT-2/3/4|
  
• BERT: 입력 전체를 양방향으로 이해하여 문맥을 정밀하게 파악하는 모델  
• GPT: 이전 단어만 보면서 다음 단어를 생성하는 모델  
• T5: 입력과 출력이 모두 텍스트인 작업을 범용적으로 처리하는 모델  
• BART: BERT처럼 인코딩하고, GPT처럼 생성(디코딩)하는 결합형 모델  


## 14주차 GAN (Generative Adversarial Network)
Generative: 생성의  
**Adversarial: 적대적인**  
Network: 신경망  
  
GAN: 새로운 데이터를 생성하는 딥러닝 모델(2014, Ian Goodfellow)   
GAN만의 특징 - 두 신경망이 경쟁하며 학습함  
• Generator: 가짜 데이터를 생성함 - 학습을 통해 점점 더 정교한 가짜를 만듦  
• Discriminator: 데이터가 진짜인지 가짜인지 판별함 - 점점 더 정확하게 진짜/가짜를 구별함  
  
전반적인 흐름  
[무작위 노이즈 z] -> 생성자 -> 생성자가 만들어낸 가짜 이미지 -> 감별자 Discriminator -> 진짜? 가짜?  
100개                           실제로 존재하는 진짜 이미지 ->   

### GAN의 구조 
GAN의 구조 - Generator 내부 구조
|설명| 출력채널 수 |층 순서| 계층 (Layer)| 입력 크기 → 출력 크기|
|------|---|------|---|
|1| ConvTranspose2d| (1, 100, 1, 1) → (1, 512, 4, 4) |512| 시작: 노이즈를 4×4, feature map으로 확장|
|2| BatchNorm2d| (1, 512, 4, 4) → (1, 512, 4, 4)| 512| 정규화|
|3 |ReLU| 유지 |512 |비선형 활성화|
|4| ConvTranspose2d| (1, 512, 4, 4) → (1, 256, 8, 8)| 256| 해상도 증가|
|5|BatchNorm2d |(1, 256, 8, 8) → (1, 256, 8, 8)| 256| 정규화|
|6| ReLU| 유지| 256|

GAN의 구조 - Discriminator 내부 구조   
• 이미지들을 입력받아 진짜/가짜 여부를 판별하는 역할    
• 진짜 이미지의 레이블을 1, 가짜 이미지의 레이블을 0으로 두고, 이진 분류 문제로 학습함   
  
--
Leaky ReLU        x if x>0  
LeakyReLU(x) = {  
                  ax if x <= 0 (a~~0.01 ~ 0.2)   
알파라는 고정된 값 0 이하(0 이하일 때도 적게나마 반영, 뉴런이 죽지 않게)   
PReLU (Parametic ReLU) 기울기가 고정되있지 않게, a가 학습함에 따라 바뀌는 매게변수  

|위치| 사용 함수| 출력 범위| 사용하는 이유|
|----|----------|---------|---------------|
|Generator 은닉층| ReLU| 0 ~ ∞| 음수는 제거하고 양수만 반영→ 빠른 수렴, 비선형성 확보(레이어 깊이 ㅅ)|
|Generator 마지막 층| Tanh -1 ~ 1| 이미지 출력값을 정규화→ 픽셀값 범위와 맞춤|
|Discriminator 은닉층| LeakyReLU| 음수 영역 포함|음수도 일부 반영→ 죽은 뉴런 문제 방지, 안정적 학습|
|Discriminator 마지막 층| Sigmoid| 0 ~ 1|진짜일 확률 반환→ 이진 분류를 위한 확률값 출력|
  
*죽은 뉴런 문제: 뉴런의 출력이 항상 0이 되어버려서 더이상 해당 뉴런이 학습에 기여하지 않는 문제  


• 무작위 입력을 바탕으로 사람의 얼굴을 생성하는 GAN  
• 저해상도 이미지를 입력받아 고해상도 이미지를 생성하는 GAN (ex:SRGAN super revolution)  
• 텍스트를 기반으로 이미지를 생성하는 GAN (ex: DALL-E)  
• 이미지의 스타일을 변환해주는 GAN  
• 의료 영상의 노이즈를 제거해주는 GAN  
• 디자인 시안을 제작해주는 GAN  













### 실습: 사람 얼굴 생성하는 GAN
### 실습: 화질을 개선하는 SRGAN

  
중간고사랑 유사하게 나옴  
중간고사 이후부터 14주차 내용까지 전부 시험 범위  


