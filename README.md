이것은 Multi Layer Perceptron(mlp) 스터디 당시 설명을 위해 사용했던 코드 입니다.

코드에는 모두 original source URL 이 포함되어 있으며, 필요한 부분을 수정해 사용하였습니다.  



## 파일 설명

back_propagation.py - 히든 레이어가 1층인 신경망을 구현한 코드입니다. 백프로파게이션을 실제로 어떻게 구현하는지 설명하기 위한 목적으로 쓰였으며, 코드를 본다면 각 층에서 입력이 어떤 형태로 변하는지를 (A*B) 형식으로 작성해가면서 읽는다면 이해에 도움이 되리라 생각합니다. 추가로 넣은 예시 데이터셋에 대한 작업은 아직 진행 중입니다. 

mlp_plot.ipynb - mlp 가 데이터를 어떻게 구분하는 지를 시각적으로 나타낸 플롯들과 sklearn 에서 제공하는 MLPClassifier 을 이용하여  mnist 문제를 풀어본 코드가 있습니다. 





## 추가로 해야할 일

-back_propagation 예시 데이터 셋 학습 하는 과정 완성 및 기존의 코드와 구분 짓기

-mlp_plot 에서 preamable.py 파일 인식되도록 수정하기

