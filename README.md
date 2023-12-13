# Transformer - Multi30K 예제 (독일어 - 영어)

## 들어가기 전

- 본 문서는 Transformer에 대한 기본 지식을 가진 딥러닝 초보자를 위한 문서입니다.
- 이 프로젝트는 "Attention Is All You Need" 논문을 기반으로 한 트랜스포머 모델의 기본 구현을 다루며, Google Colab에서 실행되도록 설계되었습니다. 이 튜토리얼은 독일어에서 영어로 번역하는 모델을 구축하고 훈련시키는 과정을 보여줍니다.

## Transformer 란?

- 2017년 구글에서 소개한 <Attention is All You Need> 논문에서 처음 등장하였으며, Attention Mechanism 만을 활용하여 Neural Network 구조를 쌓지 않고 크고 제한적인 데이터를 효과적으로 처리할 수 있는 기술입니다.
- Attention (2015) 년 이전까지의 RNN은 모두 고정된 크기의 Context Vector 를 사용하여 문장을 압축시키는 과정에서 Bottleneck 현상이 발생하는 등의 단점을 가지고 있었습니다.
- Attention 이 소개된 이후로는 입력된 Sequence(문장) 전체에서 정보를 추출하여 결과값을 제공하는 기술을 사용하였고, 이는 눈부신 성능 향상을 가져왔습니다.
- 이 기술을 기반으로 하여 이후 GPT, BERT 등 기계번역 분야의 발전이 이루어질 수 있었던 계기가 되었습니다.
- Encoder 와 Decoder 파트로 구성되어 있으며, 각 파트는 Self-Attention 레이어와 Neural Network 로 구성 되어 있습니다.

## 데이터 셋 소개

- Multi30k 데이터 셋을 사용했습니다.
- Multi30k는 독일어와 영어로 된 약 30,000개의 문장 쌍을 포함하며, 이미지 설명 번역 작업에 주로 사용됩니다.
- 원 코드에서는 torchtext를 통해 바로 다운로드 하였으나, 해당 서버가 더이상 살아있지 않아, github에서 원문 파일을 직접 다운로드 하여 Google Drive 에 업로드한 뒤, Colab에서 불러와 실행했습니다.
- Dataset 구성
    - Training datatset : 29,000개
    - Validation dataset : 1,014개
    - Test dataset : 1,000개

## 전처리 과정

1. spacy 라이브러리를 이용하여 영어와 독일어 문장의 토큰화 (Tokenization) 를 진행합니다.
2. 모듈 설치 후, 독일어와 영어를 토큰화 하였습니다.
3. 토큰화를 하고 나면, 아래와 같이 예시로 테스트를 진행하였습니다.

    ```python
    # 간단히 토큰화(tokenization) 기능 써보기
    tokenized = spacy_en.tokenizer("I am a graduate student.")

    for i, token in enumerate(tokenized):
    print(f"인덱스 {i}: {token.text}")


    결과
    인덱스 0: I
    인덱스 1: am
    인덱스 2: a
    인덱스 3: graduate
    인덱스 4: student
    인덱스 5: .
    ```

4. 독일어와 영어의 토큰화 함수를 정의하고, torchtext의 Field 라이브러리를 사용하여, 데이터를 처리합니다.
    - Field 라이브러리를 통해 문장 시작 지점에는 <sos> 토큰을 추가하고, 문장의 끝 지점에는 <eos> 를 추가하는 작업을 진행할 수 있습니다.
5. 이후 문장을 불러와서 토큰화를 진행합니다.
6. 최소 2회 이상 등장한 단어를 선택하여 독일어와 영어 사전을 구축합니다.
    - 이를 통해 각 언어 별 초기 dimension을 파악할 수 있습니다.
    - String to I 를 호출하여, 실제 어떠한 index 값에 해당하는지 확인을 해 보았습니다.
    - 0~3번 토근은 실제 존재하는 단어는 아니지만, 네트워크가 각 문장을 적절하게 학습할 수 있도록 도움을 주는 토큰값입니다.

    ```python
    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)
    
    print(f"len(SRC): {len(SRC.vocab)}")
    print(f"len(TRG): {len(TRG.vocab)}")
    
    결과
    len(SRC): 7853
    len(TRG): 5893
    
    print(TRG.vocab.stoi["abcabc"]) # 없는 단어: 0
    print(TRG.vocab.stoi[TRG.pad_token]) # 패딩(padding): 1
    print(TRG.vocab.stoi["<sos>"]) # <sos>: 2
    print(TRG.vocab.stoi["<eos>"]) # <eos>: 3
    print(TRG.vocab.stoi["hello"])
    print(TRG.vocab.stoi["dad"])
    
    0
    1
    2
    3
    4112
    1699
    ```

## 인코더
1. Multi Head Attention 구조
    1. H개의 서로 다른 Attention 컨셉을 만들어서, 1개의 단어도 H개의 다양한 특징을 학습할 수 있도록 하는 구조입니다.
    2. Attention은 3가지 요소를 입력값으로 필요로 합니다.
        1. Query : 단어 행렬, 영향을 받는 단어 A
        2. Key : 영향을 주는 단어 B를 나타내는 변수
        3. Value : 가중치 행렬
2. Feed Forward Nueral Network
    1. 입력과 출력의 차원이 동일해야 합니다.
    2. 인코딩 레이어를 여러번 중첩하여 사용합니다.
    3. 문장에 빈 공간을 채우는 <pad> 토큰은 mask 처리 하여 값을 0으로 설정합니다.
    4. Positional Encoding 값을 구하여 문장의 임베딩과 더하고, 해당 값을 출력하여 디코더로 가져가게 됩니다.

## 디코더
1. 입력과 출력의 차원이 동일합니다.
2. Masked Multi Head Attention 레이어에서 Encoder의 출력값 (enc_src)를 Attention 합니다.
    1. 이 때 디코더의 Query 를 이용합니다.

## 모델 학습
1. Hidden Layer 의 Dimension은 256으로 설정하였으며, Head 개수는 8개로 설정하였습니다.
2. 논문과 동일하게 내부 Dimension은 512로 설정하였습니다.
3. Dropout 은 0.1 로 설정하였습니다.
4. Adam Optimizer 를 사용하였으며, Learning Rate 는 0.0005로 설정하였습니다.
5. 학습 시 마지막 <eos> 토큰은 제외하며, <sos> 토큰부터 시작하도록 처리하였습니다.
6. Epoch는 10회로 설정하였고, 실제 학습 시간은 1 Epoch 당 15초 내외로 소요되었습니다.

    ```python
    Epoch: 01 | Time: 0m 17s
	    Train Loss: 4.221 | Train PPL: 68.073
	    Validation Loss: 3.052 | Validation PPL: 21.164
    Epoch: 02 | Time: 0m 14s
    	Train Loss: 2.815 | Train PPL: 16.691
    	Validation Loss: 2.301 | Validation PPL: 9.985
    Epoch: 03 | Time: 0m 14s
    	Train Loss: 2.234 | Train PPL: 9.335
    	Validation Loss: 1.974 | Validation PPL: 7.201
    Epoch: 04 | Time: 0m 14s
    	Train Loss: 1.885 | Train PPL: 6.586
    	Validation Loss: 1.803 | Validation PPL: 6.065
    Epoch: 05 | Time: 0m 16s
    	Train Loss: 1.636 | Train PPL: 5.135
    	Validation Loss: 1.712 | Validation PPL: 5.542
    Epoch: 06 | Time: 0m 15s
    	Train Loss: 1.447 | Train PPL: 4.250
    	Validation Loss: 1.652 | Validation PPL: 5.218
    Epoch: 07 | Time: 0m 15s
    	Train Loss: 1.297 | Train PPL: 3.657
    	Validation Loss: 1.624 | Validation PPL: 5.075
    Epoch: 08 | Time: 0m 15s
    	Train Loss: 1.169 | Train PPL: 3.219
    	Validation Loss: 1.632 | Validation PPL: 5.115
    Epoch: 09 | Time: 0m 15s
    	Train Loss: 1.062 | Train PPL: 2.892
    	Validation Loss: 1.608 | Validation PPL: 4.991
    Epoch: 10 | Time: 0m 15s
    	Train Loss: 0.968 | Train PPL: 2.633
    	Validation Loss: 1.645 | Validation PPL: 5.180
    ```

## 검증
- 소스 문장과 타겟 문장을 출력하고, 모델 출력 결과를 비교하였습니다.
- 임의로 10번째의 문장을 가져와서 비교하였습니다.

    ```python
    example_idx = 10

    src = vars(test_dataset.examples[example_idx])['src']
    trg = vars(test_dataset.examples[example_idx])['trg']

    print(f'소스 문장: {src}')
    print(f'타겟 문장: {trg}')

    translation, attention = translate_sentence(src, SRC, TRG, model, device, logging=True)

    print("모델 출력 결과:", " ".join(translation))

    결과

    소스 문장: ['ein', 'junger', 'mann', 'macht', 'auf', 'einem', 'skateboard', 'ein', 'kunststück', 'in', 'der', 'luft', '.']
    타겟 문장: ['a', 'young', 'man', 'performs', 'an', 'aerial', 'stunt', 'on', 'a', 'skateboard', '.']
    전체 소스 토큰: ['<sos>', 'ein', 'junger', 'mann', 'macht', 'auf', 'einem', 'skateboard', 'ein', 'kunststück', 'in', 'der', 'luft', '.', '<eos>']
    소스 문장 인덱스: [2, 5, 96, 13, 68, 12, 6, 186, 5, 421, 7, 15, 90, 4, 3]
    모델 출력 결과: a young man does a trick on a skateboard in midair . <eos>
    ```

- 소스 문장인 독일어 문장을 구글 번역기를 이용해 번역해보았습니다.

    → **한 청년이 스케이트보드를 타고 공중에서 묘기를 선보이고 있습니다.**

    → **A young man does a stunt in the air on a skateboard.**

- Attention을 시각화 하여, 각 Head 별 단어의 가중치를 한눈에 확인하였습니다.

    ![Attention Visualization](https://example.com/attention_visualization.png)

    - Young, Man, stakeboard와 같은 단어는 지속적으로 높은 attention score를 가진 것을 확인할 수 있습니다.

## 결론

이 코드 예제를 통해 Transformer 모델의 기본 구조와 작동 방식에 대한 이해를 높일 수 있었습니다.
Multi30k 데이터셋을 활용한 이번 프로젝트는 독일어에서 영어로의 번역 과정을 통해 모델의 성능을 시험하고, 실제 언어 처리 작업에 Transformer를 어떻게 적용할 수 있는지를 보여줍니다. 

이 문서는 Transformer에 대한 기본적인 이해를 바탕으로, 실제 모델 구현과 훈련 과정을 따라가며 실습해 볼 수 있는 기회를 제공합니다. 딥러닝과 기계 번역에 관심 있는 이들에게 유익한 참고 자료가 되기를 기대합니다.

---

본 튜토리얼은 "Attention Is All You Need" 논문을 기반으로 하며, Google Colab에서 실행되도록 구성되었습니다.

**참고 문헌**
- Vaswani, A., et al. (2017). Attention is All You Need. 
- Multi30k Dataset: [https://github.com/multi30k/dataset](https://github.com/multi30k/dataset)
- https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb

---

본 문서는 교육 및 학습 목적으로 작성되었습니다. 
