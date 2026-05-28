# ummtorch

Tensors and Dynamic neural networks in umjunsik-lang with no GPU acceleration (planned)

---

## 요구사항

```bash
pip install umjunsik
```

---

## 실행

```bash
# 에포크마다 x1, x2, target을 차례로 입력
printf '1\n0\n1\n1\n0\n1\n1\n0\n1\n' | umjunsik mlp.umm
```

출력 형식:

```
Y=2-0 ERR=1-0
Y=0-2 ERR=0-3
Y=2-0 ERR=1-0
```

`Y=P-N`은 예측값, `ERR=P-N`은 오차를 나타냅니다. P와 N은 각각 양수 부분과 음수 부분이며 `값 = P - N`입니다. `Y=0-2`는 예측값이 -2라는 의미입니다.

---

## 아키텍처

```
Input(2) → Hidden(2, ReLU) → Output(1)
```

손실 함수는 MSE, 옵티마이저는 SGD입니다.

---

## 구현 방식

엄랭에는 음수 리터럴, 배열, 부동소수점이 없고 조건 분기는 0일 때 점프 입니다. 

### 부호 있는 정수: P/N 분해

모든 값을 비음수 정수 두 개 P와 N으로 저장하고 `값 = P - N`으로 해석합니다. 연산 후에는 둘 중 하나가 0이 되도록 정규화합니다.

```
 2  →  P=2, N=0
-3  →  P=0, N=3
```

사칙연산은 다음과 같이 전개됩니다.

```
덧셈:  dstP += srcP; dstN += srcN; 정규화
뺄셈:  srcP와 srcN을 뒤집어서 더하기
곱셈:  dstP = aP·bP + aN·bN
       dstN = aP·bN + aN·bP
```

덧셈은 엄랭의 유일한 루프 방식인 감소 루프로 구현합니다.

```
while TMP != 0:
    TMP -= 1
    dst += 1
```

### ReLU

음수 부분을 버리는 것으로 구현합니다.

```
ReLU(Z)  →  HP = ZP,  HN = 0
```

### 메모리 레이아웃

엄랭은 배열이 없으므로 256개의 정수 슬롯을 직접 인덱싱해서 사용합니다.

```
인덱스   이름                설명
─────────────────────────────────────────────
 0–1    X1P, X1N            입력 x1
 2–3    X2P, X2N            입력 x2
 4–5    TGP, TGN            타겟
 6–13   W1 (P,N × 4)        은닉층 가중치
14–17   B1 (P,N × 2)        은닉층 바이어스
18–23   W2, B2 (P,N × 3)    출력층 가중치·바이어스
24–35   Z1, H, Z2, Y        순전파 중간값
36–63   DY, DW, DH, DB      기울기
64–65   TMPP, TMPN          임시
66      EPOCH               에포크 카운터
```

---

→ [github.com/rycont/umjunsik-lang](https://github.com/rycont/umjunsik-lang)
