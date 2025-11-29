# RE-Skin-AI API 명세서 (AI 서버)
**얼굴 3장(정면 / 왼쪽 / 오른쪽)** 을 입력받아 항목 당 분석 결과를 JSON 형태로 반환합니다.
(주름 / 모공 / 입술 건조도는 필드만 잡아두고, 추후 Mediapipe 기반 분석 코드 연결 예정, acne는 분석 가능한 상태)

---

## 공통 정보

- **Base URL (예시)**
  - 개발 환경: `http://127.0.0.1:8000`
  - 배포 환경: `http://{AI_SERVER_HOST}:8000` (백엔드에서 ENV로 관리)

- **인증**
  - 현재 없음 (내부 서버 간 통신용, 필요 시 추후 토큰 추가)

---

## 1. Health Check

AI 서버 상태 확인용 엔드포인트

### `GET /api/health`

#### Request

- Query/body 없음

#### Response

- **200 OK**

```json
{
  "status": "ok"
}

```

---

## 2. 피부 분석 (여드름 + 확장 예정)

정면 / 좌측 / 우측 얼굴 이미지를 업로드하면, 각 view별 여드름 분석 결과와 전체 여드름 심각도 점수를 반환

> ⚠️ 현재는 여드름(acne) 만 실제 분석 결과가 들어가고, wrinkle, pores, lip_dryness 는 null 로 반환됩니다. (필드 스펙만 미리 잡아둔 상태)
> 

### `POST /api/skin/analyze`

- **Method**: `POST`
- **URL**: `/api/skin/analyze`
- **Content-Type**: `multipart/form-data`

### Request (Form fields)

| 필드명 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| front | file (jpg / png) | ✅ | 정면 얼굴 이미지 |
| left | file (jpg / png) | ✅ | 왼쪽 측면 얼굴 이미지 |
| right | file (jpg / png) | ✅ | 오른쪽 측면 얼굴 이미지 |

---

### Response

- **Status code**
    - `200 OK` : 분석 성공
    - `422 Unprocessable Entity` : 필수 필드 누락 등 요청 형식 오류
    - `500 Internal Server Error` : 내부 처리 중 예외 발생
- **Body**: `application/json`

예시:

```json
{
  "acne": {
    "front": {
      "pred_class": "acne",
      "probs": {
        "acne": 0.7246100902557373,
        "pimple": 0.054907187819480896,
        "spot": 0.2204827219247818
      },
      "severity": 0.8236992195248604
    },
    "left": {
      "pred_class": "acne",
      "probs": {
        "acne": 0.5437281131744385,
        "pimple": 0.21097517013549805,
        "spot": 0.24529676139354706
      },
      "severity": 0.7439022436738014
    },
    "right": {
      "pred_class": "acne",
      "probs": {
        "acne": 0.5839601159095764,
        "pimple": 0.16034303605556488,
        "spot": 0.2556968033313751
      },
      "severity": 0.7568749785423279
    },
    "overall_severity": 0.77482548058033
  },
  "wrinkle": null,
  "pores": null,
  "lip_dryness": null
}

```

---

### Response 필드 설명

### 최상위 구조

| 필드명 | 타입 | 설명 |
| --- | --- | --- |
| acne | object | 여드름 분석 결과 블록 |
| wrinkle | object | null |
| pores | object | null |
| lip_dryness | object | null |

---

### `acne` 블록 구조

```json
"acne": {
  "front": { ... },
  "left": { ... },
  "right": { ... },
  "overall_severity": 0.77
}

```

| 필드명 | 타입 | 설명 |
| --- | --- | --- |
| front | object | 정면 이미지에 대한 여드름 분석 결과 |
| left | object | 왼쪽 측면 이미지에 대한 분석 결과 |
| right | object | 오른쪽 측면 이미지에 대한 분석 결과 |
| overall_severity | number | 세 뷰의 severity 평균 (0.0 ~ 1.0) |

---

### 각 view (`front` / `left` / `right`) 구조

예시 (`front` 기준):

```json
"front": {
  "pred_class": "acne",
  "probs": {
    "acne": 0.72,
    "pimple": 0.05,
    "spot": 0.22
  },
  "severity": 0.82
}

```

| 필드명 | 타입 | 설명 |
| --- | --- | --- |
| pred_class | string | 모델이 예측한 최종 클래스 (`"acne"`, `"pimple"`, `"spot"` 중 1개) |
| probs | object | 각 클래스별 softmax 확률 (0~1, 합=1) |
| severity | number | 여드름 심각도 점수 (0.0 ~ 1.0, 아래 식 참고) |

### `probs` 필드

```json
"probs": {
  "acne":   0.72,
  "pimple": 0.05,
  "spot":   0.22
}

```

- `acne` : 활동성 여드름 비율이 높은 패턴
- `pimple` : 뾰루지/개별 병변 위주 패턴
- `spot` : 여드름 자국/반점/흉터 등 잡티 위주 패턴

### `severity` 계산식

여드름 심각도(severity)는 다음과 같이 **가중합**으로 계산되는 방식

```
p_acne   = P(class = "acne")
p_pimple = P(class = "pimple")
p_spot   = P(class = "spot")

severity = 1.0 * p_acne + 0.6 * p_pimple + 0.3 * p_spot
(결과 범위: 0.0 ~ 1.0)

```

- `1.0` : 활동성 여드름(acne)에 가장 높은 가중치
- `0.6` : 뾰루지(pimple)는 중간 정도
- `0.3` : 자국/반점(spot)은 상대적으로 낮은 심각도

**→반환값 해석에 사용하길!!!!**

### severity 값 해석 (권장 구간)

UI/문구 설계 시 참고용:

| 구간 | 레벨 | 설명(예시) |
| --- | --- | --- |
| 0.0 ~ 0.3 | Clear | 거의 여드름이 보이지 않는 깨끗한 피부 |
| 0.3 ~ 0.6 | Mild | 국소적으로 여드름/뾰루지가 보이는 정도 |
| 0.6 ~ 0.8 | Moderate | 얼굴 전반에 여드름이 눈에 띄는 편 |
| 0.8 ~ 1.0 | Severe | 활동성 여드름이 많아 적극적인 관리가 필요한 상태 |

---

## 3. 추후 확장 (wrinkle / pores / lip_dryness) →참고만 하길

- 현재는 JSON에 `wrinkle`, `pores`, `lip_dryness` 필드만 `null`로 포함되어 있음
- 추후 Mediapipe + OpenCV 기반의 분석 모듈이 완성되면, 같은 패턴으로 다음 형식을 권장함

```json
"wrinkle": {
  "front":  { "severity": 0.4, "details": { ... } },
  "left":   { "severity": 0.5, "details": { ... } },
  "right":  { "severity": 0.6, "details": { ... } },
  "overall_severity": 0.5
},
"pores": {
  "...": "..."
},
"lip_dryness": {
  "...": "..."
}

```

각 블록은 `acne`와 동일하게 view별 결과 + `overall_severity` 구조를 따르는 것을 기본 컨벤션으로 하고, 실제 필드 이름/구조는 해당 모듈 구현 후 확정.
