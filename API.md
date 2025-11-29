# RE-Skin-AI API 명세서 (AI 서버)
얼굴 사진 3장(정면 / 왼쪽 / 오른쪽)을 입력받아, 항목별 피부 상태를 분석한 결과를 JSON 형태로 반환하는 AI 서버

- 분석 항목:
  - 여드름 (**acne**) – ResNet50 분류 모델
  - 주름 (**wrinkle**) – Mediapipe FaceMesh + Canny edge 기반 규칙 분석
  - 모공 (**pores**) – OpenCV 기반 규칙 분석
  - 입술 건조도 (**lip_dryness**) – Mediapipe FaceMesh 기반 규칙 분석

---

## 공통 정보

- **Base URL (예시)**
  - 개발 환경: `http://127.0.0.1:8000`
  - 배포 환경: `http://{AI_SERVER_HOST}:8000` (백엔드에서 ENV로 관리)

- **인증**
  - 현재 없음 (백엔드 서버 ↔ AI 서버 간 내부 통신용)
  - 필요 시 추후 토큰 기반 인증 추가 예정

---

## 1. Health Check

AI 서버 상태 확인용 엔드포인트

### `GET /api/health`

#### Request

- Query / Body 없음

#### Response

- **200 OK**

```json
{
  "status": "ok"
}
````

---

## 2. 피부 분석 (Acne / Wrinkle / Pores / Lip Dryness)

정면 / 좌측 / 우측 얼굴 이미지를 업로드하면, 각 view별 분석 결과와 전체 심각도 점수를 반환합니다.

### `POST /api/skin/analyze`

* **Method**: `POST`
* **URL**: `/api/skin/analyze`
* **Content-Type**: `multipart/form-data`

### 2.1. Request (Form fields)

| 필드명   | 타입               | 필수 | 설명            |
| ----- | ---------------- | -- | ------------- |
| front | file (jpg / png) | ✅  | 정면 얼굴 이미지     |
| left  | file (jpg / png) | ✅  | 왼쪽 측면 얼굴 이미지  |
| right | file (jpg / png) | ✅  | 오른쪽 측면 얼굴 이미지 |

**제약사항(권장)**

* 얼굴이 중앙에 가깝게 위치한 상반신 셀피 권장
* 너무 어둡거나 과도한 보정/필터가 들어간 이미지는 분석 정확도 저하 가능

---

### 2.2. Response 개요

* **Status code**

  * `200 OK` : 분석 성공
  * `422 Unprocessable Entity` : 필수 필드 누락 등 요청 형식 오류
  * `500 Internal Server Error` : 내부 처리 중 예외 발생

* **Body**: `application/json`

최상위 구조:

```json
{
  "acne": { ... },
  "wrinkle": { ... },
  "pores": { ... },
  "lip_dryness": { ... }
}
```

각 블록은 공통적으로 다음 구조를 가집니다.

```json
{
  "front": { ... },
  "left":  { ... },
  "right": { ... },
  "overall_severity": 0.0
}
```

---

### 2.3. Response 예시

실제 샘플 이미지로 분석한 예시입니다.

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
  "wrinkle": {
    "front": {
      "severity": 0.43990872210953347
    },
    "left": {
      "severity": 0.3622482131254061
    },
    "right": {
      "severity": 0.25688073394495414
    },
    "overall_severity": 0.3530125563932979
  },
  "pores": {
    "front": {
      "severity": 0.1446685791015625
    },
    "left": {
      "severity": 0.055361
    },
    "right": {
      "severity": 0.0394140749874412
    },
    "overall_severity": 0.07981455136300124
  },
  "lip_dryness": {
    "front": {
      "severity": 0.015676567656765675
    },
    "left": {
      "severity": 0.0
    },
    "right": {
      "severity": 0.0
    },
    "overall_severity": 0.005225522552255225
  }
}
```

※ 실제 값은 이미지에 따라 달라지며, 대부분 0.0 ~ 1.0 범위의 실수이니 이 점 참고하시길길!

---

### 2.4. 최상위 필드 설명

| 필드명           | 타입     | 설명              |
| ------------- | ------ | --------------- |
| `acne`        | object | 여드름 분석 결과 블록    |
| `wrinkle`     | object | 주름 분석 결과 블록     |
| `pores`       | object | 모공 분석 결과 블록     |
| `lip_dryness` | object | 입술 건조도 분석 결과 블록 |

각 블록은 공통적으로 다음 필드를 가집니다.

| 필드명                | 타입     | 설명                           |
| ------------------ | ------ | ---------------------------- |
| `front`            | object | 정면 이미지에 대한 분석 결과             |
| `left`             | object | 왼쪽 측면 이미지에 대한 분석 결과          |
| `right`            | object | 오른쪽 측면 이미지에 대한 분석 결과         |
| `overall_severity` | number | 세 뷰의 severity 평균 (0.0 ~ 1.0) |

---

## 3. Acne 블록 상세

### 3.1. 구조

```json
"acne": {
  "front":  { ... },
  "left":   { ... },
  "right":  { ... },
  "overall_severity": 0.77
}
```

각 view(`front` / `left` / `right`)는 다음과 같은 구조를 가집니다.

```json
"front": {
  "pred_class": "acne",
  "probs": {
    "acne":   0.72,
    "pimple": 0.05,
    "spot":   0.22
  },
  "severity": 0.82
}
```

| 필드명          | 타입     | 설명                                                   |
| ------------ | ------ | ---------------------------------------------------- |
| `pred_class` | string | 모델이 예측한 최종 클래스 (`"acne"`, `"pimple"`, `"spot"` 중 1개) |
| `probs`      | object | 각 클래스별 softmax 확률 (0~1, 합 = 1)                       |
| `severity`   | number | 여드름 심각도 점수 (0.0 ~ 1.0, 아래 계산식 참고)                    |

### 3.2. `probs` 필드

```json
"probs": {
  "acne":   0.72,
  "pimple": 0.05,
  "spot":   0.22
}
```

* `acne`   : 활동성 여드름이 많은 패턴
* `pimple` : 개별 뾰루지/병변 위주 패턴
* `spot`   : 여드름 자국/반점/흉터 등 잡티 위주 패턴

### 3.3. `severity` 계산식

여드름 심각도(severity)는 다음과 같은 **가중합**으로 계산됩니다
결과값 해석 시에 참고하면 될 듯합니다.

```text
p_acne   = P(class = "acne")
p_pimple = P(class = "pimple")
p_spot   = P(class = "spot")

severity = 1.0 * p_acne + 0.6 * p_pimple + 0.3 * p_spot
(결과 범위: 0.0 ~ 1.0)
```

* `1.0` : 활동성 여드름(acne)에 가장 높은 가중치
* `0.6` : 뾰루지(pimple)는 중간 정도 가중치
* `0.3` : 자국/반점(spot)은 상대적으로 낮은 심각도로 취급

### 3.4. Severity 값 해석 (권장 구간)

UI/문구 설계 시 참고용:

| 구간        | 레벨       | 설명(예시)                      |
| --------- | -------- | --------------------------- |
| 0.0 ~ 0.3 | Clear    | 거의 여드름이 보이지 않는 깨끗한 피부       |
| 0.3 ~ 0.6 | Mild     | 국소적으로 여드름/뾰루지가 보이는 정도       |
| 0.6 ~ 0.8 | Moderate | 얼굴 전반에 여드름이 눈에 띄는 편         |
| 0.8 ~ 1.0 | Severe   | 활동성 여드름이 많아 적극적인 관리가 필요한 상태 |

※ 실제 UX에서는 문구/색상 등은 서비스 기획에 따라 조정 가능

---

## 4. Wrinkle 블록 상세

### 4.1. 구조

```json
"wrinkle": {
  "front": {
    "severity": 0.44
  },
  "left": {
    "severity": 0.36
  },
  "right": {
    "severity": 0.26
  },
  "overall_severity": 0.35
}
```

현재 각 view는 다음 필드를 가집니다.

| 필드명      | 타입     | 설명                             |
| -------- | ------ | ------------------------------ |
| severity | number | 주름 정도 (0.0 ~ 1.0, 1에 가까울수록 많음) |

### 4.2. 계산 방식(요약)

* **기술 스택**: Mediapipe FaceMesh + OpenCV (Canny edge)

* 주요 절차:

  1. FaceMesh를 사용해 왼쪽 눈/눈썹 주변 랜드마크를 추출
  2. 해당 영역(bounding box + padding)을 ROI로 잘라냄
  3. ROI를 그레이스케일 + 블러 후 Canny edge 적용
  4. edge 픽셀 비율을 기반으로 0.0 ~ 1.0 사이 값으로 스케일링

* 해석:

  * 0.0 에 가까울수록 주름 패턴이 거의 없는 상태
  * 1.0 에 가까울수록 눈가 주름/잔주름 패턴이 뚜렷한 상태

---

## 5. Pores 블록 상세

### 5.1. 구조

```json
"pores": {
  "front": {
    "severity": 0.14
  },
  "left": {
    "severity": 0.06
  },
  "right": {
    "severity": 0.04
  },
  "overall_severity": 0.08
}
```

각 view는 다음 필드를 가집니다.

| 필드명      | 타입     | 설명                             |
| -------- | ------ | ------------------------------ |
| severity | number | 모공 정도 (0.0 ~ 1.0, 1에 가까울수록 많음) |

### 5.2. 계산 방식(요약)

* **기술 스택**: OpenCV

* 주요 절차:

  1. 그레이스케일 변환 후 `medianBlur` 적용
  2. `CLAHE` 로 대비 향상
  3. `adaptiveThreshold` + `erode` 로 모공 후보 영역 추출
  4. YCrCb 기반 스킨 마스크로 피부 영역만 남김
  5. 모공 후보 픽셀 수 / 전체 피부 픽셀 수 → 0.0 ~ 1.0 사이 비율로 변환

* 해석:

  * 0.0 에 가까울수록 모공이 잘 보이지 않는 상태
  * 1.0 에 가까울수록 모공이 전체 피부에서 차지하는 비율이 높은 상태

---

## 6. Lip Dryness 블록 상세

### 6.1. 구조

```json
"lip_dryness": {
  "front": {
    "severity": 0.02
  },
  "left": {
    "severity": 0.0
  },
  "right": {
    "severity": 0.0
  },
  "overall_severity": 0.01
}
```

각 view는 다음 필드를 가집니다.

| 필드명      | 타입     | 설명                               |
| -------- | ------ | -------------------------------- |
| severity | number | 입술 건조도 (0.0 ~ 1.0, 1에 가까울수록 건조함) |

### 6.2. 계산 방식(요약)

* **기술 스택**: Mediapipe FaceMesh + OpenCV

* 주요 절차:

  1. FaceMesh의 `FACEMESH_LIPS` 인덱스를 사용해 입술 영역 polygon 추출
  2. 입술 영역을 mask로 추출 후 그레이스케일 변환
  3. 일정 threshold 이상(예: 190)인 밝은 픽셀 비율을 계산

     * 입술 각질/건조가 심할수록 밝은 픽셀이 많아지는 경향
  4. 0.0 ~ 1.0 범위로 정규화

* 해석:

  * 0.0 에 가까울수록 촉촉한 입술 상태
  * 1.0 에 가까울수록 각질/건조가 심한 상태

---

## 7. 공통 요약

* 모든 severity 값은 **0.0 ~ 1.0 사이의 연속값(float)**
  (프론트에서 필요에 따라 반올림하여 사용)
* 각 블록(`acne`, `wrinkle`, `pores`, `lip_dryness`)은
  공통적으로 `front`, `left`, `right`, `overall_severity` 필드를 갖는다.

---
