from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.skin_service import SkinAnalysisService


app = FastAPI(
    title="RE-Skin-AI",
    description="Face skin analysis AI service (acne + wrinkle/pores/lip in future)",
    version="0.1.0",
)

# CORS 설정 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 인스턴스 (device=None이면 내부에서 cuda/ cpu 자동 선택하는 구조라면 None으로 놔도 됨)
service = SkinAnalysisService(device=None)


def file_to_pil(file: UploadFile) -> Image.Image:
    """FastAPI UploadFile -> PIL.Image 로 변환"""
    contents = file.file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    return img


@app.get("/api/health")
async def health_check():
    """서버 확인용 체크 엔드포인트"""
    return {"status": "ok"}


@app.post("/api/skin/analyze")
async def analyze_skin(
    front: UploadFile = File(..., description="정면 얼굴 이미지"),
    left: UploadFile = File(..., description="왼쪽 측면 얼굴 이미지"),
    right: UploadFile = File(..., description="오른쪽 측면 얼굴 이미지"),
):
    """
    정면/좌/우 3장의 얼굴 이미지를 입력받아
    여드름 + (추후) 주름/모공/입술 건조도 분석 결과를 반환
    """
    front_img = file_to_pil(front)
    left_img = file_to_pil(left)
    right_img = file_to_pil(right)

    result = service.analyze(front_img, left_img, right_img)
    return result
