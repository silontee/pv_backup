# -*- coding: utf-8 -*-
"""
PRD v4 기반 발표용 PPT 생성 스크립트
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# 색상 팔레트
DARK_BG = RGBColor(0x1A, 0x1A, 0x2E)       # 짙은 남색
ACCENT_BLUE = RGBColor(0x00, 0x96, 0xC7)    # 파란색
ACCENT_ORANGE = RGBColor(0xF7, 0x7F, 0x00)  # 주황
ACCENT_GREEN = RGBColor(0x2D, 0xCE, 0x89)   # 초록
ACCENT_RED = RGBColor(0xF5, 0x36, 0x5C)      # 빨강
ACCENT_YELLOW = RGBColor(0xFF, 0xD6, 0x00)   # 노랑
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xE0, 0xE0, 0xE0)
MID_GRAY = RGBColor(0x99, 0x99, 0x99)
DARK_TEXT = RGBColor(0x33, 0x33, 0x33)
CARD_BG = RGBColor(0x24, 0x24, 0x3E)        # 카드 배경

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def add_bg(slide, color=DARK_BG):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_shape(slide, left, top, width, height, color, alpha=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape

def add_rect(slide, left, top, width, height, color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape

def add_text_box(slide, left, top, width, height, text, font_size=18, color=WHITE, bold=False, align=PP_ALIGN.LEFT, font_name="맑은 고딕"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = align
    return txBox

def add_bullet_text(slide, left, top, width, height, items, font_size=16, color=WHITE, spacing=Pt(8)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "맑은 고딕"
        p.space_after = spacing
        p.level = 0
    return txBox

def add_card(slide, left, top, width, height, title, body_lines, title_color=ACCENT_BLUE, icon=None):
    card = add_shape(slide, left, top, width, height, CARD_BG)
    # 상단 액센트 바
    add_rect(slide, left, top, width, Inches(0.06), title_color)
    # 제목
    add_text_box(slide, left + Inches(0.3), top + Inches(0.2), width - Inches(0.6), Inches(0.5),
                 title, font_size=18, color=title_color, bold=True)
    # 내용
    y = top + Inches(0.7)
    for line in body_lines:
        add_text_box(slide, left + Inches(0.3), y, width - Inches(0.6), Inches(0.35),
                     line, font_size=13, color=LIGHT_GRAY)
        y += Inches(0.32)

# ============================================================
# SLIDE 1: 표지
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

# 상단 라인
add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), ACCENT_BLUE)

# 부제
add_text_box(slide, Inches(1), Inches(2.0), Inches(11), Inches(0.6),
             "한국남동발전 내부 의사결정 지원 플랫폼", font_size=20, color=ACCENT_BLUE, bold=False)

# 메인 제목
add_text_box(slide, Inches(1), Inches(2.6), Inches(11), Inches(1.5),
             "태양광 변동성 대비\n화력 백업 최적화 플랫폼", font_size=44, color=WHITE, bold=True)

# 서브라인
add_text_box(slide, Inches(1), Inches(4.3), Inches(11), Inches(0.8),
             "예측 → 대응 계획 → 실시간 보정까지 일관된 운영 지원", font_size=22, color=MID_GRAY)

# 하단 정보
add_text_box(slide, Inches(1), Inches(6.2), Inches(5), Inches(0.5),
             "PRD v4  |  2026.04", font_size=16, color=MID_GRAY)

# ============================================================
# SLIDE 2: 문제 정의 — 왜 필요한가
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "01  왜 이 플랫폼이 필요한가", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

# 3개 카드
add_card(slide, Inches(0.8), Inches(1.5), Inches(3.6), Inches(2.8),
         "태양광: 무조건 최대 발전", [
             "변동비 0원 → Merit Order 최우선",
             "나오는 만큼 다 쓰고, 전량 입찰",
             "발전사에게 선택의 여지 없음",
             "",
             "BUT: 내일 얼마 나올지 모른다",
             "예측 오차 10~30%, 급변 시 50%+",
         ], title_color=ACCENT_YELLOW)

add_card(slide, Inches(4.8), Inches(1.5), Inches(3.6), Inches(2.8),
         "화력: 이미 돌아가고 있다", [
             "최소출력 합계 ~6,387MW가 24시간 가동",
             "석탄 최소출력: 정격의 60%",
             "LNG 최소출력: 정격의 48%",
             "",
             "출력 정밀도: ±1~3% (정확함)",
             "BUT: 미리 준비해야 한다 (느림)",
         ], title_color=ACCENT_ORANGE)

add_card(slide, Inches(8.8), Inches(1.5), Inches(3.6), Inches(2.8),
         "충돌: 불확실 × 느림", [
             "태양광 급감 → 화력 올려야",
             "  → 석탄 램프업: 9~26MW/분",
             "  → Cold 기동: 12~24시간",
             "",
             "태양광 과잉 → 화력 내려야",
             "  → 최소출력 이하로 못 내림",
             "  → 끄면 재기동 수 시간",
         ], title_color=ACCENT_RED)

# 하단 핵심 메시지
add_shape(slide, Inches(0.8), Inches(4.8), Inches(11.7), Inches(2.0), RGBColor(0x00, 0x2B, 0x5C))
add_text_box(slide, Inches(1.3), Inches(4.95), Inches(10.7), Inches(0.5),
             "핵심 문제", font_size=20, color=ACCENT_BLUE, bold=True)
add_text_box(slide, Inches(1.3), Inches(5.4), Inches(10.7), Inches(1.2),
             "태양광은 \"내일 얼마 나올지 모르겠다\" (불확실)\n"
             "화력은 \"얼마를 내라고 하면 정확하게 내지만, 미리 말해줘야 한다\" (정확하지만 느림)\n\n"
             "→ 태양광 예측을 잘 해서, 화력에게 미리 정확한 지시를 내리는 것이 핵심",
             font_size=16, color=LIGHT_GRAY)

# ============================================================
# SLIDE 3: 현행 제도 & 벤치마크
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "02  현행 제도와 경쟁 환경", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

# CBP 설명
add_card(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(2.5),
         "한국 전력시장 (CBP) 입찰 프로세스", [
             "D-1일 10:00  1차 예측 & 가용용량 신고 제출",
             "D-1일 17:00  2차 예측 (최신 기상으로 보정)",
             "D-1일 저녁    KPX 급전계획 확정, 발전사 통보",
             "D일 실시간    급전지시에 따라 발전 수행",
             "",
             "※ 변동비 기반 → 가격 최적화 아닌 가용용량 최적화",
         ], title_color=ACCENT_BLUE)

add_card(slide, Inches(6.8), Inches(1.5), Inches(5.5), Inches(2.5),
         "재생에너지 발전량 예측제도 (2021.10~)", [
             "20MW 초과 태양광/풍력 → 예측 제출 의무",
             "D-1일 10시, 17시 2차에 걸쳐 제출",
             "기준오차율 충족 시 인센티브, 미충족 시 페널티",
             "",
             "→ 남동발전도 이미 예측을 하고 있음",
             "→ 하지만 구체적 시스템/정확도 미공개",
         ], title_color=ACCENT_GREEN)

# 벤치마크
add_shape(slide, Inches(0.8), Inches(4.5), Inches(11.7), Inches(2.4), CARD_BG)
add_text_box(slide, Inches(1.3), Inches(4.6), Inches(10), Inches(0.5),
             "경쟁 벤치마크", font_size=20, color=ACCENT_ORANGE, bold=True)

# 표 형태로
headers = ["발전사", "예측 시스템", "정확도", "화력 연계"]
data = [
    ["한국중부발전", "AI 기반 (제이케이코어 공동 개발)", "99% (2026.2 발표)", "2030년까지 통합 운영 목표"],
    ["한전 (KEPCO)", "AI 기반 재생에너지 예측 기술", "2km 격자, 1주일 예측", "개발 완료 (2021)"],
    ["한국남동발전", "미공개", "미공개", "본 플랫폼이 해결"],
]

y_start = Inches(5.15)
for col_i, header in enumerate(headers):
    x = Inches(1.3 + col_i * 2.7)
    add_text_box(slide, x, y_start, Inches(2.5), Inches(0.35),
                 header, font_size=13, color=ACCENT_BLUE, bold=True)

for row_i, row in enumerate(data):
    for col_i, cell in enumerate(row):
        x = Inches(1.3 + col_i * 2.7)
        y = y_start + Inches(0.4) + Inches(row_i * 0.4)
        c = ACCENT_YELLOW if row_i == 2 and col_i >= 2 else LIGHT_GRAY
        add_text_box(slide, x, y, Inches(2.5), Inches(0.35),
                     cell, font_size=12, color=c)

# ============================================================
# SLIDE 4: 3단계 운영 모드
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "03  플랫폼 3단계 운영 모드", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

# 3단계 카드
add_card(slide, Inches(0.8), Inches(1.5), Inches(3.6), Inches(4.5),
         "STEP 1: 사전 대비 (D-1일)", [
             "",
             "  D-1 10:00  1차 예측 & 대응 계획",
             "  D-1 17:00  2차 보정",
             "",
             "사용 기상 데이터:",
             "  LDAPS 수치예보 (하루 8회)",
             "  단기예보 (하루 8회)",
             "  초단기예보 (30분마다)",
             "",
             "산출물:",
             "  24시간 태양광 예측",
             "  화력 액션 플랜 (타임라인)",
             "  시나리오별 비교",
             "  가용용량 신고안",
         ], title_color=ACCENT_BLUE)

add_card(slide, Inches(4.8), Inches(1.5), Inches(3.6), Inches(4.5),
         "STEP 2: 실시간 감시 (D일)", [
             "",
             "  매 30분 예측 갱신",
             "  매 10분 위성 구름 확인",
             "",
             "사용 데이터:",
             "  초단기예보 (30분마다)",
             "  위성 영상 (10분마다)",
             "  발전소 기상 실측 (실시간)",
             "  태양광 발전 실측 (실시간)",
             "",
             "핵심:",
             "  예측 vs 실측 괴리 감지",
             "  향후 3~6시간 예측 지속 갱신",
         ], title_color=ACCENT_GREEN)

add_card(slide, Inches(8.8), Inches(1.5), Inches(3.6), Inches(4.5),
         "STEP 3: 실시간 대응 (이벤트)", [
             "",
             '  "현재 태양광이 예측보다',
             '   300MW 적게 나오고 있습니다"',
             "",
             "시스템 자동 실행:",
             "  최신 기상으로 향후 예측 재계산",
             "  화력 조정 지시 즉시 업데이트",
             "",
             "출력 예시:",
             '  "영흥 2호 지금 출력 증가"',
             '  "분당 1호 기동 준비 (45분)"',
             "",
             "→ 사후 대비가 핵심 가치",
         ], title_color=ACCENT_RED)

# 하단 화살표 연결
add_text_box(slide, Inches(0.8), Inches(6.3), Inches(11.7), Inches(0.8),
             "사전 계획으로 대부분 커버  →  실시간 감시로 이탈 탐지  →  즉각 대응으로 리스크 제거",
             font_size=18, color=MID_GRAY, align=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 5: 기상 데이터 전략
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "04  기상 데이터 전략: 기상청 오차 극복", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

add_text_box(slide, Inches(0.8), Inches(1.4), Inches(11), Inches(0.5),
             '"기상청이 틀린만큼 틀리는 것 아닌가?" → 기상청 데이터만 쓰면 그렇다. 하지만 다중 소스를 쓰면 극복 가능.',
             font_size=16, color=ACCENT_YELLOW)

# 데이터 계층 테이블
layers = [
    ("LDAPS 수치예보", "1.5km 격자", "하루 8회", "48시간", "하루전 예측의 주 입력"),
    ("단기예보 (동네예보)", "5km 격자", "하루 8회", "3일", "하루전 예측 보조"),
    ("초단기예보", "5km 격자", "30분마다", "6시간", "실시간 보정 핵심"),
    ("위성 영상 (천리안2A)", "2km", "10분마다", "실시간", "구름 급변 탐지"),
    ("발전소 현장 관측", "포인트", "실시간", "현재", "실측 기반 보정"),
    ("태양광 발전 실측", "인버터별", "실시간", "현재", "예측 vs 실측 비교"),
]

headers_met = ["데이터 소스", "해상도", "갱신 주기", "예측 범위", "용도"]
y = Inches(2.1)
for col_i, h in enumerate(headers_met):
    x = Inches(0.8 + col_i * 2.4)
    add_text_box(slide, x, y, Inches(2.2), Inches(0.4), h, font_size=14, color=ACCENT_BLUE, bold=True)

for row_i, row in enumerate(layers):
    bg_color = CARD_BG if row_i % 2 == 0 else DARK_BG
    add_rect(slide, Inches(0.7), y + Inches(0.45) + Inches(row_i * 0.5), Inches(12), Inches(0.48), bg_color)
    for col_i, cell in enumerate(row):
        x = Inches(0.8 + col_i * 2.4)
        yt = y + Inches(0.5) + Inches(row_i * 0.5)
        c = ACCENT_GREEN if "실시간" in cell or "30분" in cell or "10분" in cell else LIGHT_GRAY
        add_text_box(slide, x, yt, Inches(2.2), Inches(0.4), cell, font_size=13, color=c)

# 하단 핵심
add_shape(slide, Inches(0.8), Inches(5.5), Inches(11.7), Inches(1.5), RGBColor(0x00, 0x2B, 0x5C))
add_text_box(slide, Inches(1.3), Inches(5.65), Inches(10.7), Inches(1.2),
             "핵심: 기상청 수치예보(LDAPS)만 쓰면 구름 예측 오차 = 발전량 예측 오차\n"
             "→ 위성 구름 관측 + 실시간 실측 + 과거 패턴 보정을 더하면 기상청보다 정확할 수 있음\n"
             "→ 특히 2~3시간 이내 초단기 예측에서는 위성 기반이 NWP보다 우수",
             font_size=15, color=LIGHT_GRAY)

# ============================================================
# SLIDE 6: 태양광 예측 모델
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "05  태양광 예측 모델", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

# 모델 비교 카드
models = [
    ("통계 모델\n(회귀, ARIMA)", "nRMSE\n15~25%", ACCENT_RED),
    ("ML\n(XGBoost 등)", "nRMSE\n10~18%", ACCENT_ORANGE),
    ("DL\n(LSTM, GRU)", "nRMSE\n8~15%", ACCENT_YELLOW),
    ("베이지안+GRU\n(제안 모델)", "nRMSE\n7~12%", ACCENT_GREEN),
    ("위성 Nowcast\n(초단기)", "nRMSE\n5~10%", ACCENT_BLUE),
]

for i, (name, acc, color) in enumerate(models):
    x = Inches(0.8 + i * 2.4)
    card = add_shape(slide, x, Inches(1.5), Inches(2.1), Inches(2.2), CARD_BG)
    add_rect(slide, x, Inches(1.5), Inches(2.1), Inches(0.06), color)
    add_text_box(slide, x + Inches(0.15), Inches(1.7), Inches(1.8), Inches(0.8),
                 name, font_size=14, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text_box(slide, x + Inches(0.15), Inches(2.6), Inches(1.8), Inches(0.8),
                 acc, font_size=20, color=color, bold=True, align=PP_ALIGN.CENTER)

# MVP 전략
add_shape(slide, Inches(0.8), Inches(4.1), Inches(11.7), Inches(1.0), RGBColor(0x00, 0x2B, 0x5C))
add_text_box(slide, Inches(1.3), Inches(4.2), Inches(10.7), Inches(0.8),
             "MVP 전략: 검증된 ML(XGBoost/LSTM)로 시작 → 데이터 축적 후 베이지안+GRU 확장\n"
             "전제 조건: 남동발전 기존 예측보다 정확하지 않으면 가치 없음 → 기존 정확도 먼저 파악",
             font_size=16, color=ACCENT_YELLOW)

# 2단계 모델 구조
add_card(slide, Inches(0.8), Inches(5.4), Inches(5.5), Inches(1.7),
         "1단계: 베이지안 계층모형 (구조 추정)", [
             "지역별 기상 민감도, 시간/계절 구조를 posterior로 추정",
             "부분풀링 → 데이터 적은 사이트도 안정 추정",
             "출력: baseline 예측 + 잔차",
         ], title_color=ACCENT_BLUE)

add_card(slide, Inches(6.8), Inches(5.4), Inches(5.5), Inches(1.7),
         "2단계: FiLM-조건부 GRU (잔차 학습)", [
             "구름 급변, NWP 오차, ramp event 등 비선형 패턴",
             "지역별 posterior 요약을 FiLM 조건벡터로 주입",
             "Direct multi-horizon 출력 (1~24시간)",
         ], title_color=ACCENT_GREEN)

# ============================================================
# SLIDE 7: 화력 대응 계획 — 핵심 기능
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "06  화력 대응 계획 생성 (핵심 기능)", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

add_text_box(slide, Inches(0.8), Inches(1.3), Inches(11), Inches(0.5),
             "태양광 예측이 입력이고, 나오는 결과는 \"발전기별로 몇 시에 무슨 행동을 하라\"는 액션 플랜이다.",
             font_size=16, color=ACCENT_YELLOW)

# 흐름도
flow_steps = [
    ("태양광 예측", "내일 시간대별\n발전량 예측", ACCENT_YELLOW),
    ("갭 계산", "수요 - 태양광 - 고정화력\n= 부족/과잉", ACCENT_ORANGE),
    ("UC 최적화", "MILP로 24시간\n최적 조합 산출", ACCENT_BLUE),
    ("역산 타이밍", "발전기별\n\"몇시에 몇MW\"", ACCENT_GREEN),
    ("액션 플랜", "운영자용\n타임라인 출력", WHITE),
]

for i, (title, desc, color) in enumerate(flow_steps):
    x = Inches(0.6 + i * 2.5)
    card = add_shape(slide, x, Inches(1.9), Inches(2.2), Inches(1.8), CARD_BG)
    add_rect(slide, x, Inches(1.9), Inches(2.2), Inches(0.05), color)
    add_text_box(slide, x + Inches(0.1), Inches(2.05), Inches(2.0), Inches(0.4),
                 title, font_size=16, color=color, bold=True, align=PP_ALIGN.CENTER)
    add_text_box(slide, x + Inches(0.1), Inches(2.5), Inches(2.0), Inches(1.0),
                 desc, font_size=13, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)
    if i < 4:
        add_text_box(slide, x + Inches(2.2), Inches(2.4), Inches(0.3), Inches(0.5),
                     "→", font_size=24, color=MID_GRAY, align=PP_ALIGN.CENTER)

# 액션 플랜 예시
add_shape(slide, Inches(0.8), Inches(4.1), Inches(11.7), Inches(3.0), CARD_BG)
add_text_box(slide, Inches(1.2), Inches(4.2), Inches(4), Inches(0.4),
             "액션 플랜 출력 예시", font_size=18, color=ACCENT_GREEN, bold=True)

timeline_data = [
    ("12:37", "영흥 1호", "출력 증가 시작", "520 → 700MW"),
    ("13:00", "영흥 1호", "700MW 도달", "700MW"),
    ("13:30", "영흥 4호", "출력 증가 시작", "520 → 870MW"),
    ("14:00", "분당 1호", "기동 준비 시작", "(정지 → 기동)"),
    ("14:23", "영흥 4호", "870MW 도달", "870MW (최대)"),
    ("14:45", "분당 1호", "최소출력 도달", "221MW"),
    ("15:06", "분당 1호", "400MW 도달", "400MW"),
]

cols = ["시간", "발전기", "액션", "목표 출력"]
for ci, c in enumerate(cols):
    x = Inches(1.2 + ci * 2.7)
    add_text_box(slide, x, Inches(4.65), Inches(2.5), Inches(0.3),
                 c, font_size=12, color=ACCENT_BLUE, bold=True)

for ri, row in enumerate(timeline_data):
    for ci, cell in enumerate(row):
        x = Inches(1.2 + ci * 2.7)
        y = Inches(4.95 + ri * 0.3)
        c = ACCENT_GREEN if "도달" in cell else ACCENT_YELLOW if "시작" in cell else LIGHT_GRAY
        add_text_box(slide, x, y, Inches(2.5), Inches(0.3), cell, font_size=11, color=c)

# ============================================================
# SLIDE 8: 시나리오 비교
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "07  시나리오 비교 — 운영자 의사결정 지원", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

# 시나리오 테이블
scen_headers = ["시나리오", "태양광 가정", "추가 기동", "연료비", "예비력", "대응 실패"]
scen_data = [
    ["낙관", "예측 +10%", "없음", "기본", "충분", "없음"],
    ["중립", "예측 그대로", "영흥1호 증가", "+2.1억", "충분", "없음"],
    ["비관", "예측 -30%", "영흥1,4호+분당", "+5.8억", "주의", "없음"],
    ["급변", "14시 50% 감소", "분당 즉시 기동", "+7.2억", "위험", "지연 가능"],
]
scen_colors = [ACCENT_GREEN, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_RED]

y = Inches(1.6)
for ci, h in enumerate(scen_headers):
    x = Inches(0.8 + ci * 2.0)
    add_text_box(slide, x, y, Inches(1.9), Inches(0.4), h, font_size=14, color=ACCENT_BLUE, bold=True)

for ri, row in enumerate(scen_data):
    bg = CARD_BG if ri % 2 == 0 else DARK_BG
    add_rect(slide, Inches(0.7), y + Inches(0.45 + ri * 0.6), Inches(12), Inches(0.55), bg)
    for ci, cell in enumerate(row):
        x = Inches(0.8 + ci * 2.0)
        yt = y + Inches(0.5 + ri * 0.6)
        c = scen_colors[ri] if ci == 0 else (ACCENT_RED if "위험" in cell or "지연" in cell else LIGHT_GRAY)
        add_text_box(slide, x, yt, Inches(1.9), Inches(0.4), cell, font_size=13, color=c)

# 의사결정 포인트
add_shape(slide, Inches(0.8), Inches(4.5), Inches(11.7), Inches(2.5), RGBColor(0x00, 0x2B, 0x5C))
add_text_box(slide, Inches(1.3), Inches(4.65), Inches(10.7), Inches(0.4),
             "운영자 의사결정 포인트", font_size=20, color=ACCENT_ORANGE, bold=True)
add_text_box(slide, Inches(1.3), Inches(5.1), Inches(10.7), Inches(1.5),
             "\"비관 시나리오 대비 시 연료비 3.7억 더 들지만, 급전 대응 실패 리스크 0.\n"
             " 오늘 날씨가 불확실하니 비관으로 간다.\"\n\n"
             "→ 시스템은 추천하고, 사람이 최종 판단한다.\n"
             "→ 판단 근거(비용/예비력/ESG)를 한 화면에서 비교할 수 있게 하는 것이 대시보드의 가치.",
             font_size=16, color=LIGHT_GRAY)

# ============================================================
# SLIDE 9: 화력 운전 제약 파라미터
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "08  화력 발전기 운전 제약 파라미터", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

# 파라미터 테이블
param_headers = ["파라미터", "영흥 (석탄)", "삼천포 (석탄)", "분당 (LNG)", "의미"]
param_data = [
    ["정격출력", "870MW/호기", "560MW/호기", "460MW", "최대 출력"],
    ["최소출력", "520MW (60%)", "336MW (60%)", "221MW (48%)", "이하로 못 내림"],
    ["램프업", "~15MW/분", "~10MW/분", "~30MW/분", "분당 증가 한계"],
    ["최소기동시간", "12시간", "12시간", "4시간", "켜면 최소 유지"],
    ["최소정지시간", "8시간", "8시간", "2시간", "끄면 최소 유지"],
    ["Cold 기동", "16시간", "16시간", "6시간", "완전 정지 후 재가동"],
    ["기동비용", "~3억원/회", "~2억원/회", "~5천만원/회", "1회 기동 비용"],
    ["변동비", "~80원/kWh", "~80원/kWh", "~160원/kWh", "연료비 기반"],
]

y = Inches(1.5)
for ci, h in enumerate(param_headers):
    x = Inches(0.6 + ci * 2.5)
    add_text_box(slide, x, y, Inches(2.3), Inches(0.4), h, font_size=14, color=ACCENT_BLUE, bold=True)

for ri, row in enumerate(param_data):
    bg = CARD_BG if ri % 2 == 0 else DARK_BG
    add_rect(slide, Inches(0.5), y + Inches(0.4 + ri * 0.48), Inches(12.3), Inches(0.45), bg)
    for ci, cell in enumerate(row):
        x = Inches(0.6 + ci * 2.5)
        yt = y + Inches(0.44 + ri * 0.48)
        c = ACCENT_ORANGE if ci == 3 else LIGHT_GRAY
        add_text_box(slide, x, yt, Inches(2.3), Inches(0.4), cell, font_size=12, color=c)

# 핵심 trade-off
add_shape(slide, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.2), RGBColor(0x00, 0x2B, 0x5C))
add_text_box(slide, Inches(1.3), Inches(5.9), Inches(10.7), Inches(1.0),
             "핵심 Trade-off: 석탄 = 싸지만 느리다 (램프 15MW/분, Cold 16시간)  vs  LNG = 비싸지만 빠르다 (램프 30MW/분, Cold 6시간)\n"
             "→ 태양광 급변 대비에 LNG가 유리하지만 비용 2배. 이 균형을 최적화하는 것이 UC의 역할.",
             font_size=15, color=LIGHT_GRAY)

# ============================================================
# SLIDE 10: 남동발전 포트폴리오
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "09  남동발전 발전 포트폴리오", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

# 세 영역 시각화
add_card(slide, Inches(0.8), Inches(1.5), Inches(3.6), Inches(3.5),
         "고정 영역 (건드릴 수 없음)", [
             "",
             "필수운전 화력 + 최소출력 구간",
             "합계: ~6,387MW",
             "",
             "영흥: 최소 3,048MW",
             "삼천포: 최소 1,248MW",
             "강릉: 최소 1,248MW",
             "여수: 최소 401MW",
             "분당: 최소 442MW",
         ], title_color=MID_GRAY)

add_card(slide, Inches(4.8), Inches(1.5), Inches(3.6), Inches(3.5),
         "예측 영역 (변동, 예측 대상)", [
             "",
             "태양광: 20+ 사이트, 수백MW",
             "  → 변동성 큼, 예측이 핵심",
             "",
             "풍력: 영흥, 군위 등",
             "  → 변동성 큼",
             "",
             "연료전지: 여수, 분당, 안산",
             "  → 비교적 안정적",
             "",
             "해양소수력: 소규모, 안정적",
         ], title_color=ACCENT_YELLOW)

add_card(slide, Inches(8.8), Inches(1.5), Inches(3.6), Inches(3.5),
         "최적화 영역 (조정 가능)", [
             "",
             "화력의 최소출력~최대출력 사이",
             "조정 가능 범위: ~4,442MW",
             "",
             "영흥: 3,048 → 5,080 (+2,032)",
             "삼천포: 1,248 → 2,080 (+832)",
             "분당: 442 → 920 (+478)",
             "",
             "대기 중 발전기의 기동/정지",
             "→ 이 영역이 최적화 대상",
         ], title_color=ACCENT_GREEN)

# 수식
add_shape(slide, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.5), RGBColor(0x00, 0x2B, 0x5C))
add_text_box(slide, Inches(1.3), Inches(5.5), Inches(10.7), Inches(1.0),
             "전체 공급 = 고정 영역(~6,387MW) + 태양광 예측(변동) + 최적화 영역(0~4,442MW)\n\n"
             "태양광이 많이 나오면 → 최적화 영역을 줄임 (연료비 절감)\n"
             "태양광이 적게 나오면 → 최적화 영역을 올림 (수요 충족)",
             font_size=15, color=LIGHT_GRAY)

# ============================================================
# SLIDE 11: ESG
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "10  ESG: 하이퍼로컬 대기질 영향 예측", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

add_card(slide, Inches(0.8), Inches(1.5), Inches(3.6), Inches(3.0),
         "배출량 추정", [
             "",
             "화력 발전기별 출력",
             "  × 배출계수 (SOx, NOx, PM)",
             "  = 시간대별 배출량",
             "",
             "데이터: 남동발전",
             "  대기오염물질 배출실적 API",
         ], title_color=ACCENT_ORANGE)

add_card(slide, Inches(4.8), Inches(1.5), Inches(3.6), Inches(3.0),
         "확산 모델", [
             "",
             "풍속, 풍향, 대기안정도",
             "  → 가우시안 확산 모델",
             "  → 영향권 계산",
             "",
             "데이터: 남동발전",
             "  발전소 기상정보 API (풍향/풍속)",
         ], title_color=ACCENT_BLUE)

add_card(slide, Inches(8.8), Inches(1.5), Inches(3.6), Inches(3.0),
         "지도 시각화", [
             "",
             "발전소 중심 영향권 오버레이",
             "민감지역 표시 (주거지, 학교)",
             "",
             "시나리오 A vs B 비교:",
             "  \"이 안이 환경적으로 더 나아요\"",
             "",
             "ESG 리스크 경고",
         ], title_color=ACCENT_GREEN)

# 하단
add_shape(slide, Inches(0.8), Inches(5.0), Inches(11.7), Inches(1.8), RGBColor(0x00, 0x2B, 0x5C))
add_text_box(slide, Inches(1.3), Inches(5.15), Inches(10.7), Inches(1.5),
             "가치: 화력 가동을 늘릴 때의 환경 영향을 사전에 정량화\n\n"
             "운영자가 \"석탄을 더 돌릴까, LNG를 돌릴까\" 판단할 때,\n"
             "비용뿐 아니라 \"영흥 주변 주거지역 NOx +12% vs 분당 주변 +3%\"를 함께 보여줌\n\n"
             "→ ESG 보고서에 정량적 근거 제공",
             font_size=15, color=LIGHT_GRAY)

# ============================================================
# SLIDE 12: MVP & 로드맵
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "11  MVP 범위 & 로드맵", font_size=32, color=WHITE, bold=True)
add_rect(slide, Inches(0.8), Inches(1.1), Inches(2), Inches(0.05), ACCENT_BLUE)

add_card(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(3.5),
         "MVP (Phase 1)", [
             "",
             "1. 태양광 하루전 예측 (ML 모델)",
             "2. 화력 출력 여유 분석",
             "3. 태양광+화력 겹쳐보기 시각화",
             "4. UC 기반 대응 계획 (액션 플랜 타임라인)",
             "5. 시나리오 비교 (낙관/중립/비관)",
             "6. 가용용량 신고안 출력",
             "7. 기본 운영 대시보드",
             "8. 대기질 영향 지도 (기본)",
         ], title_color=ACCENT_GREEN)

add_card(slide, Inches(6.8), Inches(1.5), Inches(5.5), Inches(3.5),
         "Phase 2 (MVP 이후)", [
             "",
             "1. 실시간 감시 & 대응 모드",
             "2. 위성 기반 초단기 예측 (Nowcasting)",
             "3. 베이지안+GRU 고급 예측 모델",
             "4. 풍력 정밀 예측",
             "5. ESS 충방전 최적화 연계",
             "",
             "제외:",
             "  KPX 자동 제출 / 송전망 제약",
             "  발전기 자동 제어 (사람이 최종 판단)",
         ], title_color=ACCENT_BLUE)

# 성공 지표
add_shape(slide, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.8), CARD_BG)
add_text_box(slide, Inches(1.3), Inches(5.4), Inches(10.7), Inches(0.4),
             "핵심 성공 지표", font_size=18, color=ACCENT_ORANGE, bold=True)

metrics = [
    ("예측 정확도", "nRMSE 10% 이내\n(기존 대비 개선)", ACCENT_YELLOW),
    ("급변 탐지율", "80% 이상\n사전 탐지", ACCENT_GREEN),
    ("대응 실패", "0건\n급전 실패 없음", ACCENT_RED),
    ("연료비 절감", "불필요 가동\n감소", ACCENT_BLUE),
]
for i, (title, value, color) in enumerate(metrics):
    x = Inches(1.3 + i * 2.7)
    add_text_box(slide, x, Inches(5.85), Inches(2.5), Inches(0.3),
                 title, font_size=14, color=color, bold=True, align=PP_ALIGN.CENTER)
    add_text_box(slide, x, Inches(6.15), Inches(2.5), Inches(0.6),
                 value, font_size=13, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 13: 마무리
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.08), ACCENT_BLUE)

add_text_box(slide, Inches(1), Inches(2.0), Inches(11), Inches(0.5),
             "One-line Summary", font_size=20, color=ACCENT_BLUE)

add_text_box(slide, Inches(1), Inches(2.6), Inches(11), Inches(2.0),
             "태양광 변동성을 다중 기상 데이터로 정밀 예측하고,\n"
             "이미 가동 중인 화력 포트폴리오에 대해\n"
             "\"언제 어떤 발전기를 얼마나 올리거나 내릴지\"\n"
             "액션 플랜을 생성하는 플랫폼",
             font_size=32, color=WHITE, bold=True)

add_text_box(slide, Inches(1), Inches(5.0), Inches(11), Inches(1.0),
             "사전 대비 → 실시간 감시 → 실시간 대응\n"
             "예측부터 행동까지, 일관된 운영 지원",
             font_size=22, color=MID_GRAY)

# 저장
output_path = os.path.join("C:/Energy_effi", "PRD_v4_presentation.pptx")
prs.save(output_path)
print(f"PPT 저장 완료: {output_path}")
