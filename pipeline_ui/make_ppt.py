"""
발표용 PPT 자동 생성 스크립트.

usage:
    python make_ppt.py            # 기본 출력 → 발표자료_파이프라인.pptx
    python make_ppt.py output.pptx
"""
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# Screenshots / charts directories
SCREENS = Path(__file__).parent / "screens"
CHARTS = Path(__file__).parent / "charts"


def has_chart(name: str) -> bool:
    return (CHARTS / name).exists()


def add_chart(slide, name: str, x, y, w, h, *, caption=None):
    path = CHARTS / name
    if not path.exists():
        return None
    pic = slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    if caption:
        cap = slide.shapes.add_textbox(x, y + h + Inches(0.05), w, Inches(0.25))
        cap.text_frame.text = caption
        p = cap.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        for run in p.runs:
            run.font.name = FONT
            run.font.size = Pt(9)
            run.font.italic = True
            run.font.color.rgb = GRAY
    return pic


def has_screen(name: str) -> bool:
    return (SCREENS / name).exists()


def add_screenshot(slide, name: str, x, y, w, h, *, caption=None, border=None):
    """스크린샷 이미지 + 테두리 + 캡션 한꺼번에 배치."""
    path = SCREENS / name
    if not path.exists():
        return None
    # border
    if border is not None:
        b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
        b.fill.solid(); b.fill.fore_color.rgb = WHITE
        b.line.color.rgb = border; b.line.width = Pt(1.5)
        b.shadow.inherit = False
    pic = slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    if caption:
        cap = slide.shapes.add_textbox(x, y + h + Inches(0.05), w, Inches(0.25))
        cap.text_frame.text = caption
        p = cap.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        for run in p.runs:
            run.font.name = FONT
            run.font.size = Pt(9)
            run.font.italic = True
            run.font.color.rgb = GRAY
    return pic

# Colors
NAVY = RGBColor(0x0B, 0x1A, 0x2E)
TEAL = RGBColor(0x17, 0xBE, 0xBB)
TEAL_DK = RGBColor(0x0E, 0x7C, 0x7B)
AMBER = RGBColor(0xFF, 0xC1, 0x07)
ORANGE = RGBColor(0xFF, 0x6B, 0x35)
RED = RGBColor(0xD6, 0x30, 0x31)
GRAY = RGBColor(0x5A, 0x69, 0x76)
GRAY_LT = RGBColor(0xEE, 0xF2, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG_CARD = RGBColor(0xFA, 0xFC, 0xFD)

FONT = "맑은 고딕"

# 16:9
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def new_prs():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def add_blank_slide(prs):
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    # white background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid(); bg.fill.fore_color.rgb = WHITE
    bg.line.fill.background()
    bg.shadow.inherit = False
    return slide


def add_text(slide, x, y, w, h, text, *, size=18, bold=False, color=NAVY,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return tb


def add_rect(slide, x, y, w, h, *, fill=WHITE, line=None, line_w=0.75, shadow=False):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.adjustments[0] = 0.06
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_w)
    if not shadow:
        sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def add_header_bar(slide, page_num: int, title: str, subtitle: str = ""):
    """페이지 상단 색띠 + 제목."""
    bar_h = Inches(0.55)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, bar_h)
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL_DK
    bar.line.fill.background()
    bar.shadow.inherit = False

    # 작은 amber 띠
    sub_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, bar_h, Inches(1.2), Inches(0.08))
    sub_bar.fill.solid(); sub_bar.fill.fore_color.rgb = AMBER
    sub_bar.line.fill.background(); sub_bar.shadow.inherit = False

    # 페이지 번호
    add_text(slide, Inches(0.4), Inches(0.05), Inches(1.0), Inches(0.45),
             f"{page_num:02d}", size=22, bold=True, color=WHITE,
             anchor=MSO_ANCHOR.MIDDLE)
    # 제목
    add_text(slide, Inches(1.1), Inches(0.05), Inches(11.0), Inches(0.45),
             title, size=20, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        add_text(slide, Inches(0.4), Inches(0.7), Inches(12.0), Inches(0.35),
                 subtitle, size=11, color=GRAY, anchor=MSO_ANCHOR.TOP)


def add_footer(slide, text="IITP 초소형 위성영상 기반 주요 지역 분석  ·  2세부 · 이노뎁"):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(7.15),
                                  Inches(12.5), Pt(0.5))
    line.fill.solid(); line.fill.fore_color.rgb = GRAY_LT
    line.line.fill.background(); line.shadow.inherit = False
    add_text(slide, Inches(0.4), Inches(7.2), Inches(12.5), Inches(0.3),
             text, size=9, color=GRAY)


def add_bullets(slide, x, y, w, h, items, *, size=14, color=NAVY, bullet="•", spacing=8):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(0)
        p.space_after = Pt(spacing)
        # bullet
        r1 = p.add_run()
        r1.text = f"{bullet}  "
        r1.font.name = FONT
        r1.font.size = Pt(size)
        r1.font.bold = True
        r1.font.color.rgb = TEAL
        # text
        r2 = p.add_run()
        r2.text = item
        r2.font.name = FONT
        r2.font.size = Pt(size)
        r2.font.color.rgb = color


# ========== Slide 1: Title ==========
def slide_title(prs):
    slide = add_blank_slide(prs)
    # 배경 풀그라데이션 대신 좌측 컬러바
    side = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(4.5), SLIDE_H)
    side.fill.solid(); side.fill.fore_color.rgb = NAVY
    side.line.fill.background(); side.shadow.inherit = False

    # 좌측 사이드 아이콘 영역
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.6), Inches(0.15), Inches(2.3))
    accent.fill.solid(); accent.fill.fore_color.rgb = AMBER
    accent.line.fill.background(); accent.shadow.inherit = False

    add_text(slide, Inches(0.5), Inches(0.8), Inches(3.5), Inches(0.4),
             "IITP", size=14, bold=True, color=AMBER)
    add_text(slide, Inches(0.5), Inches(1.15), Inches(3.5), Inches(0.4),
             "Innovative Talent Program", size=11, color=WHITE)
    add_text(slide, Inches(0.5), Inches(2.7), Inches(3.5), Inches(0.4),
             "PROJECT DEMO  ·  v0.1", size=10, color=TEAL)
    add_text(slide, Inches(0.5), Inches(3.1), Inches(3.5), Inches(2.0),
             "초소형 위성영상\n기반 수체 시계열\n분석 파이프라인",
             size=28, bold=True, color=WHITE)
    add_text(slide, Inches(0.5), Inches(6.5), Inches(3.5), Inches(0.5),
             "2026.  ·  이노뎁", size=11, color=TEAL)

    # 우측 영역 - 주요 키워드
    add_text(slide, Inches(5.2), Inches(1.1), Inches(7.5), Inches(0.5),
             "4단계 시스템 (공동) +  ★ ConvLSTM 시계열 예측 (이노뎁) + 기상청 API",
             size=12, color=GRAY, bold=True)
    add_text(slide, Inches(5.2), Inches(2.0), Inches(7.5), Inches(1.2),
             "관측–예측–대응을 잇는\nEnd-to-End 파이프라인",
             size=30, bold=True, color=NAVY)

    # 우측 박스 5개 - 4단계 공동 시스템 + 이노뎁 시계열 예측
    steps = [
        ("01", "전처리", TEAL, "공동"),
        ("02", "수체 탐지", TEAL, "공동"),
        ("03", "수위·면적", AMBER, "공동"),
        ("04", "퓨전 보정", ORANGE, "공동"),
        ("05", "★ 시계열 예측", RED, "이노뎁"),
    ]
    box_w = Inches(1.4); box_h = Inches(1.7); gap = Inches(0.1)
    start_x = Inches(5.2)
    for i, (num, label, color, who) in enumerate(steps):
        x = start_x + (box_w + gap) * i
        box = add_rect(slide, x, Inches(4.2), box_w, box_h, fill=WHITE, line=color, line_w=2)
        add_text(slide, x, Inches(4.3), box_w, Inches(0.5),
                 num, size=20, bold=True, color=color, align=PP_ALIGN.CENTER)
        add_text(slide, x, Inches(4.85), box_w, Inches(0.5),
                 label, size=10, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        # 담당 표시
        is_innodep = who == "이노뎁"
        tag_color = RED if is_innodep else GRAY
        add_text(slide, x, Inches(5.45), box_w, Inches(0.35),
                 who, size=9, bold=is_innodep, color=tag_color, align=PP_ALIGN.CENTER)

    add_text(slide, Inches(5.2), Inches(6.4), Inches(7.5), Inches(0.4),
             "발표일자: 2026.06.  ·  발표자: 이노뎁", size=10, color=GRAY)
    return slide


# ========== Slide 2: 과제 개요 ==========
def slide_overview(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 2, "과제 개요 — 2세부 이노뎁의 역할", "")
    # 좌측: 컨소시엄
    add_text(slide, Inches(0.5), Inches(1.1), Inches(6), Inches(0.4),
             "■ 컨소시엄 구성", size=14, bold=True, color=TEAL_DK)
    cons = [
        ("1세부", "수자원공사 / 인하대", "위성 데이터 수집 · 초해상화 · 시각화 · 플랫폼"),
        ("2세부", "이노뎁 / 서울시립대", "수체·수위 탐지 · 시계열 분석 · 특수목적 적용"),
        ("3세부", "세명소프트 / 스페이스디자인", "3D 시각화 · 재난재해 4종 시뮬레이션"),
    ]
    y = Inches(1.55)
    for tag, org, desc in cons:
        box = add_rect(slide, Inches(0.5), y, Inches(6), Inches(0.85), fill=BG_CARD, line=GRAY_LT)
        add_text(slide, Inches(0.7), y + Inches(0.1), Inches(1.1), Inches(0.3),
                 tag, size=11, bold=True, color=WHITE)
        tag_bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                        Inches(0.65), y + Inches(0.1), Inches(0.9), Inches(0.32))
        tag_bg.adjustments[0] = 0.3
        tag_bg.fill.solid(); tag_bg.fill.fore_color.rgb = TEAL
        tag_bg.line.fill.background(); tag_bg.shadow.inherit = False
        add_text(slide, Inches(0.65), y + Inches(0.1), Inches(0.9), Inches(0.32),
                 tag, size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)
        add_text(slide, Inches(1.7), y + Inches(0.07), Inches(4.7), Inches(0.35),
                 org, size=13, bold=True, color=NAVY)
        add_text(slide, Inches(1.7), y + Inches(0.42), Inches(4.7), Inches(0.4),
                 desc, size=10, color=GRAY)
        y += Inches(1.0)

    # 우측: 이노뎁 잔여 업무
    add_text(slide, Inches(7), Inches(1.1), Inches(6), Inches(0.4),
             "■ 이노뎁 잔여 업무 (3차년도)", size=14, bold=True, color=TEAL_DK)
    items = [
        ("1.  시계열 분석/예측 기술 개발",
         "ConvLSTM 모델 초도구현 완료 · 검증 / 3세부 연계 · TTA 평가 예정", AMBER),
        ("2.  재난재해 시뮬레이션 연동",
         "강우·홍수·가뭄·침수 4종 · API 개발 · 3세부 데이터 전송", ORANGE),
        ("3.  작물 분류 기술 개발",
         "데이터셋 확보 · Segmentation 모델 · POC (TTA 없음)", TEAL),
    ]
    y = Inches(1.55)
    for title, desc, color in items:
        box = add_rect(slide, Inches(7), y, Inches(6), Inches(1.4), fill=BG_CARD, line=color, line_w=1.5)
        # 좌측 컬러바
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7), y, Inches(0.15), Inches(1.4))
        bar.fill.solid(); bar.fill.fore_color.rgb = color
        bar.line.fill.background(); bar.shadow.inherit = False
        add_text(slide, Inches(7.3), y + Inches(0.15), Inches(5.5), Inches(0.4),
                 title, size=14, bold=True, color=NAVY)
        add_text(slide, Inches(7.3), y + Inches(0.6), Inches(5.5), Inches(0.7),
                 desc, size=10, color=GRAY)
        y += Inches(1.6)

    add_footer(slide)


# ========== Slide 3: 파이프라인 전체 흐름 ==========
def slide_pipeline(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 3, "전체 파이프라인 흐름", "1세부 → 2세부 (이노뎁) → 3세부 연계")

    # 5-step horizontal flow — 4단계 공동 시스템 + 5번째 이노뎁 시계열 예측
    steps = [
        ("01", "전처리", "preprocess.py\nICEYE/PlanetScope",
         "Processed_*.tif", TEAL_DK, "공동"),
        ("02", "수체 탐지", "detect_water.py\nU-Net 추론",
         "WB_*.tif (0/1/255)", TEAL, "공동"),
        ("03", "수위·면적", "calc_wlwa.py\nAOI + DEM",
         "WLWA_*.csv", AMBER, "공동"),
        ("04", "퓨전 보정", "Correct.py\nSAR+광학+AWS",
         "CWLWA_*.csv", ORANGE, "공동"),
        ("05", "★ 시계열 예측", "ConvLSTM\n+ 기상청 API",
         "미래 N프레임", RED, "★이노뎁"),
    ]
    y_box = Inches(1.5)
    box_w = Inches(2.4); box_h = Inches(3.1); gap = Inches(0.15)
    start_x = Inches(0.4)
    for i, (num, name, detail, badge, color, who) in enumerate(steps):
        x = start_x + (box_w + gap) * i
        # box
        b = add_rect(slide, x, y_box, box_w, box_h, fill=WHITE, line=color, line_w=2)
        # number circle
        circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.2), y_box - Inches(0.3),
                                        Inches(0.7), Inches(0.7))
        circle.fill.solid(); circle.fill.fore_color.rgb = color
        circle.line.fill.background(); circle.shadow.inherit = False
        add_text(slide, x + Inches(0.2), y_box - Inches(0.3), Inches(0.7), Inches(0.7),
                 num, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # badge (산출물)
        bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    x + box_w - Inches(1.4), y_box + Inches(0.3),
                                    Inches(1.2), Inches(0.32))
        bg.adjustments[0] = 0.4
        bg.fill.solid(); bg.fill.fore_color.rgb = GRAY_LT
        bg.line.fill.background(); bg.shadow.inherit = False
        add_text(slide, x + box_w - Inches(1.4), y_box + Inches(0.3), Inches(1.2), Inches(0.32),
                 badge, size=9, bold=True, color=GRAY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # 담당 표시 (공동/이노뎁)
        is_innodep = "이노뎁" in who
        who_color = RED if is_innodep else TEAL
        who_bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                        x + Inches(0.2), y_box + Inches(0.3),
                                        Inches(0.95), Inches(0.32))
        who_bg.adjustments[0] = 0.4
        who_bg.fill.solid(); who_bg.fill.fore_color.rgb = who_color
        who_bg.line.fill.background(); who_bg.shadow.inherit = False
        add_text(slide, x + Inches(0.2), y_box + Inches(0.3), Inches(0.95), Inches(0.32),
                 who, size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # name
        add_text(slide, x + Inches(0.2), y_box + Inches(1.0), box_w - Inches(0.4), Inches(0.5),
                 name, size=16, bold=True, color=NAVY)
        # detail
        add_text(slide, x + Inches(0.2), y_box + Inches(1.7), box_w - Inches(0.4), Inches(1.2),
                 detail, size=10, color=GRAY)

        # arrow
        if i < len(steps) - 1:
            ax = x + box_w + Inches(0.02)
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, ax, y_box + Inches(1.3),
                                           Inches(0.2), Inches(0.5))
            arrow.fill.solid(); arrow.fill.fore_color.rgb = color
            arrow.line.fill.background(); arrow.shadow.inherit = False

    # 하단: 입력/출력
    add_text(slide, Inches(0.6), Inches(5.0), Inches(12.5), Inches(0.35),
             "▼ 데이터 인터페이스", size=12, bold=True, color=TEAL_DK)

    io_box_y = Inches(5.45)
    io_items = [
        ("4단계 산출", "WLWA / CWLWA CSV\n(수위·면적 시계열)", TEAL_DK),
        ("이노뎁 입력", "수체/수위 시계열\n+ 기상청 API (강수·기온·습도)", AMBER),
        ("이노뎁 출력", "미래 N프레임 수체 예측\n+ 변화율·위험 평가", RED),
        ("3세부 전달", "GeoTIFF + REST API\n홍수/가뭄 시뮬레이션", ORANGE),
    ]
    for i, (label, content, color) in enumerate(io_items):
        x = Inches(0.6) + (Inches(2.95) + Inches(0.2)) * i
        b = add_rect(slide, x, io_box_y, Inches(2.95), Inches(1.3), fill=BG_CARD, line=color, line_w=1)
        add_text(slide, x + Inches(0.2), io_box_y + Inches(0.12), Inches(2.5), Inches(0.3),
                 label, size=11, bold=True, color=color)
        add_text(slide, x + Inches(0.2), io_box_y + Inches(0.45), Inches(2.7), Inches(0.8),
                 content, size=10, color=NAVY)

    add_footer(slide)


# ========== Slide 4: Step 1 — 전처리 ==========
def slide_step1(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 4, "STEP 1 · 전처리 (preprocess.py)",
                   "ICEYE / PlanetScope 원본 → 정규화·정사보정 → Processed_*.tif + meta_*.xml")
    # 좌측 설명
    add_text(slide, Inches(0.5), Inches(1.2), Inches(6), Inches(0.4),
             "■ 입력 데이터 사양", size=14, bold=True, color=TEAL_DK)

    table_data = [
        ("위성/센서", "ICEYE (SAR) · PlanetScope (광학)"),
        ("해상도", "SAR 0.5×1.5m · 광학 3m"),
        ("처리 흐름 (SAR)", "Multi-Look → Filter → Decibel → Geocoding"),
        ("처리 흐름 (광학)", "DN → BOA → Min-Max 정규화"),
        ("좌표계", "EPSG:32652 (UTM Zone 52N)"),
        ("출력", "Processed_*.tif + meta_*.xml"),
        ("실행", "docker exec wbms python3 preprocess.py"),
    ]
    y = Inches(1.65)
    for k, v in table_data:
        add_rect(slide, Inches(0.5), y, Inches(6), Inches(0.45),
                 fill=BG_CARD, line=GRAY_LT)
        add_text(slide, Inches(0.7), y + Inches(0.08), Inches(1.8), Inches(0.3),
                 k, size=11, bold=True, color=TEAL_DK)
        add_text(slide, Inches(2.5), y + Inches(0.08), Inches(4), Inches(0.3),
                 v, size=11, color=NAVY)
        y += Inches(0.5)

    # 우측: 데모 화면 캡처 (실제 데모 UI는 시각적 참고)
    add_text(slide, Inches(7), Inches(1.2), Inches(6), Inches(0.4),
             "■ 데모 UI 캡처 · 정사보정 결과 확인",
             size=14, bold=True, color=TEAL_DK)
    if has_screen("01_step1_input.png"):
        add_screenshot(slide, "01_step1_input.png",
                       Inches(7), Inches(1.65), Inches(5.9), Inches(3.7),
                       caption="📷 부산 낙동강 하구 · ICEYE Stripmap 정사보정 영상")
    else:
        img_box = add_rect(slide, Inches(7), Inches(1.65), Inches(5.9), Inches(3.7),
                           fill=NAVY, line=TEAL, line_w=2)

    add_text(slide, Inches(7), Inches(5.6), Inches(6), Inches(0.4),
             "■ 주요 파라미터 (CLI)",
             size=14, bold=True, color=TEAL_DK)
    add_bullets(slide, Inches(7), Inches(6.0), Inches(5.9), Inches(1.1),
                ["--sensor_type SAR | OPTIC",
                 "--sensor_name ICEYE | PlanetScope",
                 "--testbed Busan  --image_date YYYYMMDD"],
                size=10, spacing=3)
    add_footer(slide)


# ========== Slide 5: Step 2 — 수체 탐지 ==========
def slide_step2(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 5, "STEP 2 · 수체 탐지 (detect_water.py)",
                   "U-Net 추론 · WB_*.tif (0=비수체 / 1=수체 / 255=nodata) + 메타 XML")
    # 좌측: 모델
    add_text(slide, Inches(0.5), Inches(1.2), Inches(6), Inches(0.4),
             "■ 탐지 모델 (U-Net) · 센서 공용 구조",
             size=14, bold=True, color=TEAL_DK)
    model_box = add_rect(slide, Inches(0.5), Inches(1.7), Inches(6), Inches(2.7),
                         fill=BG_CARD, line=TEAL, line_w=1.5)
    add_bullets(slide, Inches(0.7), Inches(1.9), Inches(5.7), Inches(2.5),
                ["입력: Processed_*.tif (전처리 산출물)",
                 "출력: WB_*.tif · 0=비수체 / 1=수체 / 255=nodata",
                 "patch 추론 + 결과 병합 + Layover/Shadow 보정",
                 "Class weights [0.15, 0.85] · BCE+Dice Loss",
                 "선택: --export_geojson (3세부 표출용)",
                 "센서별 가중치 분리 학습 · 공통 추론 코드"],
                size=11, spacing=7)

    # 좌측 하단: 성능 메트릭
    add_text(slide, Inches(0.5), Inches(4.55), Inches(6), Inches(0.4),
             "■ 검증 성능 (water class)", size=14, bold=True, color=TEAL_DK)
    metrics = [
        ("IoU", "0.84", TEAL),
        ("Precision", "0.86", TEAL),
        ("Recall", "0.97", AMBER),
        ("F1", "0.91", AMBER),
    ]
    for i, (name, val, color) in enumerate(metrics):
        x = Inches(0.5) + Inches(1.5) * i
        b = add_rect(slide, x, Inches(5.05), Inches(1.4), Inches(1.1), fill=WHITE, line=color, line_w=1.5)
        add_text(slide, x, Inches(5.15), Inches(1.4), Inches(0.3),
                 name, size=10, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
        add_text(slide, x, Inches(5.45), Inches(1.4), Inches(0.6),
                 val, size=24, bold=True, color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # 우측: 실제 데모 스크린샷 (SAR / 마스크 / 오버레이 3패널 + 메트릭)
    add_text(slide, Inches(7), Inches(1.2), Inches(6), Inches(0.4),
             "■ 데모 UI · 탐지 결과 (SAR → 마스크 → 오버레이)",
             size=14, bold=True, color=TEAL_DK)
    if has_screen("09_step3_clean.png"):
        add_screenshot(slide, "09_step3_clean.png",
                       Inches(7), Inches(1.65), Inches(5.9), Inches(3.7),
                       caption="📷 입력 SAR · 예측 마스크 · 오버레이 + 면적/비율 메트릭")
    else:
        box_w = Inches(2.85); box_h = Inches(2.85)
        in_box = add_rect(slide, Inches(7), Inches(1.7), box_w, box_h, fill=NAVY, line=TEAL, line_w=1.5)
        out_box = add_rect(slide, Inches(10.0), Inches(1.7), box_w, box_h,
                           fill=RGBColor(0x14, 0x25, 0x3D), line=AMBER, line_w=1.5)

    # 하단: 산출 메타
    add_text(slide, Inches(7), Inches(5.6), Inches(6), Inches(0.4),
             "■ 산출물", size=14, bold=True, color=TEAL_DK)
    add_bullets(slide, Inches(7), Inches(6.0), Inches(5.9), Inches(1.1),
                ["GeoTIFF 수체 마스크 (*_SR_label.tif) · 시계열 분석 입력",
                 "수체 면적 (km²) · 비율 (%) 자동 산출",
                 "시립대(2세부) 산출물과 동일 포맷 → 호환 검증 완료"],
                size=10, spacing=3)
    add_footer(slide)


# ========== Slide 6: Step 3 — 수위·면적 산정 ==========
def slide_step3(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 6, "STEP 3 · 수위·면적 산정 (calc_wlwa.py)",
                   "AOI 크롭 + DEM 기반 간접 수위 · WLWA_*.csv (Water_Level, Water_Area)")

    # 좌측: 처리 흐름
    add_text(slide, Inches(0.5), Inches(1.2), Inches(6), Inches(0.4),
             "■ 처리 흐름 (시계열 수위 산출)", size=14, bold=True, color=TEAL_DK)
    box = add_rect(slide, Inches(0.5), Inches(1.65), Inches(6), Inches(2.1),
                   fill=BG_CARD, line=AMBER, line_w=1.5)
    add_bullets(slide, Inches(0.7), Inches(1.8), Inches(5.6), Inches(2.0),
                ["WB 결과 폴더 스캔 · 날짜 정렬",
                 "Water Line Detection (수체 경계선 추출)",
                 "Extract DEM Value · Select Median",
                 "AOI 위경도 (top_left ↔ bottom_right) 크롭",
                 "→ Time-Series Water Level (XML/CSV)"],
                size=11, spacing=5)

    # 입력 파라미터 표
    add_text(slide, Inches(0.5), Inches(3.95), Inches(6), Inches(0.4),
             "■ 필수 입력 파라미터 (CLI)", size=14, bold=True, color=TEAL_DK)
    vars_ = [
        ("--wb_result_dir", "STEP 2 산출 폴더"),
        ("--roi.top_left / bottom_right", "AOI 위경도"),
        ("--start_date / --end_date", "분석 기간"),
        ("--dem", "수위 추정용 DEM (*.tif)"),
        ("--output_dir", "WLWA_<TB>_<sat>.csv"),
    ]
    y = Inches(4.4)
    for jcol, ko in vars_:
        add_rect(slide, Inches(0.5), y, Inches(6), Inches(0.4), fill=WHITE, line=GRAY_LT)
        add_text(slide, Inches(0.65), y + Inches(0.06), Inches(2.7), Inches(0.3),
                 jcol, size=10, bold=True, color=AMBER)
        add_text(slide, Inches(3.3), y + Inches(0.06), Inches(3.2), Inches(0.3),
                 ko, size=10, color=NAVY)
        y += Inches(0.45)

    # 우측: 산출 XML
    add_text(slide, Inches(7), Inches(1.2), Inches(6), Inches(0.4),
             "■ 산출 메타 XML (예시)", size=14, bold=True, color=TEAL_DK)
    xml_box = add_rect(slide, Inches(7), Inches(1.65), Inches(5.9), Inches(2.7),
                       fill=NAVY, line=AMBER, line_w=1.5)
    add_text(slide, Inches(7.15), Inches(1.75), Inches(5.7), Inches(2.6),
             "<TimeSeriesWaterLevelArea>\n"
             "  <TestbedName>Busan</TestbedName>\n"
             "  <SatelliteName>ICEYE</SatelliteName>\n"
             "  <OrbitNumber>54</OrbitNumber>\n"
             "  <ROI>{'top_left':[42.05,128.0],\n"
             "        'bottom_right':[41.95,128.12]}</ROI>\n"
             "  <Entry date=\"20260611\">\n"
             "    <Water_Level>2189.12</Water_Level>\n"
             "    <Water_Area>9.123</Water_Area>\n"
             "    <Quality>normal</Quality>\n"
             "  </Entry>\n"
             "</TimeSeriesWaterLevelArea>",
             size=10, color=AMBER, font="Consolas")

    # 우측 하단: 시계열 차트
    add_text(slide, Inches(7), Inches(4.55), Inches(6), Inches(0.4),
             "■ 산출 시계열 차트 (수위)", size=14, bold=True, color=TEAL_DK)
    if has_chart("04_timeseries_predict.png"):
        add_chart(slide, "04_timeseries_predict.png",
                  Inches(7), Inches(5.0), Inches(5.9), Inches(2.0))
    add_footer(slide)


# ========== Slide 7: Step 4 — ConvLSTM 시계열 예측 ==========
def slide_step4(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 7, "STEP 4 · 이종 센서 퓨전 보정 (Correct.py)",
                   "ICEYE(SAR) + PlanetScope(광학) + AWS 60일 기상 → corrected_water_level (CWLWA_*)")

    # 좌측: 입력 데이터 구조
    add_text(slide, Inches(0.5), Inches(1.2), Inches(6), Inches(0.4),
             "■ 퓨전 입력 (4채널 × 60일)", size=14, bold=True, color=TEAL_DK)
    arch_box = add_rect(slide, Inches(0.5), Inches(1.65), Inches(6), Inches(1.8),
                        fill=BG_CARD, line=TEAL_DK, line_w=1.5)
    add_bullets(slide, Inches(0.7), Inches(1.8), Inches(5.6), Inches(1.6),
                ["위성 관측 수위 시계열 · [SAR, Optic, 위성 종류]",
                 "직전 60일 AWS 기상 · 기온·강수·습도·일사",
                 "DEM 기반 절대 수위 reference",
                 "선택: in-situ 실측 → 지역 편차(bias) 산출"],
                size=11, spacing=4)

    # 출력 컬럼
    add_text(slide, Inches(0.5), Inches(3.65), Inches(6), Inches(0.4),
             "■ 최종 산출 CSV 컬럼 (CWLWA_*)", size=14, bold=True, color=TEAL_DK)
    cols = [
        ("water_level_m", "보정 전 수위"),
        ("water_area_km2", "관측 면적"),
        ("corrected_water_level_m", "보정 후 수위"),
        ("correction_mode", "absolute · relative · none"),
    ]
    y = Inches(4.1)
    for k, v in cols:
        add_rect(slide, Inches(0.5), y, Inches(6), Inches(0.4),
                 fill=WHITE, line=GRAY_LT)
        add_text(slide, Inches(0.65), y + Inches(0.06), Inches(2.5), Inches(0.3),
                 k, size=10, bold=True, color=AMBER)
        add_text(slide, Inches(3.2), y + Inches(0.06), Inches(3.2), Inches(0.3),
                 v, size=10, color=NAVY)
        y += Inches(0.45)

    # 보정 모드
    add_text(slide, Inches(0.5), Inches(6.0), Inches(6), Inches(0.4),
             "■ 보정 모드 (실측 유무에 따라)", size=14, bold=True, color=TEAL_DK)
    risks = [("absolute", "실측 있음", TEAL),
             ("relative", "실측 없음", AMBER),
             ("none", "보정 안 함", GRAY)]
    for i, (lab, rng, color) in enumerate(risks):
        x = Inches(0.5) + Inches(2.0) * i
        b = add_rect(slide, x, Inches(6.45), Inches(1.9), Inches(0.6),
                     fill=WHITE, line=color, line_w=1.5)
        add_text(slide, x, Inches(6.48), Inches(1.9), Inches(0.27),
                 lab, size=11, bold=True, color=color, align=PP_ALIGN.CENTER)
        add_text(slide, x, Inches(6.72), Inches(1.9), Inches(0.27),
                 rng, size=9, color=NAVY, align=PP_ALIGN.CENTER)

    # 우측: 퓨전 흐름도 + 보정 결과 시각화
    add_text(slide, Inches(7), Inches(1.2), Inches(6), Inches(0.4),
             "■ 데모 UI · 보정 결과 시각화",
             size=14, bold=True, color=TEAL_DK)
    if has_screen("11_step4_final.png"):
        add_screenshot(slide, "11_step4_final.png",
                       Inches(7), Inches(1.65), Inches(5.9), Inches(4.9),
                       caption="📷 관측 마스크(T₀) → 예측·보정 시계열 · 변화 평가")
    else:
        ph = add_rect(slide, Inches(7), Inches(1.65), Inches(5.9), Inches(4.9),
                      fill=BG_CARD, line=AMBER, line_w=1.5)

    add_footer(slide)


# ========== Slide 8: 데모 UI ==========
def slide_demo(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 10, "데모 UI (Streamlit Web)",
                   "발표 시연용 인터랙티브 대시보드 · 단일 클릭으로 1차→2차 처리")

    # 상단: 헤더 + 진행 인디케이터 스크린샷 (와이드)
    add_text(slide, Inches(0.5), Inches(1.2), Inches(12.5), Inches(0.4),
             "■ 상단 헤더 + 4단계 진행 인디케이터",
             size=14, bold=True, color=TEAL_DK)
    if has_screen("00_header.png"):
        add_screenshot(slide, "00_header.png",
                       Inches(0.5), Inches(1.65), Inches(12.4), Inches(1.6))

    # 하단 좌: 사이드바 스크린샷
    add_text(slide, Inches(0.5), Inches(3.5), Inches(6), Inches(0.4),
             "■ CONTROL PANEL (사이드바)",
             size=14, bold=True, color=TEAL_DK)
    if has_screen("02_sidebar.png"):
        # 사이드바는 세로 길쭉
        add_screenshot(slide, "02_sidebar.png",
                       Inches(0.5), Inches(3.95), Inches(2.6), Inches(3.0))
    # 사이드바 설명
    add_bullets(slide, Inches(3.4), Inches(3.95), Inches(3.2), Inches(3.0),
                ["📍 관측 대상 · 지역/일자/시퀀스",
                 "🌧️ 기상 데이터 · API 키/융합",
                 "🔬 모델 설정 · 탐지/시계열/임계값",
                 "🔄 파이프라인 초기화 버튼"],
                size=11, spacing=8)

    # 하단 우: 화면 단계 흐름
    add_text(slide, Inches(7), Inches(3.5), Inches(6), Inches(0.4),
             "■ 화면 진행 흐름",
             size=14, bold=True, color=TEAL_DK)
    steps_ui = [
        ("STEP 1", "위성 SAR 영상 입력 / 업로드", TEAL_DK),
        ("▶ 1차", "수체 탐지 실행 (U-Net)", AMBER),
        ("STEP 2", "탐지 결과 · 면적/비율 메트릭", TEAL),
        ("STEP 3", "기상 데이터 자동 연동", AMBER),
        ("▶ 2차", "추가 시계열 분석 실행", ORANGE),
        ("STEP 4", "예측 + 변화율 + 위험 평가", RED),
    ]
    y = Inches(3.95)
    for tag, desc, color in steps_ui:
        is_btn = "▶" in tag
        bg_c = color if is_btn else WHITE
        text_c = WHITE if is_btn else NAVY
        b = add_rect(slide, Inches(7), y, Inches(5.9), Inches(0.45),
                     fill=bg_c, line=color, line_w=1.5)
        add_text(slide, Inches(7.15), y + Inches(0.1), Inches(0.9), Inches(0.3),
                 tag, size=10, bold=True,
                 color=WHITE if is_btn else color, align=PP_ALIGN.CENTER)
        add_text(slide, Inches(8.1), y + Inches(0.1), Inches(4.6), Inches(0.3),
                 desc, size=10, bold=is_btn, color=text_c)
        y += Inches(0.5)

    # 하단: 접속 정보
    add_text(slide, Inches(0.5), Inches(7.0), Inches(12.5), Inches(0.3),
             "🌐  내부망 http://172.18.10.113:8501    ·    로컬 http://localhost:8501    ·    "
             "실행: bash run.sh",
             size=10, bold=True, color=TEAL_DK)
    add_footer(slide)


# ========== Slide 9: 3세부 연계 — 재난재해 ==========
def slide_disaster(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 11, "3세부 연계 — 재난재해 4종 시뮬레이션",
                   "분석 결과 기반 재해 시각화 데이터 자동 생성")

    add_text(slide, Inches(0.5), Inches(1.2), Inches(12.5), Inches(0.4),
             "■ 4종 시뮬레이션 정의 및 데이터 매핑", size=14, bold=True, color=TEAL_DK)

    disasters = [
        ("강우/폭우", "기상청 특보 활용\n위험 지역 선정 → 집중 관찰 요구",
         "강수량 · 기상특보\n(기상청 API)", "📦", AMBER),
        ("홍수", "수체 면적의 임계값 이상 확대",
         "과거 수체 추론 + 미래 수체 예상\n(2세부 산출물)", "🌊", RED),
        ("가뭄", "수체 면적의 임계값 이상 축소\n(DEM 보간 활용)",
         "과거 수체 추론 + 미래 수체 예상\n(2세부 산출물)", "☀️", ORANGE),
        ("침수", "수원 없는 지역 내 수체 생성 확률 ↑\n통계 기반 임의 수체 시각화 (DEM 활용)",
         "기상·지역별 침수 발생 통계", "🏘️", TEAL_DK),
    ]
    bw = Inches(3.0); bh = Inches(4.5); gap = Inches(0.15)
    sx = Inches(0.5)
    y = Inches(1.65)
    for i, (name, defn, data, emoji, color) in enumerate(disasters):
        x = sx + (bw + gap) * i
        b = add_rect(slide, x, y, bw, bh, fill=BG_CARD, line=color, line_w=2)
        # 상단 헤더 컬러
        head = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, bw, Inches(0.7))
        head.fill.solid(); head.fill.fore_color.rgb = color
        head.line.fill.background(); head.shadow.inherit = False
        # round corner overlap fix - tolerable for demo
        add_text(slide, x + Inches(0.2), y + Inches(0.15), Inches(0.5), Inches(0.4),
                 emoji, size=18, anchor=MSO_ANCHOR.MIDDLE)
        add_text(slide, x + Inches(0.85), y + Inches(0.1), bw - Inches(1), Inches(0.5),
                 name, size=15, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
        # 정의
        add_text(slide, x + Inches(0.2), y + Inches(0.9), bw - Inches(0.4), Inches(0.4),
                 "정의", size=10, bold=True, color=color)
        add_text(slide, x + Inches(0.2), y + Inches(1.25), bw - Inches(0.4), Inches(1.4),
                 defn, size=11, color=NAVY)
        # 필요 데이터
        add_text(slide, x + Inches(0.2), y + Inches(2.95), bw - Inches(0.4), Inches(0.4),
                 "필요 데이터", size=10, bold=True, color=color)
        add_text(slide, x + Inches(0.2), y + Inches(3.3), bw - Inches(0.4), Inches(1.2),
                 data, size=11, color=GRAY)

    # 하단: API 흐름
    add_text(slide, Inches(0.5), Inches(6.35), Inches(12.5), Inches(0.4),
             "■ 데이터 흐름 (REST API)", size=12, bold=True, color=TEAL_DK)
    add_text(slide, Inches(0.5), Inches(6.8), Inches(12.5), Inches(0.4),
             "1세부 (영상 수신) → 2세부 이노뎁 (수체 탐지 + 시계열 예측) → 3세부 (재난재해 표출)",
             size=11, color=NAVY)
    add_footer(slide)


# ========== Slide 10: 향후 계획 ==========
def slide_roadmap(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 12, "향후 계획 · 3차년도 잔여 일정",
                   "TTA 평가 / 3세부 연계 / 작물 분류 POC")

    # 4분면 카드
    items = [
        ("01", "TTA 평가 준비", "재현성 검증 + 테스트셋 확보\n성능표 & 평가 시나리오 작성\n3차년도 평가 일정 대응", TEAL_DK),
        ("02", "3세부 연계 API", "REST API 스펙 협의\n데이터 포맷 (GeoTIFF/JSON)\n홍수/가뭄 시나리오 송신",  ORANGE),
        ("03", "재난재해 4종", "강우(기상청 API 래핑)\n홍수·가뭄(예측 모델 재활용)\n침수(DEM + 통계 모델 신규)", RED),
        ("04", "작물 분류 POC", "데이터셋 확보\n작물 Segmentation 모델\n자체 평가 (TTA 없음)", AMBER),
    ]
    bw = Inches(6.1); bh = Inches(2.7); gap = Inches(0.2)
    sx = Inches(0.5)
    sy = Inches(1.3)
    for idx, (num, title, desc, color) in enumerate(items):
        r = idx // 2; c = idx % 2
        x = sx + c * (bw + gap)
        y = sy + r * (bh + gap)
        b = add_rect(slide, x, y, bw, bh, fill=WHITE, line=color, line_w=2)
        # 좌측 컬러바
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(0.3), bh)
        bar.fill.solid(); bar.fill.fore_color.rgb = color
        bar.line.fill.background(); bar.shadow.inherit = False
        add_text(slide, x + Inches(0.5), y + Inches(0.2), Inches(1.5), Inches(0.7),
                 num, size=36, bold=True, color=color)
        add_text(slide, x + Inches(1.9), y + Inches(0.3), Inches(4), Inches(0.6),
                 title, size=18, bold=True, color=NAVY)
        add_text(slide, x + Inches(0.5), y + Inches(1.2), bw - Inches(0.8), Inches(1.5),
                 desc, size=12, color=GRAY)

    # 하단: 일정
    add_text(slide, Inches(0.5), Inches(7.0), Inches(12.5), Inches(0.3),
             "3차년도: 2026.01.01 ~ 2026.12.31    ·    전체 과제: 2024.04 ~ 2026.12",
             size=10, color=GRAY, align=PP_ALIGN.CENTER)
    add_footer(slide)


# ========== Slide A: 이노뎁 — 시계열 예측 모델 개발 과정 ==========
def slide_innodep_dev(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 8, "★ 이노뎁 차별 영역 · 시계열 예측 모델",
                   "4단계 시스템 산출물(WLWA) + 기상청 API → ConvLSTM → 미래 시점 수체/수위 예측")

    # ───────── 상단: 입력 → 모델 → 출력 흐름 ─────────
    add_text(slide, Inches(0.4), Inches(1.2), Inches(12.5), Inches(0.4),
             "■ Input → Model → Output 흐름 (이노뎁 차별 영역)",
             size=14, bold=True, color=TEAL_DK)

    flow_y = Inches(1.7)
    flow_h = Inches(1.6)

    # ① INPUT 카드 (왼쪽)
    in_box = add_rect(slide, Inches(0.4), flow_y, Inches(3.8), flow_h,
                      fill=BG_CARD, line=TEAL_DK, line_w=2)
    add_text(slide, Inches(0.55), flow_y + Inches(0.1), Inches(3.5), Inches(0.35),
             "📥  INPUT", size=11, bold=True, color=TEAL_DK)
    add_text(slide, Inches(0.55), flow_y + Inches(0.45), Inches(3.5), Inches(0.32),
             "① 수체/수위 시계열", size=11, bold=True, color=NAVY)
    add_text(slide, Inches(0.7), flow_y + Inches(0.75), Inches(3.4), Inches(0.25),
             "WLWA_*.csv (4단계 시스템 산출)",
             size=9, color=GRAY)
    add_text(slide, Inches(0.55), flow_y + Inches(1.02), Inches(3.5), Inches(0.32),
             "② 기상청 API ⭐", size=11, bold=True, color=AMBER)
    add_text(slide, Inches(0.7), flow_y + Inches(1.3), Inches(3.4), Inches(0.25),
             "강수·기온·습도·풍속 (일자료)",
             size=9, color=GRAY)

    # 화살표 1
    arr1 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                  Inches(4.25), flow_y + Inches(0.65),
                                  Inches(0.35), Inches(0.4))
    arr1.fill.solid(); arr1.fill.fore_color.rgb = TEAL
    arr1.line.fill.background(); arr1.shadow.inherit = False

    # ② MODEL 카드 (가운데)
    md_box = add_rect(slide, Inches(4.7), flow_y, Inches(3.8), flow_h,
                      fill=NAVY, line=AMBER, line_w=2)
    add_text(slide, Inches(4.85), flow_y + Inches(0.1), Inches(3.5), Inches(0.35),
             "🧠  MODEL", size=11, bold=True, color=AMBER)
    add_text(slide, Inches(4.85), flow_y + Inches(0.45), Inches(3.5), Inches(0.45),
             "ConvLSTM\nEncoder-Decoder", size=14, bold=True, color=WHITE)
    add_text(slide, Inches(4.85), flow_y + Inches(1.0), Inches(3.5), Inches(0.55),
             "512→256→128 (Enc)\n+ Weather Fusion (60일)\n128→256→512 (Dec)",
             size=9, color=TEAL)

    # 화살표 2
    arr2 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                  Inches(8.55), flow_y + Inches(0.65),
                                  Inches(0.35), Inches(0.4))
    arr2.fill.solid(); arr2.fill.fore_color.rgb = TEAL
    arr2.line.fill.background(); arr2.shadow.inherit = False

    # ③ OUTPUT 카드 (오른쪽)
    out_box = add_rect(slide, Inches(9.0), flow_y, Inches(3.9), flow_h,
                       fill=BG_CARD, line=RED, line_w=2)
    add_text(slide, Inches(9.15), flow_y + Inches(0.1), Inches(3.6), Inches(0.35),
             "📤  OUTPUT", size=11, bold=True, color=RED)
    add_text(slide, Inches(9.15), flow_y + Inches(0.45), Inches(3.6), Inches(0.32),
             "③ 미래 N프레임 수체", size=11, bold=True, color=NAVY)
    add_text(slide, Inches(9.3), flow_y + Inches(0.75), Inches(3.5), Inches(0.25),
             "T+1, T+2, T+3 ... GeoTIFF",
             size=9, color=GRAY)
    add_text(slide, Inches(9.15), flow_y + Inches(1.02), Inches(3.6), Inches(0.32),
             "④ 변화율 + 위험 평가", size=11, bold=True, color=NAVY)
    add_text(slide, Inches(9.3), flow_y + Inches(1.3), Inches(3.5), Inches(0.25),
             "→ 3세부 재난재해 시뮬레이션",
             size=9, color=GRAY)

    # ───────── 중단: 개발 파이프라인 다이어그램 ─────────
    add_text(slide, Inches(0.4), Inches(3.5), Inches(12.5), Inches(0.4),
             "■ 모델 개발 파이프라인 (Data → Model → Eval → Deploy)",
             size=13, bold=True, color=TEAL_DK)
    if has_chart("07_pipeline_dev.png"):
        add_chart(slide, "07_pipeline_dev.png",
                  Inches(0.4), Inches(3.95), Inches(12.5), Inches(1.55))

    # ───────── 하단 좌: 데이터셋 분할 ─────────
    add_text(slide, Inches(0.4), Inches(5.65), Inches(7), Inches(0.4),
             "■ 학습 데이터 구성 (시간 기반)",
             size=13, bold=True, color=TEAL_DK)
    if has_chart("06_data_split.png"):
        add_chart(slide, "06_data_split.png",
                  Inches(0.4), Inches(6.05), Inches(7.2), Inches(1.1))

    # ───────── 하단 우: 학습 설정 ─────────
    add_text(slide, Inches(7.8), Inches(5.65), Inches(5.2), Inches(0.4),
             "■ 학습 설정 & 검증 전략",
             size=13, bold=True, color=TEAL_DK)
    set_box = add_rect(slide, Inches(7.8), Inches(6.05), Inches(5.1), Inches(1.1),
                       fill=BG_CARD, line=ORANGE, line_w=1.5)
    add_bullets(slide, Inches(8.0), Inches(6.15), Inches(4.8), Inches(1.0),
                ["Loss: BCE + Dice  ·  Adam(lr=1e-4)",
                 "Mixed precision · Early stop(patience=10)",
                 "Hold-out 15% + Δt별 정확도 + 잔차 분석"],
                size=10, spacing=2)

    add_footer(slide)


# ========== Slide B: 이노뎁 — 정확도 시각화 ==========
def slide_innodep_accuracy(prs):
    slide = add_blank_slide(prs)
    add_header_bar(slide, 9, "이노뎁 · 시계열 예측 정확도 시각화",
                   "학습곡선 · 1:1 산점도 · Δt별 성능 저하 · 잔차 분포")

    # 상단 좌: 학습곡선
    add_text(slide, Inches(0.4), Inches(1.2), Inches(6.3), Inches(0.4),
             "■ 학습 / 검증 곡선 (Loss & IoU)",
             size=13, bold=True, color=TEAL_DK)
    if has_chart("01_training_curve.png"):
        add_chart(slide, "01_training_curve.png",
                  Inches(0.4), Inches(1.65), Inches(6.4), Inches(2.4))

    # 상단 우: 1:1 산점도
    add_text(slide, Inches(7.0), Inches(1.2), Inches(6), Inches(0.4),
             "■ 1:1 산점도 · 수위 예측 회귀 정확도",
             size=13, bold=True, color=TEAL_DK)
    if has_chart("02_scatter_water_level.png"):
        add_chart(slide, "02_scatter_water_level.png",
                  Inches(7.0), Inches(1.65), Inches(2.7), Inches(2.4))
    # 산점도 옆: 메트릭 요약
    mbox = add_rect(slide, Inches(9.9), Inches(1.65), Inches(3.0), Inches(2.4),
                    fill=BG_CARD, line=TEAL, line_w=1.5)
    add_text(slide, Inches(10.0), Inches(1.75), Inches(2.8), Inches(0.35),
             "회귀 메트릭 (수위)", size=11, bold=True, color=TEAL_DK)
    metric_rows = [
        ("MAE", "0.365 m", TEAL),
        ("RMSE", "0.476 m", TEAL),
        ("MAPE", "5.8 %", AMBER),
        ("R²", "0.973", ORANGE),
    ]
    yy = Inches(2.15)
    for k, v, c in metric_rows:
        add_rect(slide, Inches(10.0), yy, Inches(2.8), Inches(0.42),
                 fill=WHITE, line=c, line_w=1)
        add_text(slide, Inches(10.15), yy + Inches(0.08), Inches(1.4), Inches(0.3),
                 k, size=11, bold=True, color=GRAY)
        add_text(slide, Inches(11.5), yy + Inches(0.08), Inches(1.2), Inches(0.3),
                 v, size=13, bold=True, color=c, align=PP_ALIGN.RIGHT)
        yy += Inches(0.47)

    # 하단 좌: Δt별 성능
    add_text(slide, Inches(0.4), Inches(4.2), Inches(12.5), Inches(0.4),
             "■ 예측 시점(Δt)별 정확도 저하 — 마스크 IoU/Dice & 수위 MAE",
             size=13, bold=True, color=TEAL_DK)
    if has_chart("03_horizon_metrics.png"):
        add_chart(slide, "03_horizon_metrics.png",
                  Inches(0.4), Inches(4.65), Inches(8.6), Inches(2.3))

    # 하단 우: 잔차 분포 (좁게)
    add_text(slide, Inches(9.2), Inches(4.2), Inches(4), Inches(0.4),
             "■ 잔차 분석 (Bias 검증)",
             size=13, bold=True, color=TEAL_DK)
    if has_chart("05_residual_qq.png"):
        add_chart(slide, "05_residual_qq.png",
                  Inches(9.2), Inches(4.65), Inches(3.7), Inches(1.8))
    add_text(slide, Inches(9.2), Inches(6.5), Inches(3.7), Inches(0.4),
             "평균 잔차 +0.05 m · 편향 없음",
             size=10, color=ORANGE, bold=True)
    add_text(slide, Inches(9.2), Inches(6.78), Inches(3.7), Inches(0.4),
             "정규성 확보 → 모델 가정 만족",
             size=9, color=GRAY)

    # ───────── 결과 한 줄 요약 (강조 띠) ─────────
    sum_y = Inches(7.0)
    sum_bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(0.4), sum_y, Inches(8.6), Inches(0.45))
    sum_bg.adjustments[0] = 0.25
    sum_bg.fill.solid(); sum_bg.fill.fore_color.rgb = AMBER
    sum_bg.line.fill.background(); sum_bg.shadow.inherit = False
    add_text(slide, Inches(0.55), sum_y, Inches(1.3), Inches(0.45),
             "✓ 결과 요약", size=11, bold=True, color=NAVY,
             anchor=MSO_ANCHOR.MIDDLE)
    add_text(slide, Inches(1.85), sum_y, Inches(7.0), Inches(0.45),
             "수위 R² 0.97 · MAE 36 cm · 단기(11~22일) IoU 0.86 · 장기(55일) IoU 0.62 유지",
             size=11, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)

    add_footer(slide)


# ========== Slide 11: Thank you / Q&A ==========
def slide_thanks(prs):
    slide = add_blank_slide(prs)
    # 배경 네이비
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background(); bg.shadow.inherit = False
    # 상단 컬러바
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.8), SLIDE_W, Inches(0.1))
    accent.fill.solid(); accent.fill.fore_color.rgb = AMBER
    accent.line.fill.background(); accent.shadow.inherit = False
    add_text(slide, Inches(0.5), Inches(2.0), Inches(12.5), Inches(0.8),
             "감사합니다", size=64, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, Inches(0.5), Inches(3.1), Inches(12.5), Inches(0.5),
             "Q & A", size=24, bold=True, color=AMBER, align=PP_ALIGN.CENTER)
    add_text(slide, Inches(0.5), Inches(4.5), Inches(12.5), Inches(0.4),
             "IITP 초소형 위성영상 기반 주요 지역 분석 및 실감화 지능 기술 개발",
             size=14, color=TEAL, align=PP_ALIGN.CENTER)
    add_text(slide, Inches(0.5), Inches(4.95), Inches(12.5), Inches(0.4),
             "2세부 · 이노뎁",
             size=12, color=GRAY_LT, align=PP_ALIGN.CENTER)
    add_text(slide, Inches(0.5), Inches(6.5), Inches(12.5), Inches(0.4),
             "minkyu_choi@innodep.com",
             size=11, color=GRAY_LT, align=PP_ALIGN.CENTER)


# ========== Main ==========
def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "발표자료_파이프라인.pptx"
    prs = new_prs()
    slide_title(prs)
    slide_overview(prs)
    slide_pipeline(prs)
    slide_step1(prs)
    slide_step2(prs)
    slide_step3(prs)
    slide_step4(prs)
    slide_innodep_dev(prs)        # NEW · 이노뎁 시계열 예측 개발 과정
    slide_innodep_accuracy(prs)   # NEW · 이노뎁 정확도 시각화
    slide_demo(prs)
    slide_disaster(prs)
    slide_roadmap(prs)
    slide_thanks(prs)
    prs.save(str(out))
    print(f"✓ {len(prs.slides)} slides → {out}")


if __name__ == "__main__":
    main()
