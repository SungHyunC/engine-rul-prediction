"""Build bilingual completed-project PDFs from measured artifacts.

Run with the bundled document Python. The script refuses missing result files;
it never substitutes illustrative metrics. Rendering and visual QA follow build.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle, Polygon, Group
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
NASA = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
PAPER = "https://doi.org/10.3390/data6010005"
MIRROR = "https://doi.org/10.6084/m9.figshare.20436504.v1"
MODELS = ("dummy_mean", "ridge", "hist_gradient_boosting")
LABELS = {"dummy_mean": "Mean baseline", "ridge": "Ridge", "hist_gradient_boosting": "HGB"}
NAVY = colors.HexColor("#18384A")
TEAL = colors.HexColor("#147F88")
ORANGE = colors.HexColor("#D78A36")
PALE = colors.HexColor("#EEF5F5")
GRAY = colors.HexColor("#607580")
GRID = colors.HexColor("#D9E4E8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Report:
    def __init__(self, lang, output, font_dir):
        self.lang, self.output = lang, output
        self.regular, self.bold = f"Report-{lang}", f"Report-{lang}-Bold"
        pdfmetrics.registerFont(TTFont(self.regular, str(font_dir / "malgun.ttf")))
        pdfmetrics.registerFont(TTFont(self.bold, str(font_dir / "malgunbd.ttf")))
        pdfmetrics.registerFontFamily(self.regular, normal=self.regular, bold=self.bold,
                                     italic=self.regular, boldItalic=self.bold)
        wrap = "CJK" if lang == "ko" else None
        self.body = ParagraphStyle("Body", fontName=self.regular, fontSize=10.5, leading=14.5,
                                   spaceAfter=7, alignment=TA_LEFT, wordWrap=wrap)
        self.title = ParagraphStyle("Title", parent=self.body, fontName=self.bold, fontSize=20, leading=26, spaceAfter=10, textColor=NAVY)
        self.heading = ParagraphStyle("Heading", parent=self.body, fontName=self.bold, fontSize=13, leading=17,
                                      spaceBefore=10, spaceAfter=6, keepWithNext=True, textColor=NAVY)
        self.small = ParagraphStyle("Small", parent=self.body, fontSize=8.5, leading=11.5, spaceAfter=6)
        self.cell = ParagraphStyle("Cell", parent=self.body, fontSize=8.5, leading=11, spaceAfter=0)
        self.cellhead = ParagraphStyle("CellHead", parent=self.cell, fontName=self.bold)
        self.story = []
        self.width = 174 * mm

    def tr(self, ko, en):
        return ko if self.lang == "ko" else en

    def para(self, ko, en, style=None):
        self.story.append(Paragraph(escape(self.tr(ko, en)), style or self.body))

    def head(self, ko, en):
        self.para(ko, en, self.heading)

    def page(self, ko, en):
        if self.story:
            self.story.append(PageBreak())
        self.para(ko, en, self.title)

    def table(self, headers, rows, widths):
        content = [[Paragraph(escape(str(value)), self.cellhead if i == 0 else self.cell)
                    for value in row] for i, row in enumerate([headers] + rows)]
        table = Table(content, colWidths=[x * mm for x in widths], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PALE),
            ("LINEBELOW", (0, 0), (-1, 0), .7, TEAL),
            ("LINEBELOW", (0, 1), (-1, -1), .35, GRID),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        self.story.extend([table, Spacer(1, 7)])

    def figure(self, path, max_height=260):
        image = Image(str(path))
        scale = min(self.width / image.imageWidth, max_height / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
        image.hAlign = "CENTER"
        self.story.extend([image, Spacer(1, 4)])

    def draw(self, drawing):
        self.story.extend([drawing, Spacer(1, 7)])

    def text(self, d, x, y, value, size=9, color=NAVY, bold=False, anchor="start"):
        d.add(String(x, y, str(value), fontName=self.bold if bold else self.regular,
                     fontSize=size, fillColor=color, textAnchor=anchor))

    def metrics(self, items):
        d = Drawing(self.width, 57)
        col = self.width / len(items)
        for i, (value, label, detail) in enumerate(items):
            x = i * col
            d.add(Rect(x, 0, col-8, 57, fillColor=PALE, strokeColor=None))
            self.text(d,x+10,32,value,21,TEAL,True)
            self.text(d,x+10,17,label,8.5,NAVY,True)
            self.text(d,x+10,5,detail,7.5,GRAY)
        self.draw(d)

    def pipeline(self):
        d = Drawing(self.width, 79)
        labels = self.tr([("수집·검사", "DS02 HDF5"),("경계별 윈도", "18채널 × 60"),("학습·예측", "Ridge/HGB/CNN"),("평균·선택", "비행 집계 → CV"),("추론·검증", "정답 없이 실행")],
                         [("Inspect", "DS02 HDF5"),("Window", "18 channels × 60"),("Fit + predict", "Ridge/HGB/CNN"),("Mean + select", "Flight mean → CV"),("Infer + audit", "No targets needed")])
        step = self.width/5
        for i,(title,subtitle) in enumerate(labels):
            x=i*step
            d.add(Circle(x+13,63,9,fillColor=TEAL,strokeColor=None))
            self.text(d,x+13,60,str(i+1),8,colors.white,True,"middle")
            d.add(Rect(x,7,step-12,40,fillColor=PALE,strokeColor=None))
            self.text(d,x+7,31,title,9,NAVY,True)
            self.text(d,x+7,16,subtitle,7.5,GRAY)
            if i<4:
                d.add(Line(x+step-9,27,x+step-3,27,strokeColor=TEAL,strokeWidth=1))
                d.add(Polygon([x+step-2,27,x+step-5,29,x+step-5,25],fillColor=TEAL,strokeColor=None))
        self.draw(d)

    def fold_matrix(self, folds):
        d=Drawing(self.width,111)
        units=[2,5,10,16,18,20,11,14,15]
        x0,step=92,43
        self.text(d,0,96,self.tr("개발 / 시험 분할", "Engine split / folds"),9,NAVY,True)
        for j,u in enumerate(units):
            x=x0+j*step
            self.text(d,x+17,96,u,9,NAVY,True,"middle")
            self.text(d,x+17,83,f"C{3 if u not in (14,15) else (1 if u==14 else 2)}",7,GRAY,False,"middle")
        for i,fold in enumerate(folds):
            y=58-i*20
            self.text(d,0,y+5,self.tr("폴드 ","Fold ")+str(fold["fold"]),8.5)
            val=json.loads(fold["validation_units"])
            for j,u in enumerate(units):
                color=TEAL if u in val else PALE if u in units[:6] else colors.HexColor("#F5E8D9")
                d.add(Rect(x0+j*step,y,34,16,fillColor=color,strokeColor=None))
                label=self.tr("검증","CV") if u in val else self.tr("학습","Fit") if u in units[:6] else self.tr("시험","Test")
                self.text(d,x0+j*step+17,y+5,label,7,colors.white if u in val else NAVY,False,"middle")
        self.text(d,0,1,self.tr("C = 비행 클래스. 시험 엔진은 학습·선택에 사용하지 않는다.","C = flight class. Test engines are excluded from fitting and selection."),8,GRAY)
        self.draw(d)

    def window_diagram(self):
        d=Drawing(self.width,76)
        self.text(d,0,62,self.tr("한 엔진의 한 비행 사이클", "One engine, one recorded flight"),9,NAVY,True)
        for j in range(3):
            x=j*94
            d.add(Rect(x,28,83,23,fillColor=PALE,strokeColor=TEAL,strokeWidth=.6))
            self.text(d,x+41.5,36,f"{j*60+1}-{(j+1)*60}",8,NAVY,False,"middle")
        d.add(Rect(282,28,25,23,fillColor=colors.HexColor("#F5E8D9"),strokeColor=None))
        self.text(d,294.5,36,self.tr("꼬리","tail"),6.5,ORANGE,False,"middle")
        self.text(d,321,43,self.tr("18채널 × 5통계 = 90특징", "18 channels × 5 stats = 90 features"),8.2,TEAL,True)
        self.text(d,321,29,"mean / std / min / max / slope",7.8)
        self.text(d,0,10,self.tr("윈도 = 60, stride = 60; 불완전 꼬리 제외; 다음 비행에서 초기화", "Window = 60, stride = 60; drop incomplete tail; reset at next flight"),8,GRAY)
        self.draw(d)

    def comparison(self, rows, height=151):
        """Paired zero-origin bars, annotated directly from saved macro RMSE."""
        d=Drawing(self.width,height)
        left,right=142,self.width-40
        xmax=max(max(a,b) for _,a,b in rows)*1.12
        scale=(right-left)/xmax
        self.text(d,0,height-10,self.tr("개발 CV", "Development CV"),8,TEAL,True)
        self.text(d,113,height-10,self.tr("시험 (기술적 비교)", "Test (descriptive)"),8,ORANGE,True)
        self.text(d,self.width,height-10,self.tr("RMSE · 사이클 ↓", "RMSE · cycles ↓"),8,GRAY,False,"end")
        for tick in range(0,int(xmax)+1,5):
            x=left+tick*scale
            d.add(Line(x,21,x,height-29,strokeColor=GRID,strokeWidth=.4))
            self.text(d,x,7,str(tick),7,GRAY,False,"middle")
        dy=(height-40)/len(rows)
        for i,(label,dev,test) in enumerate(rows):
            y=height-40-i*dy
            self.text(d,0,y-3,label,8.5,NAVY,True)
            for offset,value,color in [(2,dev,TEAL),(-10,test,ORANGE)]:
                d.add(Rect(left,y+offset,value*scale,8,fillColor=color,strokeColor=None))
                self.text(d,left+value*scale+4,y+offset,value.__format__(".3f"),7.5,color)
        self.draw(d)

    def heatmap(self,candidates):
        d=Drawing(self.width,233)
        pairs=sorted({(int(x['window']),int(x['stride'])) for x in candidates})
        cols=[('ridge',False),('ridge',True),('hist_gradient_boosting',False),('hist_gradient_boosting',True)]
        lookup={(int(x['window']),int(x['stride']),x['model'],str(x['nonnegative']).lower()=='true'):x for x in candidates}
        vals=[float(x['cv_macro_engine_cycle_rmse']) for x in candidates]
        lo,hi=min(vals),max(vals)
        self.text(d,0,218,self.tr("윈도 / stride", "Window / stride"),8,NAVY,True)
        for j,label in enumerate(self.tr(["Ridge 원시","Ridge 비음수","HGB 원시","HGB 비음수"],["Ridge raw","Ridge ≥0","HGB raw","HGB ≥0"])):
            self.text(d,149+j*97,218,label,8,NAVY,True,"middle")
        for i,(w,s) in enumerate(pairs):
            y=178-i*29
            self.text(d,0,y+10,f"{w} / {s}",9,NAVY,True)
            for j,(model,policy) in enumerate(cols):
                row=lookup[(w,s,model,policy)]; value=float(row['cv_macro_engine_cycle_rmse']); t=(value-lo)/(hi-lo)
                color=colors.linearlyInterpolatedColor(TEAL,colors.HexColor('#EFF4F5'),0,1,t)
                selected=str(row['selected_by_dev_cv']).lower()=='true'
                d.add(Rect(102+j*97,y,94,27,fillColor=color,strokeColor=ORANGE if selected else colors.white,strokeWidth=2 if selected else .5))
                self.text(d,149+j*97,y+9,f"{value:.3f}"+(' *' if selected else ''),9,colors.white if t<.45 else NAVY,selected,"middle")
        self.text(d,0,12,self.tr("진한 청록 = 낮은 오차   * 개발 CV로 선택   단위: 사이클", "Darker teal = lower error   * selected by development CV   Unit: cycles"),8,GRAY)
        self.draw(d)

    def architecture(self):
        d=Drawing(self.width,89)
        blocks=[("18 × 60",self.tr("정규화 입력","Scaled input")),("16 × 60","Conv1d + ReLU"),("32 × 60","Conv1d + ReLU"),("32",self.tr("전역 평균","Global mean")),("RUL ≥ 0","32→1 + Softplus")]
        step=self.width/5
        for i,(shape,name) in enumerate(blocks):
            x=i*step
            d.add(Rect(x,30,step-12,43,fillColor=TEAL if i==4 else PALE,strokeColor=None))
            self.text(d,x+(step-12)/2,53,shape,12,colors.white if i==4 else TEAL,True,"middle")
            self.text(d,x+(step-12)/2,38,name,7.4,colors.white if i==4 else NAVY,False,"middle")
            if i<4:
                self.text(d,x+step-7,47,"→",10,GRAY,False,"middle")
        self.text(d,0,12,self.tr("합성곱: kernel=5, padding=2  |  Linear 32→1  |  출력 × 100  |  4,081개 파라미터", "Convolutions: kernel=5, padding=2  |  Linear 32→1  |  output × 100  |  4,081 parameters"),8,GRAY)
        self.draw(d)

    def diagnostic(self,predictions):
        d=Drawing(self.width,230)
        regions=[(36,24,184,167),(292,24,182,167)]
        palette={11:TEAL,14:ORANGE,15:NAVY}
        for idx,(x,y,w,h) in enumerate(regions):
            self.text(d,x,y+h+17,self.tr("실제 vs 예측 RUL","Actual vs predicted RUL") if idx==0 else self.tr("잔차 vs 실제 RUL","Residual vs actual RUL"),9,NAVY,True)
            axis=Group(transform=(0,1,-1,0,x-27,y+h/2))
            self.text(axis,0,0,self.tr("예측 RUL (사이클)","Predicted RUL (cycles)") if idx==0 else self.tr("잔차 (사이클)","Residual (cycles)"),7,GRAY,False,"middle")
            d.add(axis)
            for tick in [0,20,40,60,80]:
                xx=x+tick/80*w
                d.add(Line(xx,y,xx,y+h,strokeColor=GRID,strokeWidth=.4))
                self.text(d,xx,y-12,tick,7,GRAY,False,"middle")
            yticks=[0,20,40,60,80] if idx==0 else [-20,-10,0,10,20]
            for tick in yticks:
                yy=y+(tick/80 if idx==0 else (tick+20)/40)*h
                d.add(Line(x,yy,x+w,yy,strokeColor=GRAY if idx==1 and tick==0 else GRID,strokeWidth=.6))
                self.text(d,x-5,yy-2,tick,7,GRAY,False,"end")
            if idx==0: d.add(Line(x,y,x+w,y+h,strokeColor=GRAY,strokeWidth=.8,strokeDashArray=[3,2]))
            for row in predictions:
                actual=float(row['target']); prediction=float(row['prediction'])
                xx=x+actual/80*w; yy=y+(prediction/80 if idx==0 else (prediction-actual+20)/40)*h
                d.add(Circle(xx,yy,1.7,fillColor=palette[int(row['unit'])],strokeColor=None))
            self.text(d,x+w/2,y-21,self.tr("실제 RUL (사이클)","Actual RUL (cycles)"),7,GRAY,False,"middle")
        for i,u in enumerate([11,14,15]):
            d.add(Circle(144+i*87,224,2.8,fillColor=palette[u],strokeColor=None))
            self.text(d,151+i*87,221,self.tr(f"엔진 {u}",f"Engine {u}"),8,GRAY)
        self.draw(d)

    def benchmark_chart(self,summary,fixed_chunk):
        d=Drawing(self.width,190)
        rows=sorted([x for x in summary if int(x['chunk_rows'])==fixed_chunk],key=lambda x:int(x['workers']))
        self.text(d,0,177,self.tr("시간 중앙값 / 최소-최대", "Time median / min-max"),9,NAVY,True)
        self.text(d,275,177,self.tr("메모리와 속도비", "Memory and speedup"),9,NAVY,True)
        for tick in [0,1,2]:
            x=55+tick/2.5*174
            d.add(Line(x,28,x,159,strokeColor=GRID,strokeWidth=.5))
            self.text(d,x,14,tick,7,GRAY,False,"middle")
        for i,row in enumerate(rows):
            y=143-i*31; median=float(row['median_seconds']); lo=float(row['min_seconds']); hi=float(row['max_seconds']); workers=int(row['workers'])
            c=TEAL if workers==4 else NAVY
            self.text(d,0,y-3,f"{workers} "+self.tr("개","worker" if workers==1 else "workers"),8,NAVY,workers==4)
            d.add(Line(55+lo/2.5*174,y,55+hi/2.5*174,y,strokeColor=c,strokeWidth=2))
            d.add(Circle(55+median/2.5*174,y,3,fillColor=c,strokeColor=None))
            self.text(d,238,y-3,f"{median:.3f}",8,c,True,"end")
            memory=float(row['peak_rss_mib'])
            d.add(Rect(275,y-5,memory/1000*111,10,fillColor=PALE,strokeColor=None))
            d.add(Rect(275,y-5,memory/1000*111,2,fillColor=c,strokeColor=None))
            self.text(d,390,y-3,f"{memory:.0f} MiB",8,c)
            self.text(d,self.width,y-3,f"{float(row['speedup_vs_one_worker']):.2f}×",9,c,True,"end")
        self.text(d,55,0,self.tr("초 (낮을수록 빠름)","Seconds (lower is faster)"),8,GRAY)
        self.draw(d)

    def life_stage_chart(self,stages):
        d=Drawing(self.width,110)
        self.text(d,0,98,self.tr("실제 RUL 구간", "True RUL bin"),8,NAVY,True)
        self.text(d,145,98,self.tr("Macro RMSE · 사이클", "Macro RMSE · cycles"),8,TEAL,True)
        self.text(d,340,98,"MAE",8,NAVY,True)
        self.text(d,392,98,self.tr("편향", "Bias"),8,NAVY,True)
        self.text(d,447,98,self.tr("사이클 수", "Cycles"),8,NAVY,True)
        for i,row in enumerate(stages):
            y=73-i*29
            self.text(d,0,y-1,row['life_stage'],8,NAVY,True)
            rmse=float(row['macro_engine_rmse'])
            d.add(Rect(133,y-3,rmse/8*139,10,fillColor=TEAL,strokeColor=None))
            self.text(d,133+rmse/8*139+5,y-1,f"{rmse:.3f}",8,TEAL,True)
            self.text(d,340,y-1,f"{float(row['macro_engine_mae']):.3f}",8)
            self.text(d,392,y-1,f"+{float(row['macro_engine_bias']):.3f}",8,ORANGE)
            self.text(d,458,y-1,row['n_cycles'],8)
        self.draw(d)

    def finish(self):
        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont(self.regular, 8)
            canvas.setFillColor(colors.HexColor("#555555"))
            canvas.setFillColor(TEAL)
            canvas.rect(18*mm, A4[1]-10*mm, 17*mm, 2, fill=1, stroke=0)
            canvas.setFillColor(GRAY)
            canvas.setFont(self.regular, 7)
            canvas.drawRightString(192*mm, A4[1]-10*mm, "N-CMAPSS DS02  /  COMPUTATIONAL EXPERIMENT")
            canvas.setFont(self.regular, 8)
            canvas.drawString(18 * mm, 12 * mm, self.tr("김성현  2021271250  |  프로젝트 보고서  2026.10.02", "김성현  2021271250  |  Project report  2026-10-02"))
            canvas.drawRightString(192 * mm, 12 * mm, str(doc.page))
            canvas.restoreState()
        doc = SimpleDocTemplate(str(self.output), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                                topMargin=17*mm, bottomMargin=20*mm, author="김성현",
                                title=self.tr("N-CMAPSS DS02 엔진 잔여수명 예측 프로젝트 보고서", "N-CMAPSS DS02 Engine RUL Prediction Project Report"))
        doc.build(self.story, onFirstPage=footer, onLaterPages=footer)


def add_extension_pages(r, extension):
    """Read the completed extension artifacts; never fill absent metrics."""
    sensitivity = extension["sensitivity"]
    cnn = extension["cnn"]
    card = extension["model_card"]
    verification = extension["verification"]
    inference = extension["inference"]
    if card["selected_candidate"] != "cnn_1d" or inference["input_target_required"]:
        raise ValueError("Review final-model and target-free-inference narrative.")
    candidates = extension["sensitivity_summary"]
    chosen = sensitivity["selected_config"]
    test = sensitivity["test_metrics"]["cycles"]
    cnn_test = cnn["test_metrics"]["cycles"]
    policy_ko = "비음수" if chosen["nonnegative"] else "원시"
    policy_en = "nonnegative" if chosen["nonnegative"] else "raw"
    selected_name = f"{LABELS[chosen['model']]} {chosen['window']}/{chosen['stride']}"
    if len(candidates) != 24:
        raise ValueError("All 24 completed sensitivity candidates are required.")

    r.page("윈도와 예측 정책 민감도", "Window and Prediction Policy Sensitivity")
    r.para("6가지 윈도/stride, Ridge와 HGB, 원시·비음수 정책의 24개 후보를 개발 엔진에서 비교했다. 비음수 정책은 max(0, 윈도 예측)을 먼저 적용하고 이후 비행 평균을 계산한다. 엔진 폴드는 원래 60/60 실험과 같게 고정했다. 모든 설정은 같은 446개 개발 비행 사이클을 포함한다. 개발 CV가 끝난 뒤 선택한 후보만 재적합하고 시험 자료를 읽었다.",
           "Six window/stride settings, Ridge and HGB, and raw/nonnegative policies yielded 24 development candidates. The nonnegative policy applies max(0, window prediction) before flight averaging. Engine folds were anchored to the original 60/60 experiment. Every setting covers the same 446 development cycles. Only the development-selected candidate was refitted and then evaluated on test data.")
    r.heatmap(candidates)
    r.para("개발 macro engine cycle RMSE이며 *는 선택 후보다. 회귀모델은 36회 CV 적합하고 같은 원시 예측에 두 정책을 평가했다. 선택은 반올림 전 수치를 사용했다. 모든 MAE·순위·엔진별 점수는 CSV에 보존했다.",
           "Development macro engine cycle RMSE; * marks selection. Thirty-six CV model fits each support both prediction policies. Selection uses unrounded values; complete MAE, ranks and engine scores remain in CSV files.", r.small)
    r.para("그림 4. 24개 후보의 개발 RMSE 히트맵. 선택된 셀은 주황 테두리로 표시한다. 확장 자체는 최초 시험 열람 이후 설계했으므로 이후 시험 점수는 탐색적 비교다.",
           "Figure 4. Development RMSE heatmap for all 24 candidates; orange outlines the selected cell. Extension design followed original test inspection, so subsequent test scores remain exploratory.",r.small)
    r.metrics([(f"{sensitivity['cv_macro_engine_cycle_rmse']:.3f}",r.tr("선택 후보 개발 RMSE","Selected CV RMSE"),r.tr("단위: 사이클","Unit: cycles")),
               ("30 / 15",r.tr("윈도 / stride","Window / stride"),r.tr("Ridge · 비음수 정책","Ridge · nonnegative")),
               ("0.087",r.tr("6개 설정의 RMSE 범위","RMSE span, six settings"),r.tr("비음수 Ridge · 사이클","Nonnegative Ridge · cycles"))])
    r.head("선택 결과", "Selected configuration")
    r.para(f"{selected_name}, {policy_ko} 정책이 선택됐다. 개발 RMSE {sensitivity['cv_macro_engine_cycle_rmse']:.3f}, MAE {sensitivity['cv_macro_engine_cycle_mae']:.3f}; 재사용 시험 RMSE {test['macro_engine_rmse']:.3f}, MAE {test['macro_engine_mae']:.3f} 사이클이다. 비음수 Ridge의 6가지 윈도 설정 간 CV RMSE 범위는 약 0.087 사이클로 작다. 이 작은 그리드의 최저점을 보편적인 최적 설정으로 단정하지 않는다.",
           f"Selected: {selected_name}, {policy_en}. Development RMSE/MAE: {sensitivity['cv_macro_engine_cycle_rmse']:.3f}/{sensitivity['cv_macro_engine_cycle_mae']:.3f}; reused-test RMSE/MAE: {test['macro_engine_rmse']:.3f}/{test['macro_engine_mae']:.3f} cycles. Across six window settings, nonnegative Ridge CV RMSE spans only about 0.087 cycles. The minimum on this small grid is not a universal optimum.")

    r.page("원시 센서 시계열 1D CNN", "1D CNN on Raw Sensor Windows")
    config,runtime = cnn["config"],cnn["runtime"]
    r.para("CNN은 90개 요약 특징 대신 18채널 × 60샘플 원시 윈도를 입력으로 받는다. 학습 폴드에서 엔진·사이클 가중 채널 평균과 표준편차를 계산한다. 3개 개발 엔진 폴드와 개발 전체 재적합마다 고정 12 epoch를 학습했다. 검증·시험 점수로 학습 길이나 하이퍼파라미터를 조정하지 않았다.",
           "The CNN receives raw 18-channel × 60-sample windows instead of 90 summary features. Channel mean and standard deviation use engine/cycle weights from the current training partition. Three development folds and a full-development refit each run a fixed 12 epochs. Validation or test scores do not tune the training duration or hyperparameters.")
    r.architecture()
    r.table(r.tr(["항목", "실제 설정"],["Item","Actual setting"]),[
        [r.tr("학습","Training"),f"{config['optimizer']}; epochs={config['epochs']}; batch={config['batch_size']}; lr={config['learning_rate']}; target/{config['target_scale']}; seed={config['seed']}"],
        [r.tr("장치와 규모","Device and size"),f"{runtime['device_name']}; {runtime['parameter_count']:,} parameters; PyTorch {runtime['torch_version']}"],
        [r.tr("실행 시간","Measured time"),r.tr(f"전체 실행 {runtime['total_seconds']:.2f}s; 전체 개발 적합 {runtime['fit_seconds_full_dev']:.2f}s; 시험 예측 {runtime['predict_seconds_test']:.2f}s",f"Total {runtime['total_seconds']:.2f}s; full-dev fit {runtime['fit_seconds_full_dev']:.2f}s; test prediction {runtime['predict_seconds_test']:.2f}s")],
    ],[31,143])
    r.para("Softplus 출력에 100을 곱해 RUL을 복원하므로 예측은 비음수다. 이는 물리적 범위의 한 제약일 뿐 정확도나 운용 적합성을 보장하지 않는다. 최적화 손실은 target/100의 엔진·사이클 가중 MSE다. 장치 간 완전한 수치 동일성을 주장하지 않는다.",
           "Softplus outputs are multiplied by 100 to recover nonnegative RUL. This range constraint does not establish accuracy or operational suitability. Optimization uses engine/cycle-weighted MSE on target/100. Exact numerical equality across hardware is not claimed.",r.small)
    r.figure(ROOT/"output/extended/cnn/training_history.png",max_height=152)
    r.para("그림 5. 학습 미니배치의 가중 RMSE 이력. 검증 곡선이나 조기 종료 기준이 아니다.",
           "Figure 5. Weighted RMSE from training minibatches; not a validation curve or an early-stopping criterion.",r.small)
    r.table(r.tr(["엔진", "사이클", "CNN MAE", "CNN RMSE"],["Engine","Cycles","CNN MAE","CNN RMSE"]),
            [[row["unit"],row["n_cycles"],f"{row['mae']:.3f}",f"{row['rmse']:.3f}"] for row in cnn_test["per_engine"]],[30,40,52,52])
    r.para(f"CNN 개발 macro RMSE는 {cnn['cv_macro_engine_cycle_rmse']:.3f}, MAE는 {cnn['cv_macro_engine_cycle_mae']:.3f}다. 재사용 시험 macro RMSE는 {cnn_test['macro_engine_rmse']:.3f}, MAE는 {cnn_test['macro_engine_mae']:.3f} 사이클이다. 전체 후보의 개발 CV RMSE가 가장 낮아 최종 추론 모델로 선택했다.",
           f"CNN development macro RMSE was {cnn['cv_macro_engine_cycle_rmse']:.3f} and MAE {cnn['cv_macro_engine_cycle_mae']:.3f}. Reused-test macro RMSE was {cnn_test['macro_engine_rmse']:.3f} and MAE {cnn_test['macro_engine_mae']:.3f} cycles. Its lowest development CV RMSE across candidates selected it as the final inference model.")

    r.page("확장 결과 비교", "Extension Comparison")
    baseline=read_json(ROOT/"output/results/metrics.json")
    baseline_cv=next(x for x in baseline["cv_summary"] if x["model"]=="ridge")
    baseline_test=baseline["test_metrics"]["ridge"]["cycles"]
    r.comparison([
        ("Ridge 60/60 raw",baseline_cv['macro_engine_cycle_rmse'],baseline_test['macro_engine_rmse']),
        (selected_name+" "+r.tr(policy_ko,policy_en),sensitivity['cv_macro_engine_cycle_rmse'],test['macro_engine_rmse']),
        ("1D CNN 60/60",cnn['cv_macro_engine_cycle_rmse'],cnn_test['macro_engine_rmse'])],height=164)
    r.para("같은 시험 엔진의 재평가 결과를 나란히 제시한다. 정답을 본 뒤 확장 범위를 결정했으므로 이 그림의 차이는 새 독립 시험에서 확인된 개선으로 해석할 수 없다. 윈도 설정·모델·정책의 선택에는 개발 점수만 사용했다.",
           "These are repeated evaluations on the same test engines. Because extension scope was chosen after seeing original targets and results, differences are not improvements confirmed on a fresh independent test. Window/model/policy selection used development scores only.",r.small)
    r.head("202개 사이클의 예측 진단", "Prediction diagnostics across 202 cycles")
    r.diagnostic(read_csv(ROOT/"output/extended/cnn/test_cycle_predictions.csv"))
    r.para("그림 6. 위: 개발/시험 macro RMSE. 아래: CNN 실제-예측 및 잔차(예측 - 정답) 산점도. 각 점은 비행 한 사이클이며 색은 엔진이다. 왼쪽 점선은 완전 예측, 오른쪽 수평선은 오차 0이다. 두 세로축의 단위는 사이클이다. 시험은 이미 열람한 자료이므로 새 독립 시험의 증거가 아니다.",
           "Figure 6. Top: development/test macro RMSE. Bottom: CNN actual-versus-predicted and residual (prediction minus target) plots. Each point is one flight cycle; color identifies the engine. The dashed diagonal denotes exact prediction; the horizontal line denotes zero error. Both vertical axes are in cycles. These are previously inspected test data, not fresh independent evidence.",r.small)
    r.para(f"재사용 시험 MAE: 원래 Ridge {baseline_test['macro_engine_mae']:.3f}, 비음수 Ridge {test['macro_engine_mae']:.3f}, CNN {cnn_test['macro_engine_mae']:.3f} 사이클. CNN은 개발 CV RMSE가 가장 낮아 선택됐으며 시험 순위가 선택 근거가 아니다.",
           f"Reused-test MAE: original Ridge {baseline_test['macro_engine_mae']:.3f}, nonnegative Ridge {test['macro_engine_mae']:.3f}, CNN {cnn_test['macro_engine_mae']:.3f} cycles. CNN was selected by the lowest development CV RMSE, not by test ranking.")
    r.page("오류 분석과 실행 검증", "Error Analysis and Execution Checks")
    audit=read_json(ROOT/"output/final/error_analysis.json")
    stages=read_csv(ROOT/"output/final/error_by_life_stage.csv")
    stage_order={"RUL <= 10":0,"10 < RUL <= 30":1,"RUL > 30":2}
    stages.sort(key=lambda row:stage_order[row["life_stage"]])
    r.para("정답 RUL로 수명 단계를 나누어 최종 CNN의 오차를 사후 분석했다. 이 구간은 모델 입력·선택 기준이 아니며 같은 시험 자료를 설명하기 위한 진단이다. 구간별 엔진 지표의 평균을 계산했고 각 구간에 시험 엔진 3대가 모두 포함된다.",
           "True RUL was used for a post-hoc life-stage audit of the final CNN. These bins are not model inputs or selection criteria; they diagnose the same reused test data. Metrics average engine-level values within each bin, with all three test engines represented.")
    r.life_stage_chart(stages)
    r.para("편향은 예측 - 정답이며 양수는 수명을 길게 추정한 것이다. 이 시험에서는 수명 초기(RUL > 30)의 오차가 말기보다 컸다. 엔진·구간의 표본 수가 작아 다른 운항 이력에 대한 보장으로 해석하지 않는다.",
           "Bias is prediction minus target; positive values overestimate remaining life. Error was larger earlier in life (RUL > 30) than near end of life in this test. Small engine/bin samples do not establish guarantees for other operating histories.")
    r.head("가장 큰 사이클 오차", "Largest cycle errors")
    worst=read_csv(ROOT/"output/final/worst_cycles.csv")[:5]
    r.table(r.tr(["엔진", "사이클", "정답", "예측", "절대 오차"],["Engine","Cycle","Target","Prediction","Abs. error"]),
            [[x["unit"],x["cycle"],f"{float(x['target']):.0f}",f"{float(x['prediction']):.3f}",f"{float(x['absolute_error']):.3f}"] for x in worst],[28,30,34,41,41])
    r.para(f"202개 사이클 중 음수 예측은 {audit['negative_predictions']}개, 최대 절대 오차는 {audit['maximum_absolute_error']:.3f} 사이클이었다. 비음수 범위는 보장하지만 개별 예측의 오차가 사라지는 것은 아니다. 원래 Ridge의 음수 문제와 CNN의 남은 오차를 함께 기록한다.",
           f"Across 202 cycles there were {audit['negative_predictions']} negative predictions; maximum absolute error was {audit['maximum_absolute_error']:.3f} cycles. Nonnegative outputs still have individual prediction errors. Both the original Ridge negatives and the remaining CNN errors are recorded.")
    r.head("저장 모델 재사용 검증", "Saved-model reuse checks")
    r.para(f"최종 CNN은 output/final/model.pt와 model_card.json에 고정했다. 가중치·채널 순서·학습 정규화와 모델 해시를 보존한다. 정답 Y를 읽지 않는 추론 경로에서 시험 윈도 {inference['n_windows']:,}개와 사이클 {inference['n_cycles']}개의 저장 예측이 재현됐다. 정답 배열을 제거한 별도 입력도 {verification['unlabelled_smoke_windows']}개 윈도·{verification['unlabelled_smoke_cycles']}개 사이클에서 확인했다. 추론에는 시험 정답이 필요하지 않다.",
           f"The final CNN is frozen in output/final/model.pt and model_card.json, retaining weights, channel order, training normalization and model hash. Target-free inference reproduced saved predictions for {inference['n_windows']:,} test windows and {inference['n_cycles']} cycles without reading Y. A separate input with the target array removed also passed for {verification['unlabelled_smoke_windows']} windows and {verification['unlabelled_smoke_cycles']} cycles. Inference does not require test labels.")
    r.head("실행과 산출물", "Execution and artifacts")
    r.para("전체 재실행은 ./tools/complete_project.ps1, 저장된 결과로 문서·제출 묶음 재생성은 ./tools/rebuild_delivery.ps1로 수행한다. 정답 없는 새 HDF5 입력은 ./run.ps1 -Stage predict -Dataset <경로>로 예측한다. CNN 체크포인트와 model_card, 윈도·사이클 예측, 출처·검증 JSON, 모든 CV 후보 및 원래 기준 결과를 보존한다.",
           "Use ./tools/complete_project.ps1 for the complete run and ./tools/rebuild_delivery.ps1 to rebuild documents/package from saved results. Predict a new target-free HDF5 file with ./run.ps1 -Stage predict -Dataset <path>. Retain the CNN checkpoint/card, window/cycle predictions, provenance/verification JSON, all CV candidates and original baseline results.")


def build(lang, output, font_dir, data):
    m, features, inspection, source, bench, summary, folds, extension = data
    r = Report(lang, output, font_dir)
    selected = m["selected_model"]
    if selected != "ridge":
        raise ValueError("Narrative requires review because the selected model changed.")
    cv = {row["model"]: row for row in m["cv_summary"]}
    tests = m["test_metrics"]
    ridge = tests[selected]["cycles"]
    baseline = tests["dummy_mean"]["cycles"]
    reduction = 100 * (1 - ridge["macro_engine_rmse"] / baseline["macro_engine_rmse"])
    fixed_chunk = features["config"]["chunk_rows"]
    scaling = [row for row in summary if int(row["chunk_rows"]) == fixed_chunk]
    fastest = min(summary, key=lambda row: float(row["median_seconds"]))
    one = next(row for row in scaling if int(row["workers"]) == 1)
    eight = next(row for row in scaling if int(row["workers"]) == 8)
    total_rows = sum(inspection["splits"][split]["rows"] for split in ("dev", "test"))
    test_log = (ROOT/"output/test_results.txt").read_text(encoding="utf-8")
    passed = re.search(r"(\d+) passed", test_log)
    if not passed:
        raise ValueError("A completed passing test log is required.")
    test_count = int(passed.group(1))

    # Page 1: completed work, measured findings, dataset scope and provenance.
    r.page("항공기 엔진 잔여수명 예측", "Aircraft Engine Remaining Useful Life Prediction")
    r.para("NASA N-CMAPSS DS02 프로젝트 보고서", "NASA N-CMAPSS DS02 Project Report", r.heading)
    r.para("김성현  |  학번 2021271250  |  DCCS 410  |  2026년 10월 2일", "김성현  |  Student ID 2021271250  |  DCCS 410  |  2 October 2026", r.small)
    r.head("목적과 완료 범위", "Objective and completed scope")
    r.para("대용량 HDF5 센서 시계열에서 엔진 잔여수명(RUL)을 예측하고 병렬 처리 비용을 평가했다. 10월 2일에 세 기준모델, 24개 윈도·정책 후보, 1D CNN과 저장 모델 추론까지 구현했다. 모든 수치는 저장된 실제 실험 결과다.",
           "The project predicts engine remaining useful life (RUL) from large HDF5 histories and measures parallel-processing costs. Three baselines, 24 window/policy candidates, a 1D CNN and saved-model inference were implemented on 2 October. All reported numbers come from actual saved experiments.")
    r.pipeline()
    final_cnn=extension['cnn']
    final_scores=final_cnn['test_metrics']['cycles']
    r.metrics([(f"{final_cnn['cv_macro_engine_cycle_rmse']:.3f}",r.tr("CNN 개발 CV RMSE","CNN development CV RMSE"),r.tr("사이클 · 최종 선택 기준","Cycles · model-selection criterion")),
               (f"{final_scores['macro_engine_rmse']:.3f}",r.tr("CNN 재사용 시험 RMSE","CNN reused-test RMSE"),r.tr("사이클 · 탐색적 비교","Cycles · exploratory comparison")),
               (f"{float(fastest['speedup_vs_one_worker']):.2f}×",r.tr("4개 작업자의 속도비","Four-worker speedup"),r.tr("캐시 준비 · 단일 컴퓨터","Warm cache · one computer"))])
    r.para(f"개발 CV로 원래 Ridge를 선택하고 확장 후 CNN을 선택했다. CNN의 재사용 시험 MAE는 {final_scores['macro_engine_mae']:.3f} 사이클이다. 시험을 열람한 뒤 확장을 설계했으므로 시험 점수 차이는 탐색적 비교다. 최단 전처리 중앙값은 {float(fastest['median_seconds']):.3f}초(청크 {int(fastest['chunk_rows']):,}행)이며 모든 실행의 특징 지문이 일치했다.",
           f"Development CV selected Ridge originally and CNN after extension. CNN reused-test MAE is {final_scores['macro_engine_mae']:.3f} cycles. Extension design followed test inspection, so test differences remain exploratory. The fastest preprocessing median was {float(fastest['median_seconds']):.3f} s ({int(fastest['chunk_rows']):,}-row chunks); every run had the same feature fingerprint.")
    r.head("검사한 데이터", "Inspected data")
    rows = []
    for split, ko, en in [("dev", "개발", "Development"), ("test", "시험", "Test")]:
        item = inspection["splits"][split]
        rows.append([r.tr(ko,en), ", ".join(map(str,item["unit_ids"])), f"{item['rows']:,}", str(item["segments"]), f"{features[split]['output_rows']:,}"])
    r.table(r.tr(["분할", "엔진 ID", "원시 행", "사이클", "윈도"], ["Split", "Engine IDs", "Raw rows", "Cycles", "Windows"]), rows, [25,48,37,27,37])
    r.para(f"총 {total_rows:,}행, 모의 엔진 9대. 개발 엔진은 모두 비행 클래스 3이다. 시험 엔진 11은 클래스 3, 14는 클래스 1, 15는 클래스 2다. 큰 행 수가 독립 엔진 표본 수를 늘려 주지는 않는다.",
           f"Total: {total_rows:,} rows from nine simulated engines. All development engines have flight class 3; test engines 11, 14 and 15 have classes 3, 1 and 2 respectively. A large row count does not increase the number of independent engines.")
    r.head("수집 출처", "Acquisition provenance")
    r.para(f"원출처는 NASA PCoE 항목 17 [1]과 N-CMAPSS 논문 [2]이다. 실제 분석 파일은 Figshare 제3자 미러 버전 1 [3]의 DS02이며 {source['bytes']:,} bytes다. 제공 MD5를 검증했다. NASA 전체 ZIP 15,760,443,389 bytes와 DS02 단독 파일 크기를 구별하며, NASA 배포본과의 바이트 동일성을 검증한 것은 아니다.",
           f"The original sources are NASA PCoE entry 17 [1] and the N-CMAPSS paper [2]. The analyzed DS02 file came from the third-party Figshare mirror v1 [3], containing {source['bytes']:,} bytes. Its published MD5 was verified. The NASA full ZIP is 15,760,443,389 bytes, not the DS02 file size; byte equivalence with NASA's archive was not independently established.", r.small)

    # Page 2: precise feature, fitting and evaluation rules.
    r.page("기준 실험 방법과 검증", "Baseline Methods and Validation")
    r.head("관측 입력과 특징", "Observable inputs and features")
    r.para("운항 조건 W 4개와 측정 센서 Xs 14개만 사용했다. 열화 변수 T, 건강 상태 hs, 가상 센서 Xv, 비행 클래스, 엔진·사이클 ID는 예측 입력에서 제외했다. ID는 경계·분할·집계에만 사용하고 Y는 정답으로만 사용했다.",
           "Inputs were four operating-condition channels W and fourteen measured-sensor channels Xs. Degradation variables T, health state hs, virtual sensors Xv, flight class, engine ID and cycle ID were excluded as predictors. IDs served only for boundaries, splits and aggregation; Y served only as the target.")
    r.window_diagram()
    r.para("윈도는 엔진·사이클 경계를 넘지 않지만 읽기 청크 경계의 상태는 보존한다. 표준편차는 ddof=0, 기울기는 샘플당 선형 기울기다. float64로 계산하고 float32로 저장했다.",
           "Windows never cross engine/cycle boundaries, while state persists across read chunks. Standard deviation uses ddof=0; slope is linear change per sample. Calculations use float64 and stored features use float32.",r.small)
    r.para(f"불완전한 꼬리 샘플은 개발 {features['dev']['incomplete_tail_samples']:,}개, 시험 {features['test']['incomplete_tail_samples']:,}개를 제외했다. 비유한 값 때문에 제외된 윈도는 두 분할 모두 0개다. 기본 4개 작업자 설정의 초기 특징 추출은 개발 {features['dev']['elapsed_seconds']:.3f}초, 시험 {features['test']['elapsed_seconds']:.3f}초였다. 이 단발 시간과 반복 벤치마크 중앙값은 구별한다.",
           f"Incomplete tails excluded {features['dev']['incomplete_tail_samples']:,} development and {features['test']['incomplete_tail_samples']:,} test samples. Neither split lost windows to non-finite values. Initial extraction with four workers took {features['dev']['elapsed_seconds']:.3f} s for development and {features['test']['elapsed_seconds']:.3f} s for test. These single-run times are distinct from repeated benchmark medians.")
    r.head("모델과 학습", "Models and fitting")
    r.table(r.tr(["모델", "고정 설정"], ["Model", "Fixed settings"]), [
        [r.tr("평균 기준모델", "Mean baseline"), r.tr("가중 학습 정답의 평균", "Weighted mean of fitting targets")],
        ["Ridge", "StandardScaler; alpha=10.0"],
        ["HGB", "max_iter=100; max_leaf_nodes=15; l2_regularization=1.0; early_stopping=False; seed=410"],
    ], [33,141])
    r.para("각 적합 자료 안에서 엔진의 총 가중치를 같게, 엔진 안에서는 사이클의 총 가중치를 같게 했다. 상대 윈도 가중치는 1 / (해당 사이클의 윈도 수 × 해당 엔진의 사이클 수)이며, 평균 1로 정규화했다. Ridge의 스케일러와 회귀를 모두 해당 학습 폴드에서만 가중 적합했다.",
           "Within each fitting partition, engines received equal total weight and cycles within each engine received equal total weight. Relative window weights were 1 / (windows in its cycle × cycles in its engine), normalized to mean 1. Ridge scaling and regression were both weighted and fitted on the current training fold only.")
    r.head("엔진 단위 교차검증", "Engine-disjoint cross-validation")
    r.fold_matrix(folds)
    r.para("GroupKFold 3개 폴드에서 각 개발 엔진은 한 번 검증에 사용됐다. 동일 엔진·사이클의 윈도 예측을 평균하여 비행 종료 시점 RUL을 얻었다. 사이클 정답의 일관성을 검사했다. 엔진별 사이클 MAE와 RMSE를 계산하고 엔진별 지표를 단순평균했다. 개발 macro RMSE가 가장 낮은 모델을 선택한 뒤 모든 후보를 개발 엔진 6대에 재적합하여 고정 시험 엔진 3대에 한 번 평가했다. 정답 변환과 예측값 clipping은 하지 않았다.",
           "Each development engine was held out once in three GroupKFold folds. Window predictions were averaged within each engine and flight cycle to estimate RUL at flight end; cycle targets were checked for consistency. Cycle-level MAE and RMSE were computed per engine, then averaged equally across engines. The lowest development macro RMSE selected the model. All candidates were refitted on six development engines and evaluated once on the fixed three-engine test split. Targets and predictions were not clipped or transformed.")

    # Page 3: all model results, not just the winner.
    r.page("보존한 기준 실험 결과", "Preserved Baseline Results")
    r.para("모든 오차의 단위는 남은 비행 사이클이다. 아래 표의 macro 지표는 각 엔진의 사이클 단위 오차 지표를 단순평균한 값이다. 개발 결과는 보지 않은 엔진 6대의 교차검증 예측, 시험 결과는 독립 엔진 3대의 202개 사이클에서 계산했다.",
           "Errors are in remaining flight cycles. Macro metrics below are unweighted means of engine-level cycle metrics. Development scores use out-of-fold predictions for all six engines; test scores use 202 cycles from three independent engines.")
    r.table(r.tr(["모델", "개발 MAE", "개발 RMSE", "시험 MAE", "시험 RMSE"], ["Model", "CV MAE", "CV RMSE", "Test MAE", "Test RMSE"]),
            [[LABELS[name] + (" *" if name == selected else ""), f"{cv[name]['macro_engine_cycle_mae']:.3f}", f"{cv[name]['macro_engine_cycle_rmse']:.3f}", f"{tests[name]['cycles']['macro_engine_mae']:.3f}", f"{tests[name]['cycles']['macro_engine_rmse']:.3f}"] for name in MODELS], [42,33,33,33,33])
    r.para("* 개발 교차검증으로 선택한 모델. HGB는 HistGradientBoostingRegressor다. 기준모델보다 낮은 시험 오차는 이 분할에서의 비교이며 통계적 유의성 또는 다른 엔진 집단에 대한 성능 보장은 아니다.",
           "* Model selected by development cross-validation. HGB denotes HistGradientBoostingRegressor. Improvement over the baseline is specific to this split; it is not a significance claim or a guarantee for other engine populations.", r.small)
    r.head("시험 엔진별 결과", "Per-engine test results")
    engine_rows=[]
    classes={11:3,14:1,15:2}
    for name in MODELS:
        for row in tests[name]["cycles"]["per_engine"]:
            engine_rows.append([LABELS[name], str(row["unit"]), str(classes[row["unit"]]), str(row["n_cycles"]), f"{row['mae']:.3f}", f"{row['rmse']:.3f}"])
    r.table(r.tr(["모델", "엔진", "클래스", "사이클", "MAE", "RMSE"], ["Model", "Engine", "Class", "Cycles", "MAE", "RMSE"]), engine_rows, [42,22,24,28,29,29])
    r.comparison([(LABELS[name],cv[name]['macro_engine_cycle_rmse'],tests[name]['cycles']['macro_engine_rmse']) for name in MODELS],height=149)
    r.para("그림 1. 비행 종료 시점 집계 후 개발·시험 RMSE. 모든 막대는 0에서 시작한다. 선택은 시험 순위와 관계없이 개발 교차검증에서 고정했다.",
           "Figure 1. Development/test RMSE after flight-end aggregation. All bars start at zero. Selection used development cross-validation, independently of test ranking.", r.small)
    delta = ridge["macro_engine_rmse"] - tests["hist_gradient_boosting"]["cycles"]["macro_engine_rmse"]
    r.para(f"Ridge의 개발 RMSE는 HGB보다 낮았으나 시험에서는 HGB가 {delta:.3f} 사이클 낮았다. 원래 기준 실험의 선택 규칙이나 설정을 시험 결과로 바꾸지 않았으며, 그 실험의 선택 모델은 Ridge로 보존한다.",
           f"Ridge had lower development RMSE, while HGB had {delta:.3f} cycles lower test RMSE. Test outcomes did not revise the original baseline's selection rules or settings; its selected model remains Ridge.")

    # Page 4: trajectories and a careful scope of inference.
    r.page("기준 모델 궤적과 한계", "Baseline Trajectories and Limits")
    r.figure(ROOT / "output/results/test_engine_rul_curves.png", max_height=385)
    r.para("그림 2. 개발 교차검증으로 선택한 Ridge의 시험 엔진별 RUL. 각 점은 같은 엔진·비행 사이클에 속한 윈도 예측의 평균이다. 실제 항공기 관측이 아닌 시뮬레이션 자료의 정답과 비교한다.",
           "Figure 2. Test-engine RUL from Ridge, selected by development cross-validation. Each point averages window predictions from the same engine and flight cycle. Targets come from simulation, not operational aircraft observations.", r.small)
    r.head("분포 변화와 오차", "Distribution shift and errors")
    r.para("개발 엔진 6대는 모두 클래스 3이므로 교차검증은 클래스 1과 2로의 일반화를 직접 검증하지 못했다. Ridge의 시험 RMSE는 엔진 11에서 11.417, 엔진 14에서 12.296, 엔진 15에서 10.116 사이클이었다. 클래스 1인 엔진 14의 오차가 가장 크지만 클래스별 시험 엔진이 한 대씩이므로 운항 클래스가 원인이라고 단정할 수 없다. 엔진별 열화 이력과 사이클 수 차이도 함께 존재한다.",
           "All six development engines belong to class 3, so cross-validation did not directly validate transfer to classes 1 and 2. Ridge test RMSE was 11.417 cycles for engine 11, 12.296 for engine 14 and 10.116 for engine 15. Class-1 engine 14 had the largest error, but one test engine per class cannot establish a causal effect of flight class. Engine degradation histories and cycle counts also differ.")
    r.head("지표를 읽는 범위", "Interpreting the metrics")
    r.para("비행 종료 시점의 평균 예측은 비행 전체의 샘플을 사용하므로 순간별 온라인 예측 성능을 뜻하지 않는다. 윈도 수가 많은 긴 비행의 영향이 점수를 지배하지 않도록 사이클·엔진 단위 집계를 사용했다. 원시 윈도 지표와 micro 지표도 metrics.json에 보관하지만 본 보고서의 모델 선택 및 핵심 결론은 macro 사이클 지표를 따른다.",
           "Flight-end averaging uses samples from the complete recorded flight and does not measure instantaneous online performance. Cycle and engine aggregation prevents longer flights with more windows from dominating the score. Raw-window and micro metrics are also retained in metrics.json; selection and the main conclusions here use macro cycle metrics.")
    predictions = [row for row in read_csv(ROOT / "output/results/test_cycle_predictions.csv") if row["model"] == selected]
    negative_count = sum(float(row["prediction"]) < 0 for row in predictions)
    r.para(f"제약 없는 Ridge는 세 시험 엔진의 수명 말기에 음의 RUL을 예측했다({len(predictions)}개 사이클 중 {negative_count}개). 원래 결과는 clipping하지 않고 보존했다. 이 관찰 후 추가한 비음수 정책은 확장 후보로 개발 교차검증에서 비교했다. 다만 정책 추가 자체가 시험 열람 이후였으므로 확장 시험 결과는 새로운 독립 일반화 증거가 아니다. 음수 RUL은 물리적으로 유효하지 않으며 원시 예측을 운항 판단에 사용할 수 없다.",
           f"Unconstrained Ridge produced negative RUL near end of life for all three test engines ({negative_count} of {len(predictions)} cycles). Original results remain unclipped. A nonnegative policy was subsequently added and compared by development cross-validation. Because that decision followed test inspection, extension test results are not new independent generalization evidence. Negative remaining life is physically invalid and raw predictions are not suitable for operational decisions.")

    # Page 5: every measured configuration, including unsuccessful scaling.
    r.page("HDF5 병렬 처리 실험", "Parallel HDF5 Processing Experiment")
    r.para(f"개발 자료 {features['dev']['input_rows']:,}행에서 동일한 {features['dev']['output_rows']:,}개 윈도를 추출했다. 작업자 수 1·2·4·8과 청크 크기 1,024·8,192·120,000행의 6가지 고유 설정을 각 {bench['repeats']}회 측정했다. 전체 자료 한 번으로 OS 캐시를 준비한 뒤 반복마다 설정 순서를 섞고 새 프로세스에서 실행했다.",
           f"Each run extracted the same {features['dev']['output_rows']:,} windows from {features['dev']['input_rows']:,} development rows. Six unique settings combined workers 1, 2, 4 and 8 with chunk sizes 1,024, 8,192 and 120,000 rows, with {bench['repeats']} measured repeats each. One complete pass warmed the OS cache; setting order was shuffled per repeat and each run used a fresh process.")
    bench_rows=[]
    for row in summary:
        bench_rows.append([row["workers"], f"{int(row['chunk_rows']):,}", f"{float(row['median_seconds']):.3f}", f"{float(row['min_seconds']):.3f}-{float(row['max_seconds']):.3f}", f"{float(row['speedup_vs_one_worker']):.3f}", f"{100*float(row['efficiency']):.1f}", f"{float(row['peak_rss_mib']):.1f}", f"{float(row['input_rows_per_second'])/1e6:.3f}"])
    r.table(r.tr(["작업자", "청크 행", "중앙값 s", "최소-최대 s", "속도비", "효율 %", "RSS MiB", "백만 행/s"], ["Workers", "Chunk rows", "Median s", "Min-max s", "Speedup", "Eff. %", "RSS MiB", "M rows/s"]), bench_rows, [17,24,22,32,20,19,20,20])
    r.para("속도비 = 1개 작업자·120,000행 청크 중앙값 / 해당 설정 중앙값. 효율 = 속도비 / 작업자 수. RSS는 반복 중 최대 프로세스 트리 합계다. 청크 실험의 속도비도 같은 기준을 사용하므로 청크 효과와 병렬 효과를 함께 포함한다.",
           "Speedup = median for one worker and 120,000-row chunks / setting median. Efficiency = speedup / worker count. RSS is the maximum sampled process-tree sum across repeats. Chunk experiments use the same baseline, so their speedup combines chunk and parallel effects.", r.small)
    r.benchmark_chart(summary,fixed_chunk)
    r.para("그림 3. 120,000행 청크. 점은 시간 중앙값, 선은 최소-최대이며 메모리는 최대 RSS다. 4개 작업자가 가장 빨랐지만 한 번은 2.285초였다. 메모리는 작업자 수와 함께 증가했다.",
           "Figure 3. At 120,000-row chunks, dots show median time and lines span min-max; memory is peak RSS. Four workers were fastest, with one 2.285 s run. Memory increased with worker count.", r.small)
    r.head("측정 범위와 해석", "Measurement scope and interpretation")
    r.para(f"1개 작업자 중앙값은 {float(one['median_seconds']):.3f}초, 8개는 {float(eight['median_seconds']):.3f}초이며 8개 작업자의 속도비는 {float(eight['speedup_vs_one_worker']):.3f}다. 작업자를 늘려 항상 빨라진다고 가정하지 않는다. 프로세스 시작, 작업 전달, 파일 읽기와 결과 조립 비용이 모두 측정에 포함되지만 이 실행만으로 개별 병목의 기여도를 분리하지는 못한다.",
           f"The one-worker median was {float(one['median_seconds']):.3f} s; eight workers took {float(eight['median_seconds']):.3f} s, giving speedup {float(eight['speedup_vs_one_worker']):.3f}. More workers are not assumed to be faster. Spawning, dispatch, reads and result assembly all contribute, but this experiment does not isolate their individual bottleneck costs.")
    r.para(f"시간에는 HDF5 검사·센서 읽기·작업자 시작(import 포함)·특징 계산·결과 조립을 포함하고 부모 프로세스의 최초 import·결과 해시·저장은 제외했다. 메모리는 100ms 간격의 프로세스 트리 RSS 합계로 공유 페이지가 중복될 수 있다. OS 캐시를 비우지 않았으므로 콜드 디스크 성능을 주장하지 않는다. 모든 실행의 순서 있는 열 스키마·자료형·행 지문이 정확히 일치했다. 코드 테스트 {test_count}개가 통과했다(output/test_results.txt).",
           f"Timing includes HDF5 scanning, sensor reads, worker startup (including imports), feature calculation and assembly. It excludes initial imports in the parent process, result hashing and saving. Memory is process-tree RSS sampled every 100 ms and may double-count shared pages. The OS cache was not flushed; these are not cold-disk results. Ordered schemas, dtypes and row fingerprints matched in every run. All {test_count} code tests passed (output/test_results.txt).", r.small)

    add_extension_pages(r, extension)

    # Final page: completed scope, genuine limitations and assignment deadlines.
    r.page("구현 완료와 재현", "Completed Scope and Reproducibility")
    r.head("해석의 한계", "Limits of the experiment")
    r.para("DS02는 모의 엔진 9대와 학습에 포함되지 않은 시험 엔진 3대뿐이다. 수백만 행이 엔진 간 일반화의 불확실성을 제거하지 않는다. 같은 시험을 재사용한 확장 비교는 탐색적이며, 선택에 쓴 개발 CV는 중첩 교차검증의 독립 성능 추정이 아니다. 클래스별 시험 엔진이 하나씩이므로 클래스의 인과 효과를 추정할 수 없다. 실제 항공기 운용이나 다중 노드 확장성을 주장하지 않는다.",
           "DS02 has nine simulated engines and three test engines excluded from fitting. Millions of rows do not eliminate engine-level uncertainty. Reused-test extension comparisons remain exploratory. Development CV used for selection is not an independent nested-CV performance estimate. One engine per test class cannot identify class effects. No operational-aircraft or multinode-scaling claim is made.")
    r.head("실행 기록과 과제 제출", "Execution record and assignment deadlines")
    r.para("구현·실험·추론·문서 생성은 10월 2일에 수행했다. 아래 남은 날짜는 제출·발표 마감이며 미구현 기능을 미룬 일정이 아니다. 시간대는 Asia/Seoul이다.",
           "Implementation, experiments, inference and document generation were performed on 2 October. Remaining dates below are submission/presentation deadlines, not deferred implementation. All use Asia/Seoul.", r.small)
    r.table(r.tr(["날짜", "항목"], ["Date", "Item"]), [
        ["10/02", r.tr("전체 구현, 실제 실험, 추론 및 한영 문서", "Implementation, actual experiments, inference and bilingual documents")],
        ["10/21", r.tr("한국어·영어 각 1쪽 제안서 제출", "Submit one-page proposals in Korean and English")],
        ["12/08", r.tr("5분 발표", "Five-minute presentation")],
        ["12/16 23:59", r.tr("보고서와 소스 코드 제출", "Submit report and source code")],
    ], [43,131])
    r.head("재현 기록", "Reproducibility record")
    r.para(f"환경: {bench['platform']}; 물리 코어 {bench['cpu_count_physical']}개, 논리 CPU {bench['cpu_count_logical']}개, RAM {bench['total_memory_bytes']/2**30:.1f} GiB. 설정은 configs/default.json, 원출처는 data/raw/source_manifest.json, 특징 검사는 data/processed/feature_manifest.json에 보관했다. 결과는 output/results/metrics.json, benchmark_summary.csv 및 benchmark_runs.csv에 기록했다. 모델·예측 CSV·그림도 함께 저장했다.",
           f"Environment: {bench['platform']}; {bench['cpu_count_physical']} physical cores, {bench['cpu_count_logical']} logical CPUs, {bench['total_memory_bytes']/2**30:.1f} GiB RAM. Configuration: configs/default.json. Provenance: data/raw/source_manifest.json. Feature audit: data/processed/feature_manifest.json. Results: output/results/metrics.json, benchmark_summary.csv and benchmark_runs.csv. Fitted models, prediction CSVs and figures are retained.", r.small)
    r.para("README와 requirements-tested.txt로 환경을 구성한 뒤 PowerShell에서 ./run.ps1 -Stage download, ./run.ps1 -Stage full, ./run.ps1 -Stage test를 실행한다. 저장 모델만 사용하려면 ./run.ps1 -Stage predict -Dataset <HDF5 경로>를 사용한다. 결과는 output/predictions에 저장한다. 시간은 하드웨어·캐시·부하에 따라 달라질 수 있다.",
           "Set up the environment using README and requirements-tested.txt. Run ./run.ps1 -Stage download, ./run.ps1 -Stage full, then ./run.ps1 -Stage test. For saved-model inference use ./run.ps1 -Stage predict -Dataset <HDF5 path>; outputs go to output/predictions. Timings vary with hardware, cache and load.", r.small)
    r.head("참고문헌과 데이터", "References and data")
    for text, url in [
        ("[1] NASA PCoE Data Set Repository. Entry 17: Turbofan Engine Degradation Simulation-2.", NASA),
        ("[2] Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2021). Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics. Data, 6(1), 5. doi:10.3390/data6010005.", PAPER),
        ("[3] Hao Li. N-CMAPSS DS02 third-party redistribution, Figshare version 1. doi:10.6084/m9.figshare.20436504.v1.", MIRROR),
    ]:
        r.story.append(Paragraph(f'<link href="{url}" color="black">{escape(text)}</link>', r.small))
    r.finish()


def build_overview(lang,font_dir,data):
    """Export a reusable, data-grounded overview PNG; vector PDF stays in QA."""
    extension=data[-1]; cnn=extension['cnn']; scores=cnn['test_metrics']['cycles']
    visual_dir=ROOT/'output/visuals'
    qa_dir=ROOT/'output/qa/reports/overview_sources'
    visual_dir.mkdir(parents=True,exist_ok=True)
    qa_dir.mkdir(parents=True,exist_ok=True)
    pdf=qa_dir/f'project_overview_{lang}.pdf'
    r=Report(lang,pdf,font_dir)
    d=Drawing(550,410)
    d.add(Rect(0,0,550,410,fillColor=colors.white,strokeColor=None))
    r.text(d,28,374,r.tr('엔진 잔여수명 예측','Engine Remaining Useful Life'),20,NAVY,True)
    r.text(d,28,351,'NASA N-CMAPSS DS02  |  6,517,190 '+r.tr('센서 행 · 모의 엔진 9대','sensor rows · 9 simulated engines'),10,GRAY)
    r.metrics([(f"{cnn['cv_macro_engine_cycle_rmse']:.3f}",r.tr('CNN 개발 CV RMSE','CNN development CV RMSE'),r.tr('사이클 · 선택 기준','Cycles · selection criterion')),
               (f"{scores['macro_engine_rmse']:.3f}",r.tr('CNN 재사용 시험 RMSE','CNN reused-test RMSE'),r.tr('사이클 · 탐색적 평가','Cycles · exploratory evaluation')),
               ('18 × 60',r.tr('CNN 원시 센서 윈도','CNN raw sensor window'),r.tr('채널 × 샘플','Channels × samples'))])
    metrics=r.story[-2]; metrics.translate(28,268); d.add(metrics)
    r.pipeline(); pipeline=r.story[-2]; pipeline.translate(28,174); d.add(pipeline)
    d.add(Rect(28,109,316,52,fillColor=PALE,strokeColor=None))
    d.add(Rect(356,109,166,52,fillColor=colors.HexColor('#FAF0E3'),strokeColor=None))
    r.text(d,40,143,r.tr('개발 6대: 2 · 5 · 10 · 16 · 18 · 20','6 development engines: 2 · 5 · 10 · 16 · 18 · 20'),9,NAVY,True)
    r.text(d,40,123,r.tr('3-fold 엔진 CV로 선택 → 6대 전체 재적합','Select by 3-fold engine CV → refit all six'),9,TEAL)
    r.text(d,368,143,r.tr('시험 3대: 11 · 14 · 15','3 test engines: 11 · 14 · 15'),9,NAVY,True)
    r.text(d,368,123,r.tr('학습·설정 선택에서 제외','Excluded from fit/selection'),8.5,ORANGE)
    notes=r.tr(['시험을 열람한 뒤 확장을 설계했다. 재사용 시험 결과는 탐색적 비교이며 새 독립 시험의 증거가 아니다.',
                '개발 엔진은 모두 비행 클래스 3; 시험은 클래스 3 / 1 / 2. 실항공기 운용 성능을 주장하지 않는다.'],
               ['Extension design followed test inspection. Reused-test scores are exploratory, not fresh independent evidence.',
                'All development engines are class 3; test classes are 3 / 1 / 2. No operational-aircraft claim is made.'])
    for i,line in enumerate(notes): r.text(d,28,78-i*17,line,8,GRAY)
    r.text(d,28,23,'김성현  2021271250  |  DCCS 410  |  2026-10-02',8,GRAY)
    renderPDF.drawToFile(d,str(pdf))
    poppler=Path(sys.executable).parents[1]/'native/poppler/Library/bin/pdftoppm.exe'
    subprocess.run([str(poppler),'-r','180','-singlefile','-png',str(pdf),str(visual_dir/f'project_overview_{lang}')],check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output")
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--lang", choices=("ko", "en"))
    args=parser.parse_args()
    if args.lang is None:
        for lang in ("ko", "en"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--lang", lang,
                            "--output-dir", str(args.output_dir), "--font-dir", str(args.font_dir)], check=True)
        return
    data=(read_json(ROOT/"output/results/metrics.json"),
          read_json(ROOT/"data/processed/feature_manifest.json"),
          read_json(ROOT/"output/data_inspection.json"),
          read_json(ROOT/"data/raw/source_manifest.json"),
          read_json(ROOT/"output/benchmark/benchmark_manifest.json"),
          read_csv(ROOT/"output/benchmark/benchmark_summary.csv"),
          read_csv(ROOT/"output/results/cv_fold_assignments.csv"),
          {"sensitivity": read_json(ROOT/"output/extended/sensitivity/metrics.json"),
           "sensitivity_summary": read_csv(ROOT/"output/extended/sensitivity/summary.csv"),
           "cnn": read_json(ROOT/"output/extended/cnn/metrics.json"),
           "model_card": read_json(ROOT/"output/final/model_card.json"),
           "verification": read_json(ROOT/"output/extended_verification.json"),
           "inference": read_json(ROOT/"output/predictions/prediction_manifest.json")})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for lang in (args.lang,):
        target=args.output_dir/f"report_{lang}.pdf"
        build(lang,target,args.font_dir,data)
        build_overview(lang,args.font_dir,data)
        print(target)


if __name__ == "__main__":
    main()
