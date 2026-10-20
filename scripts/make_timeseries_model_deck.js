#!/usr/bin/env node
/* 시계열 예측 모델 실증 결과 발표자료 — 금강 대청댐 · 낙동강 하구(부산).
 *
 * 수치는 전부 저장소 산출물에서 읽는다(합성값 없음):
 *   regions/<id>/models/gauge/evaluation.json   지점×예측일수 시험 성적(2024)
 *   regions/nakdong_busan/satellite/detection_eval.json  탐지 IoU(정답 라벨 대비, 로컬 재현)
 *   regions/<id>/satellite/area_series.json     공통 격자 면적 계열
 *   regions/<id>/gauge/series.json              지상 수위계 기간
 *   docs/presentation/data/timeseries_deck_data.json   API 에서 받아 둔 전망 예시·위성 수위 대조·백테스트 궤적
 *   docs/tta/TTA_제출문안_복붙용.md                  공식 지표(아래 상수, 문안과 같은 값)
 *
 * 실행:  NODE_PATH=<pptxgenjs 가 설치된 node_modules> PPTX_SKILL_DIR=<pptx 스킬 디렉터리> \
 *        node scripts/make_timeseries_model_deck.js [출력 경로]
 */
"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT = process.argv[2] || path.join(ROOT, "docs/presentation/시계열예측모델_실증결과_대청_부산.pptx");
const SKILL_DIR = process.env.PPTX_SKILL_DIR || "";
const J = (rel) => JSON.parse(fs.readFileSync(path.join(ROOT, rel), "utf8"));

// ───────────────────────────── 자료 읽기
const evalB = J("regions/nakdong_busan/models/gauge/evaluation.json");
const evalD = J("regions/geum_daecheong/models/gauge/evaluation.json");
const detB = J("regions/nakdong_busan/satellite/detection_eval.json");
const areaB = J("regions/nakdong_busan/satellite/area_series.json");
const areaD = J("regions/geum_daecheong/satellite/area_series.json");
const gaugeB = J("regions/nakdong_busan/gauge/series.json");
const gaugeD = J("regions/geum_daecheong/gauge/series.json");
const DATA = J("docs/presentation/data/timeseries_deck_data.json");

// 공식 지표(TTA 제출 문안 §판정표와 동일 값 — docs/tta/TTA_제출문안_복붙용.md)
const TTA = { d1: { level: 0.00094, area: 0.02663, total: 0.014 }, d7: { level: 0.00044, area: 0.02663, total: 0.014 }, target: 0.38, prev: 0.135 };
// 미래 강수 상한 실험(docs/presentation/과제성과_…md §32 · regions/*/models/gauge/experiments/future_rain.json 요약)
const FUTURE_RAIN = {
  "금강 대청댐": ["+10 / +13 / +13", "+19 / +21 / +21", "+30 / +31 / +32", "+30 / +32 / +35", "+32 / +32 / +40"],
  "낙동강 하구(부산)": ["+28 / +31 / +31", "+39 / +40 / +40", "+40 / +39 / +39", "+38 / +38 / +38", "+28 / +27 / +21"],
};

const MODEL_KO = {
  ensemble_mean: "계열 앙상블", mean_reversion: "평균 회귀 AR(1)", ridge_ar: "릿지 AR", ridge_arx: "릿지 AR+기상",
  seasonal_anomaly_ar: "계절편차 AR", analog_knn: "유사사례 k-NN", climatology: "계절 평년값", neural_gru: "GRU(GPU)",
  neural_lstm: "LSTM(GPU)", neural_mlp: "MLP(GPU)", neural_tcn: "TCN(GPU)", prophet_hybrid: "Prophet+잔차AR", persistence: "직전값 유지",
};
const TERRAIN_KO = { estuary: "하구", mainstem: "본류", tributary: "지류", reservoir: "저수지", channel: "수로" };
const cm = (m, d = 1) => (m * 100).toFixed(d);
const pct = (skill) => `${skill >= 0 ? "+" : ""}${(skill * 100).toFixed(1)} %`;
const row = (ev, st, h) => ev.rows.find((r) => r.station_id === st && r.horizon === h);
const stations = (ev) => [...new Set(ev.rows.map((r) => r.station_id))];
const HORIZONS = [1, 3, 5, 7, 10, 14, 30];
const ORDER_B = ["낙동강하구언(내)", "구포대교", "호포대교", "양산교", "삼랑진교", "정천교"];
const ORDER_D = ["대청댐", "대청조정지댐", "대청댐방수로"];
const nDays = (g, id) => g.stations.find((s) => s.id === id).n_days;

// ───────────────────────────── 테마·레이아웃
const THEME = {
  name: "WATERCAST Deep Water",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16202C", lt1: "FFFFFF", dk2: "0B2A45", lt2: "EEF3F8",
    accent1: "1F7A8C", accent2: "E07A2F", accent3: "4E9F6E", accent4: "C9A227", accent5: "6F5BA7", accent6: "B23A48",
    hlink: "1F7A8C", folHlink: "6F5BA7",
  },
};
const HX = THEME.colors;
const MUTED = "5E6B7A";
const LINE = "D9E1EA";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 × 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "수체·수위 시계열 예측 모델 실증 결과 — 금강 대청댐 · 낙동강 하구";
pres.author = "WATERCAST";
const C = pres.SchemeColor;
const FOOT = "WATERCAST · 수체·수위 시계열 예측 모델 실증 결과 (금강 대청댐 · 낙동강 하구) · 2026-10";

pres.defineSlideMaster({
  title: "LIGHT",
  background: { color: HX.lt1 },
  margin: [0.5, 0.6, 0.7, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.32, w: 12.1, h: 0.7, fontSize: 26, bold: true, color: HX.dk2, margin: 0, valign: "middle", align: "left" }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.05, w: 10, h: 0.3, fontSize: 9, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontSize: 9, color: MUTED, align: "right" },
});
pres.defineSlideMaster({
  title: "DARK",
  background: { color: HX.dk2 },
  margin: [0.5, 0.6, 0.7, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 2.3, w: 11.5, h: 1.3, fontSize: 40, bold: true, color: HX.lt1, margin: 0, valign: "middle", align: "left" }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.05, w: 10, h: 0.3, fontSize: 9, color: "9FB3C8", margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontSize: 9, color: "9FB3C8", align: "right" },
});

// ───────────────────────────── 도우미
function light(section) { return pres.addSlide({ masterName: "LIGHT", sectionTitle: section }); }
function dark(section) { return pres.addSlide({ masterName: "DARK", sectionTitle: section }); }
function title(slide, text) { slide.addText(text, { placeholder: "title" }); }
function subline(slide, text, y = 1.02) {
  slide.addText(text, { x: 0.6, y, w: 12.1, h: 0.4, fontSize: 13, italic: true, color: C.text1, margin: 0, isTextBox: true, objectName: "부제" });
}
function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.12, fill: { color: opts.fill || C.background2 },
    line: opts.line ? { color: opts.line, width: 0.75 } : { color: opts.fill || C.background2, width: 0 },
    shadow: opts.shadow ? { type: "outer", color: "16202C", blur: 6, offset: 2, angle: 90, opacity: 0.08 } : undefined,
    objectName: opts.name || "카드",
  });
}
function stat(slide, x, y, w, h, value, label, color, name, valueSize = 36) {
  card(slide, x, y, w, h, { fill: C.background2, name: name || `통계 ${label}` });
  slide.addText(value, { x: x + 0.2, y: y + 0.18, w: w - 0.4, h: h * 0.5, fontSize: valueSize, bold: true, color: color || C.accent1, margin: 0, valign: "middle", isTextBox: true, objectName: `${label} 값` });
  slide.addText(label, { x: x + 0.2, y: y + h * 0.5 + 0.1, w: w - 0.4, h: h * 0.5 - 0.25, fontSize: 11.5, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `${label} 설명` });
}
// 가로형 통계: 왼쪽 큰 숫자, 오른쪽 설명 (낮은 카드용)
function statRow(slide, x, y, w, h, value, label, color, name) {
  card(slide, x, y, w, h, { fill: C.background2, name: name || `통계 ${label}` });
  slide.addText(value, { x: x + 0.2, y, w: 2.2, h, fontSize: 24, bold: true, color: color || C.accent1, margin: 0, valign: "middle", isTextBox: true, objectName: `${label} 값` });
  slide.addText(label, { x: x + 2.45, y, w: w - 2.65, h, fontSize: 11, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `${label} 설명` });
}
const nDaysRange = (g) => { const v = g.stations.map((s) => s.n_days); return `${Math.min(...v).toLocaleString()}~${Math.max(...v).toLocaleString()}`; };
const MODEL_SHORT = {
  ensemble_mean: "앙상블", mean_reversion: "평균회귀", ridge_ar: "릿지AR", ridge_arx: "릿지AR+기상", seasonal_anomaly_ar: "계절편차AR",
  analog_knn: "k-NN", climatology: "평년값", neural_gru: "GRU", neural_lstm: "LSTM", neural_mlp: "MLP", neural_tcn: "TCN", prophet_hybrid: "Prophet",
};
function bullets(slide, items, x, y, w, h, opts = {}) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: opts.space || 6 } }));
  slide.addText(runs, { x, y, w, h, fontSize: opts.fontSize || 14, color: opts.color || C.text1, valign: "top", margin: 0, isTextBox: true, objectName: opts.name || "글머리" });
}
function note(slide, text, x = 0.6, y = 6.55, w = 12.1, h = 0.4) {
  slide.addText(text, { x, y, w, h, fontSize: 10.5, color: MUTED, margin: 0, valign: "top", isTextBox: true, objectName: "각주" });
}
function iconCircle(slide, x, y, d, label, color) {
  slide.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: color || C.accent1 }, line: { color: color || C.accent1, width: 0 }, objectName: `아이콘 ${label}` });
  slide.addText(label, { x, y, w: d, h: d, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `아이콘 글자 ${label}` });
}
const HEAD = { fill: { color: HX.dk2 }, color: HX.lt1, bold: true, fontSize: 11, valign: "middle", align: "center" };
function table(slide, rows, x, y, w, colW, opts = {}) {
  const body = rows.map((r, ri) => r.map((cell, ci) => {
    const base = ri === 0 ? { ...HEAD } : { fontSize: opts.fontSize || 10.5, color: HX.dk1, valign: "middle", align: ci === 0 ? "left" : "center", fill: { color: ri % 2 === 0 ? "F6F8FB" : "FFFFFF" } };
    if (typeof cell === "object" && cell !== null && "text" in cell) return { text: cell.text, options: { ...base, ...(cell.options || {}) } };
    return { text: String(cell), options: base };
  }));
  slide.addTable(body, { x, y, w, colW, border: { type: "solid", color: LINE, pt: 0.5 }, rowH: opts.rowH || 0.32, margin: 0.05, fontFace: THEME.bodyFontFace, autoPage: false, objectName: opts.name || "표" });
}
const CHART_BASE = {
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, valGridLine: { color: "E2E7EE", size: 0.5 }, catGridLine: { style: "none" },
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 9, dataLabelColor: "16202C", legendFontFace: "+mn-lt", legendFontSize: 10, legendColor: MUTED,
  titleFontFace: "+mj-lt", titleFontSize: 12, titleColor: "0B2A45", showTitle: true, valAxisTitleFontFace: "+mn-lt", valAxisTitleFontSize: 9, valAxisTitleColor: MUTED,
};

// 폴드(학습·검증·시험) 그림 — 막대로 그린다
function drawFolds(slide, x, y, w, trainEnd, valYears, testYear, first = 2015, label) {
  const last = testYear + 1;
  const span = last - first;
  const px = (yr) => x + ((yr - first) / span) * w;
  const rowH = 0.26, gap = 0.06;
  slide.addText(label, { x, y: y - 0.32, w, h: 0.28, fontSize: 11, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: `폴드 제목 ${label}` });
  const rows = valYears.map((vy, i) => ({ text: `폴드 ${i + 1}: ≤${vy - 1} 학습 → ${vy} 검증`, trainTo: vy, val: vy }));
  rows.push({ text: `최종: ≤${testYear - 1} 재적합 → ${testYear} 시험`, trainTo: testYear, val: testYear, test: true });
  rows.forEach((r, i) => {
    const yy = y + i * (rowH + gap);
    slide.addShape(pres.ShapeType.rect, { x, y: yy, w: px(r.trainTo) - x, h: rowH, fill: { color: "D6DEE8" }, line: { color: "D6DEE8", width: 0 }, objectName: `학습 ${r.text}` });
    slide.addShape(pres.ShapeType.rect, { x: px(r.val), y: yy, w: px(r.val + 1) - px(r.val), h: rowH, fill: { color: r.test ? HX.accent4 : HX.accent1 }, line: { color: r.test ? HX.accent4 : HX.accent1, width: 0 }, objectName: `검증 ${r.text}` });
    slide.addText(r.text, { x: x + 0.06, y: yy, w: px(r.trainTo) - x - 0.1, h: rowH, fontSize: 9, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `폴드 글 ${r.text}` });
  });
  const axisY = y + rows.length * (rowH + gap) + 0.02;
  for (let yr = first; yr <= last; yr++) {
    slide.addText(String(yr), { x: px(yr) - 0.25, y: axisY, w: 0.5, h: 0.22, fontSize: 8, color: MUTED, align: "center", margin: 0, isTextBox: true, objectName: `연도 ${yr}` });
  }
  return axisY + 0.25;
}

// ───────────────────────────── 1. 표지
pres.addSection({ title: "개요" });
{
  const s = dark("개요");
  s.addText("수체·수위 시계열 예측 모델", { x: 0.9, y: 1.45, w: 11.5, h: 0.6, fontSize: 18, color: "9FB3C8", margin: 0, isTextBox: true, objectName: "상단 라벨" });
  title(s, "초소형위성 영상과 지상 수위계 10년 자료로\n앞으로의 수위를 내다보는 모델 — 실증 결과");
  s.addText("금강 대청댐 (대전·청주)  ·  낙동강 하구 (부산·김해)", { x: 0.9, y: 3.85, w: 11.5, h: 0.5, fontSize: 20, color: C.background1, margin: 0, isTextBox: true, objectName: "지역" });
  s.addText("2026년 10월 16일  ·  모든 수치는 저장소 산출물(학습·평가 기록, 인수 검증 리포트, 위성 수위 대조) 기준 — 합성값 없음 · 탐지 대표값은 서울시립대 공식(시험 장면)과 동일", { x: 0.9, y: 4.5, w: 11.5, h: 0.4, fontSize: 13, color: "9FB3C8", margin: 0, isTextBox: true, objectName: "날짜" });
  // 모티프: 물결 원 3개
  [[10.3, 5.3, 1.1, HX.accent1], [11.2, 5.0, 0.8, HX.accent4], [11.9, 5.55, 0.55, HX.accent2]].forEach(([x, y, d, c], i) => s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: c }, line: { color: c, width: 0 }, objectName: `장식 원 ${i + 1}` }));
  s.addNotes("표지. 수치 출처: regions/*/models/gauge/evaluation.json, regions/nakdong_busan/satellite/detection_eval.json, docs/presentation/data/timeseries_deck_data.json, docs/tta/TTA_제출문안_복붙용.md");
}

// ───────────────────────────── 2. 한 장 요약
{
  const s = light("개요");
  title(s, "한 장 요약 — 무엇을 했고, 얼마나 맞았나");
  subline(s, "시험 연도 2024는 학습·모델 선택에 쓰지 않은 해다. 비교 기준은 '직전값 유지'(오늘 수위가 내일도 같다고 보는 가장 단순한 예측).");
  const cells = `${evalB.cells_beating_persistence + evalD.cells_beating_persistence} / ${evalB.cells_total + evalD.cells_total}`;
  const tsS = detB.sensors.ICEYE.test_scene, tsO = detB.sensors.PlanetScope.test_scene;
  const gupo = DATA["busan_levels:구포대교"];
  stat(s, 0.6, 1.6, 2.95, 1.75, cells, `지점×예측일수 셀 중 직전값 유지보다 나은 셀 (부산 ${evalB.cells_beating_persistence}/${evalB.cells_total} · 대청 ${evalD.cells_beating_persistence}/${evalD.cells_total})`, C.accent1, "통계 셀");
  stat(s, 3.65, 1.6, 2.95, 1.75, `${tsS.f1.toFixed(3)} / ${tsO.f1.toFixed(3)}`, `부산 수체 탐지 F1 — 시험 장면(학습 제외) 레이더 ${tsS.f1.toFixed(4)}(IoU ${tsS.water_iou.toFixed(4)}) · 광학 ${tsO.f1.toFixed(4)}(IoU ${tsO.water_iou.toFixed(4)}), 목표 F1 ≥ 0.91`, C.accent3, "통계 IoU", 28);
  stat(s, 6.7, 1.6, 2.95, 1.75, "3.4 cm", "위성 수위 vs 지상 수위계 RMSE — 부산 구포대교 8회(8월 인수 LSTM 번들) · 대청댐 35.9 cm (9회, 관측 대조)", C.accent5, "통계 수위");
  stat(s, 9.75, 1.6, 2.95, 1.75, String(TTA.d1.total), `공식 지표 종합 정규화 MSE (D+1·D+7 모두) — 목표 ${TTA.target} 이하`, C.accent2, "통계 TTA");
  bullets(s, [
    "위성 영상 한 장 → 수체 탐지 → 면적·위성 수위 산출은 '관측' 경로, 지상 수위계 10년 + 기상청 ASOS 자료로 학습한 13종 후보 모델이 '예측' 경로다.",
    "모델은 지점×예측일수(1·3·5·7·10·14·30일)마다 검증 연도별 폴드에서 가장 나은 후보를 자동으로 고른다 — 직전값 유지는 기준선이라 선택 대상에서 뺐다.",
    `대표 성적(시험 2024, D+7): 부산 구포대교 RMSE ${cm(row(evalB, "구포대교", 7).test_rmse)} cm (${pct(row(evalB, "구포대교", 7).skill_vs_persistence)}), 대청조정지댐 ${cm(row(evalD, "대청조정지댐", 7).test_rmse)} cm (${pct(row(evalD, "대청조정지댐", 7).skill_vs_persistence)}), 대청댐 본댐 ${cm(row(evalD, "대청댐", 7).test_rmse)} cm (${pct(row(evalD, "대청댐", 7).skill_vs_persistence)}).`,
    "한계도 같이 본다: 부산 위성 표본은 2020년 2~4월 갈수기뿐, 대청은 정답 라벨이 없어 탐지 정확도를 검증할 수 없고, 저수지 본댐은 방류 운영 때문에 장기 오차가 1 m 를 넘는다.",
  ], 0.6, 3.65, 12.1, 2.9, { fontSize: 13.5, space: 8 });
  s.addNotes(`셀 수: evaluation.json cells_beating_persistence/cells_total. IoU: detection_eval.json pooled. 위성 수위 대조: timeseries_deck_data.json busan_levels/dcd_levels (API /satellite/levels comparison). 공식 지표: TTA 문안. 구포대교 대조 n=${gupo.length}.`);
}

// ───────────────────────────── 3. 흐름
{
  const s = light("개요");
  title(s, "전체 흐름 — 위성은 '관측'을, 수위계와 기상은 '예측'을 맡는다");
  subline(s, "한 장면이 들어오면 ①→③ 이 돌고, ④⑤ 는 지상 자료만으로 돌아간다. 위성 수위는 ⑤ 의 입력이 아니라 ③ 의 결과를 지상 실측과 맞춰 보는 대조 자료다.");
  const steps = [
    ["1", "위성 영상", "ICEYE·Umbra SAR, PlanetScope 광학 전처리 GeoTIFF"],
    ["2", "수체 탐지", "WBMS U-Net+ASPP(SAR) · Attention U-Net(광학) → 물/비물 마스크"],
    ["3", "면적·위성 수위", "센서별 공통 격자 면적, 마스크 경계 화소 × DEM → 지점 수위"],
    ["4", "지상 자료", "WAMIS 일 수위(10년) + 기상청 ASOS 강수·기온·습도"],
    ["5", "시계열 예측", "13종 후보 → 지점×예측일수 자동 선택 → D+1~30 전망"],
    ["6", "평가", "시험 연도 2024 RMSE·MAE, 직전값 유지 대비 개선율"],
  ];
  const w = 1.9, gap = 0.14, x0 = 0.6, y0 = 1.75, h = 2.3;
  steps.forEach(([n, head, desc], i) => {
    const x = x0 + i * (w + gap);
    const obs = i < 3;
    card(s, x, y0, w, h, { fill: obs ? C.background2 : "E6F1F3", name: `단계 ${n}` });
    iconCircle(s, x + 0.15, y0 + 0.15, 0.45, n, obs ? C.accent5 : C.accent1);
    s.addText(head, { x: x + 0.15, y: y0 + 0.72, w: w - 0.3, h: 0.4, fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: `단계 제목 ${n}` });
    s.addText(desc, { x: x + 0.15, y: y0 + 1.15, w: w - 0.3, h: 1.05, fontSize: 11, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `단계 설명 ${n}` });
    if (i < steps.length - 1) s.addShape(pres.ShapeType.rightArrow, { x: x + w - 0.02, y: y0 + h / 2 - 0.1, w: gap + 0.04, h: 0.2, fill: { color: "B9C6D3" }, line: { color: "B9C6D3", width: 0 }, objectName: `화살표 ${n}` });
  });
  s.addText("관측 경로 (①~③)", { x: 0.6, y: 4.25, w: 6.0, h: 0.35, fontSize: 12, bold: true, color: C.accent5, margin: 0, isTextBox: true, objectName: "관측 라벨" });
  s.addText("예측 경로 (④~⑥)", { x: 6.72, y: 4.25, w: 6.0, h: 0.35, fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true, objectName: "예측 라벨" });
  bullets(s, [
    "탐지 정확도(IoU)는 정답 라벨이 있는 부산에서만 잰다. 대청은 라벨이 없어 면적·위성 수위 대조로만 실증한다.",
    "예측 모델의 성적표는 전부 지상 수위계 일 수위 기준이다 — 위성 면적 계열(지역당 4~10개 관측)은 모델을 가릴 표본이 못 된다.",
    "같은 규칙을 어느 지역에도 적용한다: 수위계 자료만 있어도 모델이 만들어지고, 위성 장면이 등록되면 탐지·대조 기능이 열린다.",
  ], 0.6, 4.75, 12.1, 1.8, { fontSize: 13 });
  s.addNotes("흐름 설명. 탐지 모델 이름은 regions/nakdong_busan/satellite/detection_eval.json 의 model 경로(WBMS_SAR_ICEYE.h5, WBMS_Optic_PlanetScope.h5) 기준.");
}

// ───────────────────────────── 4. 실증 데이터
pres.addSection({ title: "데이터" });
{
  const s = light("데이터");
  title(s, "실증 데이터 — 두 지역이 가진 것과 못 가진 것");
  subline(s, "로컬 저장소 regions/ 내장 자료 기준. 위성 장면 날짜는 전처리 영상의 촬영일.");
  const bScenes = areaB.sensors.ICEYE.rows.length + areaB.sensors.PlanetScope.rows.length;
  const rows = [
    ["항목", "금강 대청댐 (대전·청주)", "낙동강 하구 (부산·김해)"],
    ["위성 장면", `PlanetScope 광학 ${areaD.sensors.PlanetScope.rows.length}장 — 2019-03-23 · 03-29 · 04-03 · 04-04 · 04-27, 2022-05-14 · 05-16 · 05-17 · 05-22 · 05-23`, `ICEYE SAR 4장 (2020-03-02 · 03-30 · 04-15 · 04-16) + PlanetScope 광학 4장 (2020-02-18 · 03-12 · 03-25 · 04-14) = ${bScenes}장`],
    ["정답 라벨(GT)", "없음 → 탐지 정확도(IoU) 산출 불가", "SAR 4 + 광학 4 (수체 이진 라벨) → IoU·F1 산출. 시험 장면(학습 제외)은 SAR 04-16 · 광학 03-25 각 1장"],
    ["탐지 모델 패키지", "부산과 공용 (광학 모델)", "서울시립대 인수 패키지 wbms:0.13 (2026-08-20 수령·재동결 08-18). 시립대 납품본 wbms:v1(09-28)은 미수령 — 수령 후 교체"],
    ["수위 산정 지점", "3곳 × 10시점 — 대청댐 본댐 · 대청조정지댐 · 대청댐방수로 (대청댐만 수위대 상수 보정)", "4곳 × 8시점 — 정천교 · 호포대교 · 김해월촌리 · 구포대교 (융합 LSTM v3, 8월 인수 번들)"],
    ["지상 수위계 (WAMIS)", `3지점, ${gaugeD.fetched_range[0]} ~ ${gaugeD.fetched_range[1]} (지점당 ${nDays(gaugeD, "대청댐").toLocaleString()}일) — 2016~2018 결측이라 2019년부터`, `6지점, ${gaugeB.fetched_range[0]} ~ ${gaugeB.fetched_range[1]} (지점당 ${nDaysRange(gaugeB)}일, 품질 관리 전)`],
    ["기상 (기상청 ASOS 일자료)", "대전 133 — 강수 · 평균/최저/최고 기온 · 습도, 2015~2024", "부산 159 — 같은 항목, 2015~2024"],
    ["예측 모델 셀", `${evalD.n_stations}지점 × 7예측일수 = ${evalD.cells_total}셀`, `${evalB.n_stations}지점 × 7예측일수 = ${evalB.cells_total}셀`],
    ["학습 / 검증 / 시험", `≤${evalD.split.train_end.slice(0, 4)} / ${evalD.split.val_start.slice(0, 4)}~${evalD.split.val_end.slice(0, 4)} 연도별 폴드 / ${evalD.split.test_start.slice(0, 4)}`, `≤${evalB.split.train_end.slice(0, 4)} / ${evalB.split.val_start.slice(0, 4)}~${evalB.split.val_end.slice(0, 4)} 연도별 폴드 / ${evalB.split.test_start.slice(0, 4)}`],
  ];
  table(s, rows, 0.6, 1.5, 12.1, [2.1, 5.0, 5.0], { rowH: 0.5, fontSize: 10, name: "데이터 현황표" });
  note(s, "대청 수위 산정 중 대청조정지댐·방수로는 보정 상수가 없어 실측 대조표에 넣지 않았다. 부산 '김해월촌리'(2022640)는 WAMIS 개방 API 에 일 수위가 없어 대조에서 제외.", 0.6, 6.62, 12.1, 0.4);
  s.addNotes("출처: regions/*/region.json, gauge/series.json(stations.n_days, fetched_range), weather/asos_daily.json, satellite/area_series.json, models/gauge/evaluation.json(split, n_stations, cells_total).");
}

// ───────────────────────────── 5. 탐지 결과 부산
pres.addSection({ title: "관측: 탐지·위성 수위" });
{
  const s = light("관측: 탐지·위성 수위");
  title(s, "수체 탐지 ① 낙동강 하구 — 시험 장면 F1 0.935 · 0.974 (목표 0.91 충족)");
  subline(s, "대표값은 학습에서 제외한 시험 장면 1건(레이더 04-16 · 광학 03-25)의 정답 라벨 대비 값 — 서울시립대 공식 수치와 동일. 나머지 장면은 학습 장면이라 '탑재 후 전 장면 재현'으로만 본다.");
  const sar = detB.sensors.ICEYE, opt = detB.sensors.PlanetScope;
  const scenes = [...sar.per_scene.map((p) => ({ ...p, sensor: "SAR" })), ...opt.per_scene.map((p) => ({ ...p, sensor: "광학" }))].sort((a, b) => a.date.localeCompare(b.date));
  const lab = (p) => `${p.sensor} ${p.date.slice(5)}${p.role === "test" ? " ★시험" : ""}`;
  s.addChart(pres.ChartType.bar, [
    { name: "ICEYE SAR", labels: scenes.map(lab), values: scenes.map((p) => (p.sensor === "SAR" ? +p.water_iou.toFixed(3) : null)) },
    { name: "PlanetScope 광학", labels: scenes.map(lab), values: scenes.map((p) => (p.sensor === "광학" ? +p.water_iou.toFixed(3) : null)) },
  ], {
    ...CHART_BASE, x: 0.6, y: 1.5, w: 7.6, h: 4.9, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, barOverlapPct: 100,
    chartColors: [HX.accent1, HX.accent3], valAxisMinVal: 0.8, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: true, legendPos: "b",
    title: "장면별 IoU — 전 장면 재현(로컬 CPU, 학습 장면 포함) · ★ 시험 장면", valAxisTitle: "IoU", showValAxisTitle: true, catAxisLabelFontSize: 9,
  });
  stat(s, 8.5, 1.5, 4.2, 1.45, `F1 ${sar.test_scene.f1.toFixed(4)}`, `레이더(ICEYE) 시험 장면 ${sar.test_scene.date} — IoU ${sar.test_scene.water_iou.toFixed(4)} · 정밀도 ${sar.test_scene.precision.toFixed(3)} · 재현율 ${sar.test_scene.recall.toFixed(3)} (공식, 배포 이미지 GPU)`, C.accent1, "통계 SAR", 30);
  stat(s, 8.5, 3.1, 4.2, 1.45, `F1 ${opt.test_scene.f1.toFixed(4)}`, `광학(PlanetScope) 시험 장면 ${opt.test_scene.date} — IoU ${opt.test_scene.water_iou.toFixed(4)} · 정밀도 ${opt.test_scene.precision.toFixed(3)} · 재현율 ${opt.test_scene.recall.toFixed(3)} (공식)`, C.accent3, "통계 광학", 30);
  const p0325 = areaB.sensors.PlanetScope.rows.find((r) => r.date === "2020-03-25");
  s.addText([
    { text: "읽는 법. ", options: { bold: true } },
    { text: `전 장면 합산(IoU 레이더 ${sar.pooled.water_iou.toFixed(3)} · 광학 ${opt.pooled.water_iou.toFixed(3)})은 학습 장면이 섞여 있어 성능 대표값이 아니다. 로컬 CPU 재현(04-16 IoU ${sar.per_scene[3].water_iou.toFixed(3)})은 공식 GPU 값(${sar.test_scene.water_iou.toFixed(4)})과 소수 3자리에서 다르다. 광학 03-25 는 촬영 범위가 남쪽으로 8.6 km 옮겨가 면적이 튀지만 공통 격자 안 면적 ${p0325.area_km2_common_frame.toFixed(2)} km² 는 정답 라벨 ${p0325.gt_area_km2.toFixed(2)} km² 와 일치한다.` },
  ], { x: 8.5, y: 4.7, w: 4.2, h: 1.75, fontSize: 10, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "읽는 법" });
  note(s, "SAR 모델 WBMS_SAR_ICEYE.h5 (U-Net+ASPP, γ⁰ dB 1밴드) · 광학 모델 WBMS_Optic_PlanetScope.h5 (Attention U-Net, 4밴드) · 임계값 0.5. 인수 패키지 wbms:0.13(2026-08-20) 기준 — 시립대 납품본 wbms:v1(09-28)은 미수령.");
  s.addNotes("출처: regions/nakdong_busan/satellite/detection_eval.json (test_scene = 인수 VALREPORT_Busan_*.json per_scene 시험 장면, pooled/per_scene = 로컬 재현), area_series.json(03-25 공통 격자 vs GT). 인용 규율: 인수 문서 refreeze_20260818_v010.md '지표 인용 규율'.");
}

// ───────────────────────────── 6. 탐지 대청
{
  const s = light("관측: 탐지·위성 수위");
  title(s, "수체 탐지 ② 금강 대청댐 — 라벨이 없어 IoU 대신 면적·위성 수위로 실증");
  subline(s, "PlanetScope 10장면을 부산과 공용인 광학 모델로 탐지. 면적은 센서별 공통 격자(같은 창)로 다시 재어 날짜끼리 비교할 수 있게 했다.");
  const rowsD = areaD.sensors.PlanetScope.rows;
  const lv = Object.fromEntries(DATA.dcd_levels.map((r) => [r[0], r]));
  const rows = [["촬영일", "공통 격자 면적 (km²)", "장면 자기 범위 면적 (km²)", "장면 유효 덮개율", "위성 수위 보정값 (EL.m)", "대청댐 실측 (EL.m)", "차이 (m)"]];
  rowsD.forEach((r) => {
    const l = lv[r.date] || [];
    const corr = l[1], g = l[3];
    rows.push([r.date, r.area_km2_common_frame.toFixed(2), r.area_km2_scene_footprint.toFixed(2), `${Math.round(r.frame_valid_fraction * 100)} %`, corr != null ? corr.toFixed(2) : "— (보정 없음)", g != null ? g.toFixed(2) : "—", corr != null && g != null ? (corr - g).toFixed(2) : "—"]);
  });
  table(s, rows, 0.6, 1.5, 8.3, [1.1, 1.35, 1.45, 1.1, 1.3, 1.1, 0.9], { rowH: 0.36, fontSize: 10, name: "대청 장면표" });
  const cmp = DATA.dcd_levels.filter((r) => r[1] != null && r[3] != null);
  const rmse = Math.sqrt(cmp.reduce((a, r) => a + (r[1] - r[3]) ** 2, 0) / cmp.length);
  stat(s, 9.2, 1.5, 3.5, 1.5, `${rmse.toFixed(3)} m`, `대청댐 본댐 위성 수위(보정) vs WAMIS 실측 RMSE — ${cmp.length}장면 (03-29 는 보정 상수 없음)`, C.accent5, "통계 대청 RMSE");
  card(s, 9.2, 3.2, 3.5, 2.75, { fill: C.background2, name: "대청 설명 카드" });
  bullets(s, [
    "2019년 봄 5장은 장면 유효 덮개율이 59~87 % 라 공통 격자 면적이 들쭉날쭉하다 — 면적 근거로는 쓰지 않는다.",
    "2022년 5월 5장은 덮개율이 높고 면적 16~19 km², 수위 70.5~71.2 m 로 안정적이다.",
    "위성 수위는 '그날의 관측'이지 예측이 아니다. 예측 성적은 다음 장부터 지상 수위계로 본다.",
  ], 9.4, 3.35, 3.1, 2.5, { fontSize: 11.5, space: 6, name: "대청 설명" });
  s.addNotes("출처: regions/geum_daecheong/satellite/area_series.json(rows), docs/presentation/data/timeseries_deck_data.json dcd_levels(= API /satellite/levels?station_id=대청댐, corrected vs gauge). RMSE 는 보정값이 있는 9장면.");
}

// ───────────────────────────── 7. 위성 수위 vs 실측
{
  const s = light("관측: 탐지·위성 수위");
  title(s, "수위 산정 vs 지상 수위계 — 같은 지점·같은 날의 관측 대조");
  subline(s, "수체 경계 화소의 DEM 표고로 수위를 산정(③)하고 융합 보정(④)을 거친다. 부산 값은 시립대 '모델 1·기상 LSTM'(V1 22쪽, 0.0453 m)과 같은 번들·4지점(정천·호포·김해월촌리·구포)의 표본 내 값 — 시립대 핵심지표 2는 '모델 2·신뢰 게이트'(V1 21쪽, 0.0566 m, 부산 4곳 26셀).");
  const rows = [["지역", "수위계 지점", "위성 시점", "대조 일수", "RMSE", "MAE", "보정 방식"]];
  const add = (reg, st, key, fix) => {
    const recs = DATA[key].filter((r) => r[4] != null && (r[2] != null || r[3] != null));
    const errs = recs.map((r) => ((r[2] != null ? r[2] : r[3]) - r[4]));
    const rmse = Math.sqrt(errs.reduce((a, e) => a + e * e, 0) / errs.length), mae = errs.reduce((a, e) => a + Math.abs(e), 0) / errs.length;
    rows.push([reg, st, String(DATA[key].length), String(recs.length), `${cm(rmse)} cm`, `${cm(mae)} cm`, fix]);
  };
  add("낙동강 하구", "구포대교", "busan_levels:구포대교", "LSTM v3 (8월 인수)");
  add("낙동강 하구", "호포대교", "busan_levels:호포대교", "LSTM v3 (8월 인수)");
  add("낙동강 하구", "정천교", "busan_levels:정천교", "LSTM v3 (8월 인수)");
  rows.push(["낙동강 하구", "월촌리·양산·호포·정천", "26셀", "26", "5.7 cm", "—", "시립대 현행 신뢰 게이트"]);
  {
    const cmp = DATA.dcd_levels.filter((r) => r[1] != null && r[3] != null);
    const errs = cmp.map((r) => r[1] - r[3]);
    const rmse = Math.sqrt(errs.reduce((a, e) => a + e * e, 0) / errs.length), mae = errs.reduce((a, e) => a + Math.abs(e), 0) / errs.length;
    rows.push(["금강 대청댐", "대청댐 본댐", String(DATA.dcd_levels.length), String(cmp.length), `${cm(rmse)} cm`, `${cm(mae)} cm`, "수위대 상수 보정"]);
  }
  table(s, rows, 0.6, 1.5, 6.3, [1.0, 1.25, 0.65, 0.7, 0.75, 0.7, 1.25], { rowH: 0.4, fontSize: 10, name: "대조표" });
  const cmpD = DATA.dcd_levels.filter((r) => r[1] != null && r[3] != null);
  s.addChart(pres.ChartType.line, [
    { name: "WAMIS 실측", labels: cmpD.map((r) => r[0]), values: cmpD.map((r) => +r[3].toFixed(2)) },
    { name: "위성 수위 (보정)", labels: cmpD.map((r) => r[0]), values: cmpD.map((r) => +r[1].toFixed(2)) },
  ], {
    ...CHART_BASE, x: 7.1, y: 1.45, w: 5.6, h: 3.4, chartColors: [HX.dk2, HX.accent5], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    valAxisMinVal: 69, valAxisMaxVal: 76, valAxisMajorUnit: 1, valAxisLabelFormatCode: "0", showLegend: true, legendPos: "b",
    title: "대청댐 본댐 — 위성 수위(보정) vs 실측 (EL.m)", catAxisLabelRotate: -30, catAxisLabelFontSize: 9,
  });
  const batch = (prefix) => {
    const d = DATA.dcd_levels.filter((r) => r[0].startsWith(prefix) && r[1] != null && r[3] != null).map((r) => r[1] - r[3]);
    const rmse = Math.sqrt(d.reduce((a, e) => a + e * e, 0) / d.length);
    const abs = d.map((e) => Math.abs(e));
    return { n: d.length, lo: cm(Math.min(...abs), 0), hi: cm(Math.max(...abs), 0), rmse: cm(rmse) };
  };
  const b19 = batch("2019"), b22 = batch("2022");
  bullets(s, [
    "부산 세 지점 1~4.5 cm 는 표본 내(in-sample, 학습 30건과 동일) 값 — 같은 번들의 정식 LODO 4.53 cm 가 시립대 '모델 1'(V1 22쪽). 시립대 핵심지표 2(V1 1·5·31쪽)는 '모델 2' 신뢰 게이트 — 원 수위 84.4 → 상수 11.5 → 게이트 5.66 cm(4지점 26셀, 7곳 전체 6.8 cm), 목표 50 cm 충족.",
    `대청댐은 전체 36 cm 차이. 2019년 봄 ${b19.n}장은 덮개율이 낮고 보정 상수가 하나뿐이라 ${b19.lo}~${b19.hi} cm(RMSE ${b19.rmse} cm)로 크고, 2022년 5월 ${b22.n}장은 ${b22.lo}~${b22.hi} cm(RMSE ${b22.rmse} cm)다.`,
    "건기 표본이라 '오늘 영상을 쓰지 않는 고정값'(1.99~2.99 cm)이 두 모델보다 정확하다(시립대 명시) — 영상의 이득은 홍수기 자료로 확인해야 한다. 이 표는 관측끼리의 대조이며 예측 입력이 아니다.",
  ], 0.6, 4.35, 6.3, 2.2, { fontSize: 11, space: 5 });
  note(s, "부산 위성 시점 8회 = ICEYE 4 + PlanetScope 4 (2020-02-18~04-16, 갈수기). 정천교는 2회가 촬영 범위 밖이라 6회. 시립대 모델 2 지점별(보정 후, V0_8 22쪽 — V1 에는 미표기): 월촌리 8.8 · 양산교 3.6 · 호포대교 3.4 · 정천교 5.2 cm — 호포·정천은 우리 표본 내 값(1.0·4.5 cm)과 같은 자릿수. wbms:v1 수령 후 게이트로 재산출 예정.", 7.1, 5.0, 5.6, 1.5);
  s.addNotes("출처: timeseries_deck_data.json busan_levels:* (API /regions/nakdong_busan/satellite/levels?station_id=), dcd_levels. 시립대 수치는 2026-10-16 전체회의 시립대 발표자료 V1(45장, V0_8 에서 갱신): 1·5·31쪽 핵심지표 2 = 모델 2 신뢰 게이트 0.0566 m(부산 4곳 26셀 — 3곳 사후 제외, 7곳 전체 0.068 m), 22쪽 모델 1(정천·후포·김해·구포 4지점 30건, LSTM 0.0453·위성+지점 상수 0.0677 m), 21쪽 모델 2(원 0.8443→상수 0.1154→게이트 0.0566 m, 고정값 0.0299 m 가 더 낮음). V1 5쪽에는 모델 1 이 0.0447 m 로 적혀 있어(8·22쪽은 0.0453) 오기 확인 요청. 지점별 게이트 값(0.088/0.036/0.034/0.052 m)은 V0_8 22쪽에만 있음. 인용 규칙: 부산 2020년 2~4월 갈수기 표본, 홍수기 급변 미반영.");
}

// ───────────────────────────── 8. 시계열 입력
pres.addSection({ title: "예측: 입력·모델" });
{
  const s = light("예측: 입력·모델");
  title(s, "시계열 예측에 들어가는 자료 — 지상 수위계 10년 + 기상청 ASOS");
  subline(s, "원점(출발일) 이전 자료만 쓴다. 미래를 본 값은 특징에도, 파라미터 선택에도 들어가지 않는다.");
  const colW = 3.9, y0 = 1.55, h = 4.1;
  card(s, 0.6, y0, colW, h, { name: "입력 카드" });
  card(s, 4.7, y0, colW, h, { name: "특징 카드" });
  card(s, 8.8, y0, colW, h, { name: "출력 카드" });
  [["① 입력 계열", C.accent1], ["② 파생 특징", C.accent5], ["③ 예측 대상·채점", C.accent2]].forEach(([t, c], i) => {
    iconCircle(s, 0.8 + i * 4.1, y0 + 0.18, 0.42, String(i + 1), c);
    s.addText(t.slice(2), { x: 1.35 + i * 4.1, y: y0 + 0.18, w: colW - 0.9, h: 0.42, fontSize: 15, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: `${t} 제목` });
  });
  bullets(s, [
    "WAMIS 일 수위 — 부산 6지점 2015~2024, 대청 3지점 2019~2024 (결측은 특징 구성용으로만 선형 보간, 정답에는 쓰지 않음)",
    "기상청 ASOS 일자료 — 부산 159 · 대전 133: 강수량, 평균·최저·최고 기온, 습도 (2015~2024)",
    "WAMIS 자료 품질 관리: 배율 오류로 평평해진 구간, 물리적으로 불가능한 스파이크만 제거 — 홍수 봉우리는 보존",
    "위성 수위·면적은 입력이 아니다 (관측 대조용)",
  ], 0.8, y0 + 0.8, colW - 0.4, h - 1.0, { fontSize: 11.5, space: 6, name: "입력 목록" });
  bullets(s, [
    "최근 p일 수위(또는 변화량), 30·365일 이동평균, 연중일 조화항(2차)",
    "기상(AR-X 계열): 원점 이전 강수 1·3·7·30일 합(log1p), 7일 평균기온, 기온 편차",
    "계절편차 계열: 학습 구간 연중일 평년값을 뺀 편차",
    "신경망: 최근 30·60일 창의 수위·강수·기온 시퀀스 (GPU 학습 → 가중치 저장 → numpy 추론)",
  ], 4.9, y0 + 0.8, colW - 0.4, h - 1.0, { fontSize: 11.5, space: 6, name: "특징 목록" });
  bullets(s, [
    "예측일수 D+1 ~ D+30 (보고 지평 1·3·5·7·10·14·30일), 지점마다 직접 다중지평",
    "채점: 시험 연도 2024 전 날짜를 원점으로 한 RMSE · MAE (m, cm 병기)",
    "개선율(skill) = 1 − MSE(모델) / MSE(직전값 유지). +50 % 면 오차 제곱이 절반",
    "밴드: 검증 연도 잔차의 경험적 산포(±1σ, 5~95 %) — 신뢰구간이 아님",
  ], 9.0, y0 + 0.8, colW - 0.4, h - 1.0, { fontSize: 11.5, space: 6, name: "출력 목록" });
  note(s, "부산 ASOS 강수는 하구 지점 수위와 관계가 약하다(상류 강수·조석 영향). 대청은 댐 방류 운영이 수위를 좌우하는데, 방류량 자료는 아직 입력에 없다.");
  s.addNotes("출처: backend/app/gauge/forecasters.py(ridge_arx 특징 설명 — GET /gauge/forecasters description), regions/*/weather/asos_daily.json 열, training.py 규율.");
}

// ───────────────────────────── 9. 모델 13종 + 선택 규칙
{
  const s = light("예측: 입력·모델");
  title(s, "후보 모델 13종을 같은 규칙으로 겨루고, 셀마다 하나를 고른다");
  subline(s, "선택 규칙: 검증 연도별 폴드에서 직전값 유지 대비 MSE 비율의 평균이 가장 작은 후보 → 학습+검증 구간으로 재적합 → 시험 연도 채점.");
  const groups = [
    ["통계·회귀·사례 7종", C.accent1, ["릿지 자기회귀(AR)", "릿지 AR + 기상(AR-X)", "계절편차 자기회귀", "평균 회귀 AR(1)", "유사사례 k-NN", "계열 앙상블 평균", "계절 평년값(참고)"]],
    ["딥러닝 4종 (GPU 학습)", C.accent5, ["MLP 신경망", "LSTM 순환신경망", "GRU 순환신경망", "TCN 시간합성곱"]],
    ["구조 분해 1종", C.accent3, ["Prophet + 잔차 자기회귀"]],
    ["기준선 (선택 불가)", C.accent2, ["직전값 유지(persistence) — skill 의 분모"]],
  ];
  let y = 1.55;
  groups.forEach(([head, color, items]) => {
    const h = 0.42 + items.length * 0.21 + 0.1;
    card(s, 0.6, y, 5.3, h, { name: `모델 묶음 ${head}` });
    s.addShape(pres.ShapeType.ellipse, { x: 0.78, y: y + 0.14, w: 0.22, h: 0.22, fill: { color }, line: { color, width: 0 }, objectName: `점 ${head}` });
    s.addText(head, { x: 1.1, y: y + 0.08, w: 4.6, h: 0.34, fontSize: 13, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: `묶음 제목 ${head}` });
    bullets(s, items, 1.1, y + 0.45, 4.6, items.length * 0.21 + 0.05, { fontSize: 11, space: 1, name: `묶음 목록 ${head}` });
    y += h + 0.1;
  });
  let yy = drawFolds(s, 6.3, 1.95, 6.4, 2018, [2019, 2020, 2021, 2022, 2023], 2024, 2015, "낙동강 하구 — 학습 ≤2018 · 검증 2019~2023 (5폴드) · 시험 2024");
  yy = drawFolds(s, 6.3, yy + 0.4, 6.4, 2020, [2021, 2022, 2023], 2024, 2015, "금강 대청댐 — 학습 ≤2020 · 검증 2021~2023 (3폴드) · 시험 2024 (자료가 2019년부터)");
  note(s, "검증 연도를 하나만 쓰면(예: 2023 홍수) 특정 해에 맞춘 모델이 뽑혀 다음 해에 크게 틀렸다 — 연도별 폴드 평균으로 고른 뒤 그 문제가 사라졌다. 딥러닝은 torch(GPU)로 학습하고 서버는 numpy 로 추론한다.", 6.3, yy + 0.15, 6.4, 1.0);
  s.addNotes("출처: GET /gauge/forecasters 목록(backend/app/gauge/forecasters.py, plugins/neural.py, plugins/prophet_hybrid.py), evaluation.json note_ko(선택 규칙), region.json forecast.split.");
}

// ───────────────────────────── 10. 부산 결과 표
pres.addSection({ title: "결과" });
function resultTable(s, ev, order, y, name) {
  const rows = [["지점 (지형)", "D+1 모델", "RMSE", "개선율", "D+7 모델", "RMSE", "개선율", "D+30 모델", "RMSE", "개선율"]];
  order.forEach((st) => {
    const r1 = row(ev, st, 1), r7 = row(ev, st, 7), r30 = row(ev, st, 30);
    const cell = (r) => [MODEL_KO[r.forecaster_id] || r.forecaster_id, `${cm(r.test_rmse)} cm`, { text: pct(r.skill_vs_persistence), options: { color: r.skill_vs_persistence < 0 ? HX.accent6 : HX.dk1, bold: r.skill_vs_persistence < 0 } }];
    rows.push([`${st} (${TERRAIN_KO[r1.terrain] || r1.terrain})`, ...cell(r1), ...cell(r7), ...cell(r30)]);
  });
  table(s, rows, 0.6, y, 12.1, [1.95, 1.5, 1.0, 0.88, 1.5, 1.0, 0.88, 1.5, 1.0, 0.89], { rowH: 0.4, fontSize: 10.5, name });
}
{
  const s = light("결과");
  title(s, `결과 ① 낙동강 하구 — 6지점 × 7예측일수 중 ${evalB.cells_beating_persistence}/${evalB.cells_total}셀이 직전값 유지보다 낫다`);
  subline(s, "시험 연도 2024 (학습·선택에 쓰지 않음). RMSE 는 지점 수위 단위 cm. 개선율 = 1 − MSE(모델)/MSE(직전값 유지).");
  resultTable(s, evalB, ORDER_B, 1.55, "부산 성적표");
  const ys = row(evalB, "양산교", 30);
  bullets(s, [
    `하구 지점(낙동강하구언·구포대교)은 하구둑 조절로 변동이 작아 RMSE 4~8 cm, 개선율 +23~+52 %. 본류 삼랑진교는 변동이 커 RMSE 21~34 cm 지만 개선율 +30~+49 %.`,
    `유일하게 진 셀: 양산교 D+30 (${cm(ys.test_rmse)} cm vs 직전값 ${cm(ys.persistence_test_rmse)} cm, ${pct(ys.skill_vs_persistence)}) — 2024년에 수위 기준점이 이동한 지점(regime shift 표시)이라 장기 예측이 기준선에 진다.`,
    `기상이 뚜렷이 도운 셀 ${evalB.cells_weather_helps}/${evalB.cells_total} — 하구·본류에서 부산 ASOS 강수의 설명력이 작다(상류 강수·조석이 지배).`,
  ], 0.6, 4.6, 12.1, 1.9, { fontSize: 12, space: 6 });
  s.addNotes("출처: regions/nakdong_busan/models/gauge/evaluation.json rows (test_rmse, persistence_test_rmse, skill_vs_persistence, regime_shift_flag), cells_weather_helps.");
}

// ───────────────────────────── 11. 부산 차트
{
  const s = light("결과");
  title(s, "결과 ② 낙동강 하구 — 지점별 D+7 오차와 삼랑진교의 예측일수별 오차");
  subline(s, "막대는 시험 연도 2024 RMSE(cm). 주황은 직전값 유지(기준선), 청록은 그 셀에서 선택된 모델.");
  s.addChart(pres.ChartType.bar, [
    { name: "선택 모델", labels: ORDER_B, values: ORDER_B.map((st) => +cm(row(evalB, st, 7).test_rmse)) },
    { name: "직전값 유지", labels: ORDER_B, values: ORDER_B.map((st) => +cm(row(evalB, st, 7).persistence_test_rmse)) },
  ], { ...CHART_BASE, x: 0.6, y: 1.5, w: 6.0, h: 4.6, barDir: "col", barGapWidthPct: 50, chartColors: [HX.accent1, HX.accent2], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: true, legendPos: "b", title: "지점별 RMSE — D+7 (cm)", valAxisLabelFormatCode: "0", catAxisLabelFontSize: 9 });
  const st = "삼랑진교";
  s.addChart(pres.ChartType.bar, [
    { name: "선택 모델", labels: HORIZONS.map((h) => `D+${h}`), values: HORIZONS.map((h) => +cm(row(evalB, st, h).test_rmse)) },
    { name: "직전값 유지", labels: HORIZONS.map((h) => `D+${h}`), values: HORIZONS.map((h) => +cm(row(evalB, st, h).persistence_test_rmse)) },
  ], { ...CHART_BASE, x: 6.8, y: 1.5, w: 5.9, h: 4.6, barDir: "col", barGapWidthPct: 50, chartColors: [HX.accent1, HX.accent2], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: true, legendPos: "b", title: `${st}(본류) — 예측일수별 RMSE (cm)`, valAxisLabelFormatCode: "0" });
  note(s, `삼랑진교 선택 모델: ${HORIZONS.map((h) => `D+${h} ${MODEL_KO[row(evalB, st, h).forecaster_id]}`).join(" · ")}. 직전값 유지 오차는 예측일수가 길수록 커지는데 선택 모델은 D+3 이후 거의 평평하다 — 평균 회귀 성질을 학습한 결과.`);
  s.addNotes("출처: evaluation.json rows (nakdong_busan). 값은 test_rmse·persistence_test_rmse × 100 (cm).");
}

// ───────────────────────────── 12. 대청 결과 표
{
  const s = light("결과");
  title(s, `결과 ③ 금강 대청댐 — ${evalD.cells_beating_persistence}/${evalD.cells_total}셀 기준선 우위, 본댐은 오차 자체가 크다`);
  subline(s, "시험 연도 2024. 대청댐 본댐 수위는 EL.m(해발) 기준이라 변동 폭이 수 m — 조정지댐·방수로는 수십 cm. 아래 표는 전 예측일수의 RMSE(cm)와 선택 모델.");
  resultTable(s, evalD, ORDER_D, 1.55, "대청 성적표");
  const d7 = row(evalD, "대청댐", 7), d30 = row(evalD, "대청댐", 30), j30 = row(evalD, "대청조정지댐", 30);
  bullets(s, [
    `대청댐 본댐: D+1 ${cm(row(evalD, "대청댐", 1).test_rmse)} cm → D+7 ${cm(d7.test_rmse)} cm → D+30 ${cm(d30.test_rmse)} cm. 개선율은 +2~+13 % 에 그친다 — 홍수기 방류 운영과 저수 전략이 수위를 정하는데 그 입력(방류량·유입량)이 모델에 없다.`,
    `대청조정지댐: 변동이 작고 규칙적이라 D+30 에서 개선율 ${pct(j30.skill_vs_persistence)} (RMSE ${cm(j30.test_rmse)} cm vs 직전값 ${cm(j30.persistence_test_rmse)} cm) — 13종 중 Prophet+잔차AR 이 유일하게 뽑힌 셀.`,
    `대청댐방수로: 본류 성격. D+7 이후 개선율 +35~+47 %, 기상(AR-X·앙상블)이 뚜렷이 도운 셀 ${evalD.cells_weather_helps}/${evalD.cells_total} 은 모두 이 지점 계열.`,
  ], 0.6, 3.4, 12.1, 1.55, { fontSize: 11.5, space: 4 });
  // 전 예측일수 행렬: RMSE(cm) 와 선택 모델, 괄호는 직전값 유지 RMSE
  const matrix = [["지점 · RMSE cm (직전값 유지)", ...HORIZONS.map((h) => `D+${h}`)]];
  ORDER_D.forEach((st) => matrix.push([st, ...HORIZONS.map((h) => { const r = row(evalD, st, h); return `${cm(r.test_rmse)} (${cm(r.persistence_test_rmse)})\n${MODEL_SHORT[r.forecaster_id]}`; })]));
  table(s, matrix, 0.6, 4.8, 12.1, [2.5, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37, 1.38], { rowH: 0.42, fontSize: 9.5, name: "대청 전 지평 행렬" });
  s.addNotes("출처: regions/geum_daecheong/models/gauge/evaluation.json rows, cells_weather_helps(5 — 방수로 D+5/7/10 등 uses_weather=true 셀).");
}

// ───────────────────────────── 13. 대청 차트
{
  const s = light("결과");
  title(s, "결과 ④ 금강 대청댐 — 본댐 오차는 길수록 커지고, 조정지댐은 평평하다");
  subline(s, "막대는 시험 연도 2024 RMSE(cm). 주황은 직전값 유지(기준선), 청록은 그 셀에서 선택된 모델. 두 그래프의 세로축 단위가 다르다는 점에 주의.");
  const mk = (st, x, w, max) => s.addChart(pres.ChartType.bar, [
    { name: "선택 모델", labels: HORIZONS.map((h) => `D+${h}`), values: HORIZONS.map((h) => +cm(row(evalD, st, h).test_rmse)) },
    { name: "직전값 유지", labels: HORIZONS.map((h) => `D+${h}`), values: HORIZONS.map((h) => +cm(row(evalD, st, h).persistence_test_rmse)) },
  ], { ...CHART_BASE, x, y: 1.5, w, h: 4.6, barDir: "col", barGapWidthPct: 50, chartColors: [HX.accent1, HX.accent2], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", showLegend: true, legendPos: "b", title: `${st} — 예측일수별 RMSE (cm)`, valAxisLabelFormatCode: "0", valAxisMaxVal: max });
  mk("대청댐", 0.6, 6.0, 200);
  mk("대청조정지댐", 6.8, 5.9, 60);
  note(s, `대청댐 선택 모델: ${HORIZONS.map((h) => `D+${h} ${MODEL_KO[row(evalD, "대청댐", h).forecaster_id]}`).join(" · ")}. 조정지댐은 D+1~14 평균 회귀·계절편차 AR, D+30 Prophet+잔차AR.`);
  s.addNotes("출처: evaluation.json rows (geum_daecheong).");
}

// ───────────────────────────── 14. 전망 예시 (표본 외 원점)
{
  const s = light("결과");
  title(s, "전망 예시 — 시험 연도 원점 2024-07-01(장마철)에서 30일을 내다본 결과");
  subline(s, "원점 이전 자료만으로 만든 표본 외(out-of-sample) 전망. 회색은 원점까지 실측, 진한 남색은 그 뒤 실제 수위, 청록은 전망, 주황은 직전값 유지.");
  const mkFc = (key, label, x, w, fmt) => {
    const f = DATA[key];
    const hist = f.history.slice(-30);
    const labels = [...hist.map((h) => h.date.slice(5)), ...f.steps.map((st) => st.target_date.slice(5))];
    const nH = hist.length;
    const histVals = [...hist.map((h) => h.water_level_m), ...f.steps.map(() => null)];
    const obsVals = [...hist.map((h, i) => (i === nH - 1 ? h.water_level_m : null)), ...f.steps.map((st) => st.observed_m)];
    const fcVals = [...hist.map((h, i) => (i === nH - 1 ? h.water_level_m : null)), ...f.steps.map((st) => st.forecast_m)];
    const perVals = [...hist.map((h, i) => (i === nH - 1 ? h.water_level_m : null)), ...f.steps.map((st) => st.persistence_reference_m)];
    const vals = [...histVals, ...obsVals, ...fcVals].filter((v) => v != null);
    const lo = Math.floor(Math.min(...vals) - 0.3), hi = Math.ceil(Math.max(...vals) + 0.3);
    s.addChart(pres.ChartType.line, [
      { name: "실측 (원점까지)", labels, values: histVals },
      { name: "실제 수위 (원점 뒤)", labels, values: obsVals },
      { name: "전망", labels, values: fcVals },
      { name: "직전값 유지", labels, values: perVals },
    ], { ...CHART_BASE, x, y: 1.5, w, h: 3.9, chartColors: ["9AA7B4", HX.dk2, HX.accent1, HX.accent2], lineSize: 2, lineDataSymbol: "none", displayBlanksAs: "gap", showLegend: true, legendPos: "b", title: label, catAxisLabelFrequency: 6, catAxisLabelFontSize: 9, valAxisMinVal: lo, valAxisMaxVal: hi, valAxisLabelFormatCode: fmt });
    const errs = f.steps.filter((st) => st.observed_m != null);
    const rmse = Math.sqrt(errs.reduce((a, st) => a + (st.forecast_m - st.observed_m) ** 2, 0) / errs.length);
    const prmse = Math.sqrt(errs.reduce((a, st) => a + (st.persistence_reference_m - st.observed_m) ** 2, 0) / errs.length);
    return { rmse, prmse, n: errs.length, origin: f.origin_level_m };
  };
  const a = mkFc("geum_daecheong:대청댐", "대청댐 본댐 (EL.m)", 0.6, 6.0, "0.0");
  const b = mkFc("nakdong_busan:삼랑진교", "삼랑진교 (m)", 6.8, 5.9, "0.0");
  statRow(s, 0.6, 5.5, 6.0, 0.62, `${cm(a.rmse)} cm`, `대청댐 30일 RMSE — 직전값 유지 ${cm(a.prmse)} cm, 개선율 ${pct(1 - (a.rmse ** 2) / (a.prmse ** 2))}`, C.accent1, "통계 대청 예시");
  statRow(s, 6.8, 5.5, 5.9, 0.62, `${cm(b.rmse)} cm`, `삼랑진교 30일 RMSE — 직전값 유지 ${cm(b.prmse)} cm, 개선율 ${pct(1 - (b.rmse ** 2) / (b.prmse ** 2))}`, C.accent1, "통계 삼랑진 예시");
  s.addText("장마철 원점이라 두 지점 모두 원점 뒤 수위가 크게 오른다. 전망은 방향(상승)을 잡지만 상승 폭은 못 따라간다 — '앞으로 내릴 비'와 댐 방류를 모르는 모델의 한계가 그대로 보인다. 연간 성적표(앞 장)는 2024년 모든 날을 원점으로 한 평균이라 이 한 사례보다 좋다.", { x: 0.6, y: 6.2, w: 6.0, h: 0.75, fontSize: 9.5, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "예시 설명 1" });
  s.addText("표본 외 검증: 원점 2024-07-01 은 시험 연도 안이고, 파라미터는 시험 구간을 보지 않은 검증용(params_eval)이다. 서비스 화면 ④ '시나리오'에서 같은 원점·예측일수로 재현할 수 있다.", { x: 6.8, y: 6.2, w: 5.9, h: 0.75, fontSize: 9.5, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "예시 설명 2" });
  s.addNotes(`출처: timeseries_deck_data.json (API /regions/{id}/gauge/{station}/forecast?as_of=2024-07-01&horizon=30). 대청댐 n=${a.n}, 삼랑진교 n=${b.n}.`);
}

// ───────────────────────────── 15. 백테스트 2024 D+7
{
  const s = light("결과");
  title(s, "2024년 전체를 하루씩 원점으로 — D+7 백테스트 궤적");
  subline(s, "매일을 원점으로 7일 뒤를 예측해 실제와 겹쳐 그렸다(366원점). 연간 성적표의 D+7 값은 이 궤적의 RMSE 다.");
  const mkBt = (key, label, x, w, fmt) => {
    const t = DATA[key];
    // 월초에만 날짜 라벨(그 외는 빈 문자열) — 366 개를 다 쓰면 축이 읽히지 않는다
    const labels = t.dates.map((d) => (d.slice(8) === "01" ? d.slice(5) : ""));
    s.addChart(pres.ChartType.line, [
      { name: "실제 수위", labels, values: t.truth.map((v) => (v == null ? null : +v.toFixed(3))) },
      { name: "D+7 전망", labels, values: t.pred.map((v) => (v == null ? null : +v.toFixed(3))) },
    ], { ...CHART_BASE, x, y: 1.5, w, h: 4.2, chartColors: [HX.dk2, HX.accent1], lineSize: 1.5, lineDataSymbol: "none", displayBlanksAs: "gap", showLegend: true, legendPos: "b", title: label, catAxisLabelFontSize: 9, valAxisLabelFormatCode: fmt });
  };
  mkBt("backtest_dcd_7", "대청댐 본댐 — 2024 D+7 (EL.m)", 0.6, 6.0, "0");
  mkBt("backtest_samrangjin_7", "삼랑진교 — 2024 D+7 (m)", 6.8, 5.9, "0.0");
  const d7 = row(evalD, "대청댐", 7), s7 = row(evalB, "삼랑진교", 7);
  statRow(s, 0.6, 5.85, 6.0, 0.72, `${cm(d7.test_rmse)} cm`, `대청댐 D+7 RMSE (${MODEL_KO[d7.forecaster_id]}) · 직전값 유지 ${cm(d7.persistence_test_rmse)} cm · ${pct(d7.skill_vs_persistence)} · MAE ${cm(d7.test_mae)} cm`, C.accent1, "통계 대청 백테스트");
  statRow(s, 6.8, 5.85, 5.9, 0.72, `${cm(s7.test_rmse)} cm`, `삼랑진교 D+7 RMSE (${MODEL_KO[s7.forecaster_id]}) · 직전값 유지 ${cm(s7.persistence_test_rmse)} cm · ${pct(s7.skill_vs_persistence)} · MAE ${cm(s7.test_mae)} cm`, C.accent1, "통계 삼랑진 백테스트");
  s.addNotes("출처: timeseries_deck_data.json backtest_* (API /gauge/{station}/backtest?horizon=7 trace), evaluation.json rows D+7.");
}

// ───────────────────────────── 16. 기상·미래 강수
{
  const s = light("결과");
  title(s, "기상 자료의 효과 — 지금은 작고, '앞으로 내릴 비'를 알면 조금 더 는다");
  subline(s, "위: 과거 강수 AR-X / +3일 완전 예보 / 목표일까지 완전 예보의 지점 평균 개선율(%). 실측 미래 강수를 예보로 간주한 상한 실험이며 공식 성적표에는 넣지 않았다.");
  const rows = [["지역", "D+1", "D+3", "D+7", "D+14", "D+30"]];
  Object.entries(FUTURE_RAIN).forEach(([k, v]) => rows.push([k, ...v]));
  table(s, rows, 0.6, 1.6, 12.1, [2.5, 1.92, 1.92, 1.92, 1.92, 1.92], { rowH: 0.42, fontSize: 11, name: "미래 강수표" });
  const colW = 3.9, y0 = 3.1, h = 2.75;
  card(s, 0.6, y0, colW, h, { name: "기상 카드 1" }); card(s, 4.7, y0, colW, h, { name: "기상 카드 2" }); card(s, 8.8, y0, colW, h, { name: "기상 카드 3" });
  const heads = ["지금 성적표에서 기상이 한 일", "완전 예보 상한에서 얻는 것", "그래서 다음에 넣을 입력"];
  heads.forEach((t, i) => s.addText(t, { x: 0.8 + i * 4.1, y: y0 + 0.15, w: colW - 0.4, h: 0.4, fontSize: 13.5, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: `기상 제목 ${i + 1}` }));
  bullets(s, [
    `기상이 뚜렷이 도운 셀: 부산 ${evalB.cells_weather_helps}/${evalB.cells_total}, 대청 ${evalD.cells_weather_helps}/${evalD.cells_total}`,
    "단기(D+1)에는 릿지 AR+기상이 뽑히는 지점(삼랑진교·호포대교)이 있지만, 하구와 저수지에서는 과거 강수가 수위를 잘 설명하지 못한다",
  ], 0.8, y0 + 0.65, colW - 0.4, h - 0.8, { fontSize: 11, space: 5, name: "기상 목록 1" });
  bullets(s, [
    "전 구간 예보를 넣어도 개선율 이득은 수 %p (전체 26지점 가중: D+3 +5.4 %p, D+7 +3.4 %p)",
    "대청은 D+30 에서 +32 → +40 % 로 비교적 크다 — 저수지는 장기 강수 총량이 수위를 좌우",
    "부산 하구는 D+30 에서 오히려 줄어든다(+28 → +21 %) — 하구 수위는 조석·상류 유입·하구둑 운영이 지배",
  ], 4.9, y0 + 0.65, colW - 0.4, h - 0.8, { fontSize: 11, space: 5, name: "기상 목록 2" });
  bullets(s, [
    "기상청 단기 예보 연결 (아직 미연결, 예보 오차만큼 이득이 줄어든다)",
    "댐 방류량·유입량 (대청댐 본댐 오차의 주원인)",
    "상류 지점 수위를 하류 지점 입력으로 (삼랑진교 → 구포대교 등 지점 간 전파)",
  ], 9.0, y0 + 0.65, colW - 0.4, h - 0.8, { fontSize: 11, space: 5, name: "기상 목록 3" });
  note(s, "출처: regions/*/models/gauge/experiments/future_rain.json (요약은 docs/presentation/미래강수_실험.md). 공식 성적표(앞 장)는 과거 강수까지만 쓴 모델이다.");
  s.addNotes("미래 강수 실험은 부수 실험 — 공식 수치 아님. 표의 값은 과제성과 덱 §32 와 동일.");
}

// ───────────────────────────── 17. 공식 지표
{
  const s = light("결과");
  title(s, "공식 성과지표(IITP·TTA) — 종합 정규화 MSE 0.014, 목표 0.38 이하 적합");
  subline(s, "정의: 변수별 [0,1] 정규화 후 예측 MSE 의 동가중 평균(수위 항 + 수체면적 항). 수위는 위성 관측일 원점 24건, 면적은 공통 격자 인접 장면쌍 6개.");
  const rows = [
    ["예측일수", "수위 정규화 MSE", "수체면적 정규화 MSE", "종합 MSE", "목표", "판정"],
    ["D+1", String(TTA.d1.level), String(TTA.d1.area), { text: String(TTA.d1.total), options: { bold: true } }, `${TTA.target} 이하`, { text: "적합", options: { color: HX.accent3, bold: true } }],
    ["D+7", String(TTA.d7.level), String(TTA.d7.area), { text: String(TTA.d7.total), options: { bold: true } }, `${TTA.target} 이하`, { text: "적합", options: { color: HX.accent3, bold: true } }],
  ];
  table(s, rows, 0.6, 1.65, 7.4, [1.1, 1.5, 1.6, 1.1, 1.0, 1.1], { rowH: 0.45, fontSize: 12, name: "공식 지표표" });
  stat(s, 8.3, 1.65, 4.4, 1.45, String(TTA.d1.total), `종합 정규화 MSE (D+1 · D+7) — 목표 ${TTA.target} 의 1/27 수준`, C.accent2, "통계 공식");
  card(s, 8.3, 3.3, 4.4, 2.55, { name: "지표 설명 카드" });
  bullets(s, [
    `이전 판 ${TTA.prev} 은 장면별 자기 촬영범위 면적 + 직전값 유지로 계산한 다른 정의 — 두 값 모두 목표를 만족하며 보고서에는 계열을 병기한다.`,
    "정규화 전 원단위 수위 MSE 는 전국 26지점 평균 D+1 0.033 · D+7 0.215 · D+30 0.685 m² — 댐 지점이 들어가면 장기 지평은 0.38 을 넘는다(정규화의 효과를 분리해 읽을 것).",
  ], 8.5, 3.45, 4.0, 2.3, { fontSize: 10.5, space: 6, name: "지표 설명" });
  bullets(s, [
    "수위 항: 위성 관측일 24건을 원점으로 한 모델 MSE 0.00118 vs 기준선 0.00177 (33.6 % 개선, D+7 기준)",
    "수체면적 항: 공통 격자 인접 장면쌍 6개, 평균회귀 0.02663 vs 직전값 유지 0.02934 (9.2 % 개선)",
    "면적 항이 종합값의 대부분을 차지한다 — 면적 계열은 관측이 적고 마스크 흔들림(수 km²)이 변동폭에 가깝다",
  ], 0.6, 3.3, 7.4, 3.1, { fontSize: 12, space: 8, name: "지표 구성" });
  s.addNotes("출처: docs/tta/TTA_제출문안_복붙용.md 판정표(절차 7 재학습 후 scripts/eval_iitp_indicator.py 실행값 D+1 0.00094/0.02663/0.014, D+7 0.00044/0.02663/0.014). 이전 판 0.135 는 data/eval/timeseries_metrics.json G 계열.");
}

// ───────────────────────────── 18. 한계
pres.addSection({ title: "한계·다음 단계" });
{
  const s = light("한계·다음 단계");
  title(s, "한계 — 숫자를 읽을 때 같이 봐야 하는 것");
  subline(s, "이 자료의 성적은 아래 조건 안에서만 유효하다.");
  const items = [
    ["위성 표본이 갈수기뿐", "부산 위성 8시점은 2020년 2~4월 갈수기 — 홍수기 급변 때의 탐지·위성 수위 정확도는 검증되지 않았다.", C.accent2],
    ["대청은 정답 라벨이 없다", "탐지 IoU 를 낼 수 없고 면적·위성 수위 대조로만 실증했다. 2019년 봄 장면은 덮개율 59~87 % 라 면적 근거에서 제외.", C.accent2],
    ["면적 시계열 모델은 검증 불가", "지역·센서당 관측 4~10개. 1일 재방문 면적 차(2.4 km²)가 계열 변동폭의 2/3 — 어떤 모델도 우열을 가릴 수 없어 기준선 판정만 했다.", C.accent6],
    ["저수지 본댐의 장기 오차", `대청댐 D+7 ${cm(row(evalD, "대청댐", 7).test_rmse)} cm, D+30 ${cm(row(evalD, "대청댐", 30).test_rmse)} cm — 방류·저수 운영 입력이 없어 개선율 +2~+13 % 에 그친다.`, C.accent6],
    ["기준점 이동 지점", `양산교 D+30 은 기준선보다 나쁘다(${pct(row(evalB, "양산교", 30).skill_vs_persistence)}) — 2024년 수위 기준이 바뀐 지점은 과거로 학습한 모델이 장기에서 진다.`, C.accent5],
    ["위성 보정은 표본 내·구 번들", "부산 1~4.5 cm 는 8월 인수 LSTM 번들의 표본 내 값(LODO 4.5 cm). 시립대 현행은 신뢰 게이트 5.7 cm(4지점 26셀)이고, 건기라 위성 미사용 고정값(2.0 cm)이 더 낮다 — 영상 이득은 홍수기 자료로.", C.accent5],
    ["수위 자료는 2024-12-31 까지", "'현재' 전망은 자료 안에서 출발일을 고르는 시나리오로만 가능하다. 2025년 이후는 WAMIS 수집을 돌려야 한다.", C.accent1],
    ["예보·방류 미연결", "기상청 단기 예보, 댐 방류량은 아직 입력에 없다 — 급상승 구간의 오차가 가장 큰 이유.", C.accent1],
  ];
  const colW = 5.95, rowH = 1.18;
  items.forEach(([head, body, color], i) => {
    const x = 0.6 + (i % 2) * (colW + 0.2), y = 1.5 + Math.floor(i / 2) * (rowH + 0.1);
    card(s, x, y, colW, rowH, { name: `한계 ${head}` });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.18, y: y + 0.2, w: 0.26, h: 0.26, fill: { color }, line: { color, width: 0 }, objectName: `한계 점 ${head}` });
    s.addText(head, { x: x + 0.55, y: y + 0.12, w: colW - 0.7, h: 0.4, fontSize: 12.5, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: `한계 제목 ${head}` });
    s.addText(body, { x: x + 0.55, y: y + 0.5, w: colW - 0.7, h: rowH - 0.58, fontSize: 10, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `한계 본문 ${head}` });
  });
  s.addNotes("근거: regions/nakdong_busan/satellite/timeseries_eval.json(A 계열 경고, 참고선 LODO 0.0453/기준선 0.0199), docs/presentation/면적모델_실험.md, evaluation.json(regime_shift_flag), 메모리 인용 규칙.");
}

// ───────────────────────────── 19. 다음 단계 (마무리)
{
  const s = dark("한계·다음 단계");
  s.addText("다음 단계", { x: 0.9, y: 0.9, w: 11.5, h: 0.5, fontSize: 18, color: "9FB3C8", margin: 0, isTextBox: true, objectName: "상단 라벨" });
  s.addText("지금 모델이 못 보는 것을 입력으로 넣고, 검증 표본을 홍수기까지 넓힌다", { x: 0.9, y: 1.4, w: 11.5, h: 1.1, fontSize: 27, bold: true, color: C.background1, margin: 0, valign: "middle", isTextBox: true, objectName: "마무리 제목" });
  const items = [
    ["1", "홍수기 위성 장면 확보", "2020년 갈수기 8시점에 더해 장마철 장면·라벨을 넣어 탐지·위성 수위를 급변 조건에서 재검증"],
    ["2", "댐 방류량·상류 수위 입력", "대청댐 본댐과 본류 지점의 장기 오차를 줄일 가장 직접적인 입력. 지점 간 전파(상류 → 하류) 특징 추가"],
    ["3", "기상청 단기 예보 연결", "완전 예보 상한에서 본 수 %p 이득을 실시간 예보로 옮긴다 — 예보 오차만큼 줄어드는 이득을 함께 보고"],
    ["4", "수위 자료 갱신 · 대청 라벨", "WAMIS 2025년 이후 수집으로 '현재' 전망을 열고, 대청 정답 라벨을 만들어 탐지 IoU 를 산출"],
    ["5", "K-water 2026 자료 연계", "대청댐 직하류 Umbra SAR 3장 · PlanetScope 3장(2026-09~10). 시립대 공식: 부산 모델 그대로 추론한 Umbra IoU 0.755~0.772(PS 문턱 지도 대비), PlanetScope 는 사전 품질검사 FAIL → 위성 수위는 미공표. 우리 자체 전처리 교차 IoU 0.68~0.71 은 참고값. 다음은 QGIS 초벌 라벨 수정 → 부산+대청 파인튜닝"],
  ];
  items.forEach(([n, head, body], i) => {
    const y = 2.75 + i * 0.8;
    s.addShape(pres.ShapeType.ellipse, { x: 0.9, y: y + 0.05, w: 0.5, h: 0.5, fill: { color: HX.accent4 }, line: { color: HX.accent4, width: 0 }, objectName: `번호 원 ${n}` });
    s.addText(n, { x: 0.9, y: y + 0.05, w: 0.5, h: 0.5, fontSize: 14, bold: true, color: HX.dk2, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `번호 ${n}` });
    s.addText([{ text: head + "  ", options: { bold: true, color: HX.lt1 } }, { text: body, options: { color: "C9D6E3" } }], { x: 1.6, y, w: 10.9, h: 0.65, fontSize: 13, margin: 0, valign: "middle", isTextBox: true, objectName: `다음 단계 ${n}` });
  });
  s.addNotes("K-water 자료 수치: docs/kwater_202610/RESULTS.md (서버 처리 결과). 나머지는 한계 슬라이드의 항목을 뒤집은 것.");
}

// ───────────────────────────── 저장
(async () => {
  await pres.writeFile({ fileName: OUT });
  if (SKILL_DIR) {
    const { applyTheme } = require(path.join(SKILL_DIR, "scripts/apply_theme.js"));
    await applyTheme(OUT, THEME);
  } else {
    console.warn("PPTX_SKILL_DIR 미지정 — 테마 색을 적용하지 않았다(스킴 색이 Office 기본 팔레트로 보인다)");
  }
  console.log("saved", OUT);
})();
