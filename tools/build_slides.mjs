// Build bilingual, editable six-slide decks from the completed DS02 experiments.
// Run with the bundled Node runtime after metrics and benchmark outputs exist.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';

const workspace = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = process.env.RUNTIME_NODE_MODULES ?? 'C:/Users/sunghyun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const python = process.env.RUNTIME_PYTHON ?? 'C:/Users/sunghyun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const skill = process.env.PRESENTATIONS_SKILL_DIR ?? 'C:/Users/sunghyun/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
process.env.RUNTIME_NODE_MODULES = runtime;
const requireRuntime = createRequire(path.join(runtime, '__slides__.cjs'));
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(requireRuntime.resolve('@oai/artifact-tool')).href);
const { finalizePresentation, applyPresentationChartFont, resolvePresentationFont } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont({ fontFamily: 'Malgun Gothic' });
const output = path.join(workspace, 'output');
const build = path.join(workspace, 'tools', '.slides-build', String(Date.now()));
await fs.mkdir(build, { recursive: true });
// A fresh process avoids renderer state leaking from authoring/importing a deck.
const renderScript = path.join(build, 'render_verified.mjs');
await fs.writeFile(renderScript, `import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const req = createRequire(path.join(${JSON.stringify(runtime)}, '__render__.cjs'));
const { PresentationFile, FileBlob } = await import(pathToFileURL(req.resolve('@oai/artifact-tool')).href);
const deck = await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
for (const [index, slide] of deck.slides.items.entries()) {
  const blob = await deck.export({ slide, format: 'png', scale: 1 });
  await fs.writeFile(path.join(process.argv[3], 'slide-' + (index + 1) + '.png'), new Uint8Array(await blob.arrayBuffer()));
}
`);
const metricsPath = path.join(output, 'results', 'metrics.json');
const benchmarkPath = path.join(output, 'benchmark', 'benchmark_summary.csv');
const metrics = JSON.parse(await fs.readFile(metricsPath, 'utf8'));
const sensitivity = JSON.parse(await fs.readFile(path.join(output, 'extended/sensitivity/metrics.json'), 'utf8'));
const cnn = JSON.parse(await fs.readFile(path.join(output, 'extended/cnn/metrics.json'), 'utf8'));
const finalCard = JSON.parse(await fs.readFile(path.join(output, 'final/model_card.json'), 'utf8'));
const inference = JSON.parse(await fs.readFile(path.join(output, 'predictions/prediction_manifest.json'), 'utf8'));
if (finalCard.selected_candidate !== 'cnn_1d') throw new Error('This completed-project narrative requires the measured CNN winner; review it before changing models.');
if (inference.input_target_required !== false) throw new Error('Expected completed target-free inference verification.');

function csvRows(text) {
  const rows = []; let row = [], cell = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') { if (quoted && text[i + 1] === '"') { cell += '"'; i++; } else quoted = !quoted; }
    else if (c === ',' && !quoted) { row.push(cell); cell = ''; }
    else if (c === '\n' && !quoted) { row.push(cell.replace(/\r$/, '')); if (row.some(Boolean)) rows.push(row); row = []; cell = ''; }
    else cell += c;
  }
  if (cell || row.length) { row.push(cell.replace(/\r$/, '')); rows.push(row); }
  const header = rows.shift();
  return rows.map(values => Object.fromEntries(header.map((key, i) => [key.replace(/^\uFEFF/, ''), values[i]])));
}
const benchmark = csvRows(await fs.readFile(benchmarkPath, 'utf8'));
const cycleCurves = csvRows(await fs.readFile(path.join(output, 'extended/cnn/test_cycle_predictions.csv'), 'utf8')).map(row => ({unit:Number(row.unit), cycle:Number(row.cycle), target:Number(row.target), prediction:Number(row.prediction)}));
const benchmarkManifest = JSON.parse(await fs.readFile(path.join(output, 'benchmark/benchmark_manifest.json'), 'utf8'));
// Read a small observed trace only for the explanatory chart. No model reruns.
const traceRead = spawnSync(process.env.HDF5_PYTHON ?? 'C:/Users/sunghyun/anaconda3/python.exe', ['-c', `
import h5py,json,numpy as np,sys
with h5py.File(sys.argv[1],'r') as f:
 names=[v.decode() for v in np.asarray(f['X_s_var']).reshape(-1)]
 anames=[v.decode() for v in np.asarray(f['A_var']).reshape(-1)]
 values=np.asarray(f['X_s_dev'][:180,names.index('T24')],dtype=float)
 meta=np.asarray(f['A_dev'][:180])
 assert np.all(meta[:,anames.index('unit')]==meta[0,anames.index('unit')])
 assert np.all(meta[:,anames.index('cycle')]==meta[0,anames.index('cycle')])
 assert np.std(values)>0
 print(json.dumps({'variable':'T24','unit':int(meta[0,anames.index('unit')]),'cycle':int(meta[0,anames.index('cycle')]),'values':((values-values.mean())/values.std()).round(6).tolist(),'display_normalization':'z-score of these 180 samples only; not model preprocessing'}))
`, path.join(workspace, 'data/raw/N-CMAPSS_DS02-006.h5')], {encoding:'utf8'});
if (traceRead.status !== 0) throw new Error(`Could not read actual sensor trace: ${traceRead.stderr}`);
const trace = JSON.parse(traceRead.stdout);
function numeric(row, names) {
  for (const name of names) if (row[name] !== undefined && row[name] !== '' && Number.isFinite(Number(row[name]))) return Number(row[name]);
  throw new Error(`Missing numeric benchmark field ${names.join('/')} in ${JSON.stringify(row)}`);
}
const speedRows = benchmark.filter(row => Number(row.chunk_rows) === 120000).map(row => ({
  workers: numeric(row, ['workers', 'worker_count']),
  seconds: numeric(row, ['median_seconds', 'wall_seconds_median', 'median_wall_seconds', 'seconds_median', 'elapsed_seconds_median']),
  original: row,
})).sort((a, b) => a.workers - b.workers);
if (speedRows.length !== 4 || speedRows.map(row => row.workers).join(',') !== '1,2,4,8') throw new Error('Expected benchmark rows for exactly 1, 2, 4 and 8 workers');
const baseSeconds = speedRows[0].seconds;
for (const row of speedRows) row.speedup = baseSeconds / row.seconds;
const best = speedRows.reduce((a, b) => a.seconds < b.seconds ? a : b);
const selected = finalCard.selected_candidate;
const selectedMetrics = finalCard.test_metrics.cycles;
const dummyRmse = metrics.test_metrics.dummy_mean.cycles.macro_engine_rmse;
const improvement = 100 * (dummyRmse - selectedMetrics.macro_engine_rmse) / dummyRmse;
const fmt = (number, digits = 2) => Number(number).toFixed(digits);
const originalRidge = metrics.test_metrics.ridge.cycles;
const originalRidgeCV = metrics.cv_summary.find(row => row.model === 'ridge').macro_engine_cycle_rmse;
const source = {
  paper: 'Arias Chao et al. (2021), Data 6(1), 5. https://doi.org/10.3390/data6010005',
  nasa: 'NASA PCoE repository, dataset 17: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/',
  mirror: 'DS02-006 HDF5 mirror, Hao Li (2022): https://figshare.com/articles/dataset/N-CMAPSS_DS02-006_h5/20436504 . This is a third-party mirror, not a NASA standalone download.',
  metrics: 'Preserved initial baseline: output/results/metrics.json and output/results/cv_results.csv. Experiments completed 2026-10-02.',
  sensitivity: 'Completed exploratory sensitivity: output/extended/sensitivity/protocol.json, summary.csv and metrics.json. 24 candidates, 36 CV estimator fits, fixed development engine folds. Clipping applies per window before cycle averaging.',
  cnn: 'Completed CNN: output/extended/cnn/protocol.json, metrics.json and verification.json. Final choice: output/final/model_card.json. Extension test results reuse previously inspected test engines.',
  inference: 'Target-free prediction: output/predictions/prediction_manifest.json; src/ncmapss_rul/inference.py. Prediction does not require or read Y.',
  benchmark: 'Project experiment: output/benchmark/benchmark_summary.csv and benchmark_manifest.json. One full warmup; OS cache is not flushed. Runtime includes feature extraction and process overhead. Windows local multiprocessing.',
};
const theme = { ink: '#122536', muted: '#536171', blue: '#087F8C', navy: '#234B67', orange: '#D77938', light: '#EEF3F6', line: '#CED8E0', pale: '#E6F3F3' };

function textbox(slide, text, left, top, width, height, size = 28, color = theme.ink, bold = false) {
  const shape = slide.shapes.add({ geometry: 'textbox', position: { left, top, width, height }, fill: 'none', line: { fill: 'none', width: 0 } });
  shape.text = text;
  shape.text.style = { typeface: font, fontSize: size, color, bold, autoFit: 'none' };
  return shape;
}
function base(presentation, title, number, lang) {
  const slide = presentation.slides.add();
  slide.background.fill = '#FFFFFF';
  textbox(slide, title, 72, 45, 1136, 90, 44, theme.ink, true);
  textbox(slide, 'DCCS 410  김성현  2021271250', 72, 671, 1000, 27, 16, theme.muted);
  textbox(slide, String(number).padStart(2, '0'), 1160, 671, 50, 27, 16, theme.muted);
  return slide;
}
function table(slide, values, left, top, widths, height, size = 26) {
  const result = slide.tables.add({ rows: values.length, columns: widths.length, left, top, width: widths.reduce((a, b) => a + b), height, columnWidths: widths, values });
  result.styleOptions = { headerRow: true, bandedRows: false };
  result.cells.block({ row: 0, column: 0, rowCount: values.length, columnCount: widths.length }).assign({
    fill: '#FFFFFF', textStyle: { typeface: font, fontSize: size, color: theme.ink },
    margins: { left: 16, right: 12, top: 10, bottom: 10 },
  });
  result.borders.assign({ fill: theme.line, width: 1, style: 'solid' });
  for (let i = 0; i < widths.length; i++) {
    result.getCell(0, i).fill = theme.light;
    result.getCell(0, i).text.style = { typeface: font, fontSize: size, bold: true, color: theme.ink };
  }
  return result;
}
function notes(slide, script, citations) {
  slide.speakerNotes.textFrame.setText(script + '\n\nSources\n' + citations.join('\n'));
}

function node(slide, label, x, y, w, h, fill = theme.light, size = 25) {
  const shape = slide.shapes.add({ geometry:'rect', position:{left:x,top:y,width:w,height:h}, fill, line:{fill:theme.line,width:1.4} });
  shape.text = label;
  shape.text.style = {typeface:font,fontSize:size,color:theme.ink,bold:false,alignment:'center',verticalAlignment:'middle',autoFit:'none',insets:{left:9,right:9,top:7,bottom:7}};
  return shape;
}
function link(slide, a, b, from='right', to='left') {
  return slide.shapes.connect(a,b,{kind:'elbow',fromSide:from,toSide:to,line:{fill:theme.navy,width:2},tail:{type:'triangle',width:'sm',length:'sm'}});
}
function chart(slide, type, options) {
  const value = slide.charts.add(type, options);
  applyPresentationChartFont(value,{fontFamily:font});
  return value;
}
const round = number => Number(fmt(number,6));
const axisStyle = {fontSize:20,typeface:font,fill:theme.muted};
const grid = {fill:theme.line,width:1};

async function makeDeck(lang) {
  const ko = lang==='ko';
  const p = Presentation.create({slideSize:{width:1280,height:720}});
  let s=p.slides.add(); s.background.fill='#FFFFFF';
  textbox(s,ko?'항공기 엔진의\n남은 수명 예측':'Predicting aircraft\nengine remaining life',72,108,650,178,57,theme.ink,true);
  textbox(s,ko?'N-CMAPSS DS02\n관측 센서로 남은 비행 사이클 예측':'N-CMAPSS DS02\nRemaining flight cycles from observed sensors',76,320,660,100,29,theme.navy);
  textbox(s,fmt(selectedMetrics.macro_engine_rmse),76,475,225,100,78,theme.blue,true);
  textbox(s,ko?'사이클 RMSE\nCNN 재사용 시험 결과':'cycles RMSE\nCNN on reused test data',315,494,360,88,26,theme.ink);
  const pipeline=[
    node(s,ko?'HDF5\n약 650만 시점':'HDF5\n6.5M time points',815,104,345,85,theme.light,25),
    node(s,ko?'18개 관측 변수\n60개 표본 구간':'18 observable channels\n60-sample windows',815,241,345,85,theme.pale,25),
    node(s,ko?'작은 1D CNN\n개발 엔진 검증으로 선택':'Small 1D CNN\nSelected by development CV',815,378,345,90,theme.pale,24),
    node(s,ko?'비행별 RUL 예측':'RUL for each flight',815,520,345,75,'#FDF0E6',27),
  ];
  pipeline.slice(1).forEach((n,i)=>link(s,pipeline[i],n,'bottom','top'));
  textbox(s,'김성현  2021271250\nDCCS 410  2026-10-02',76,620,650,62,22,theme.muted);
  notes(s,ko
    ? '발표 시간 0:00-0:35. 안녕하세요. 김성현입니다. 이 프로젝트는 관측 센서로 항공기 엔진의 남은 비행 사이클을 예측합니다. 오른쪽 흐름처럼 대용량 HDF5를 구간으로 나누고, 개발 엔진 검증으로 선택한 모델을 저장해 비행별 RUL을 생성합니다. 약 육백오십만 시점의 데이터를 처리했고, 최종 모델은 작은 일차원 CNN입니다. 화면의 오차는 이미 살펴본 시험 세트를 재사용한 평가입니다. 데이터 처리와 예측 성능을 함께 확인한 실제 완료 결과를 보여드리겠습니다.'
    : 'Timing 0:00-0:35. Hello, I am 김성현. This project estimates the flight cycles remaining in an engine from observed sensors. The diagram follows the completed workflow: read HDF5, form sensor windows, select a model using development engines, and generate RUL predictions. The dataset contains about six and a half million time points. A small one-dimensional CNN is the final model. The displayed error comes from evaluation on previously inspected test data. I will show the completed processing workflow, model comparison and measured parallel performance.',[source.paper,source.cnn,source.inference]);

  s=base(p,ko?'센서 구간과 엔진별 검증':'Sensor windows and validation by engine',2,lang);
  textbox(s,ko?'같은 비행 안의 60개 표본씩':'60 samples at a time within one flight',76,143,710,50,30,theme.blue,true);
  const colors=[theme.navy,theme.blue,theme.orange];
  chart(s,'scatter',{
    position:{left:63,top:210,width:713,height:358},
    series:[0,1,2].map(i=>({name:ko?`구간 ${i+1}`:`Window ${i+1}`,xValues:Array.from({length:60},(_,j)=>i*60+j+1),values:trace.values.slice(i*60,(i+1)*60),line:{fill:colors[i],width:3},marker:{symbol:'none'},valuesFormatCode:'0.00',xValuesFormatCode:'0'})),
    scatterOptions:{style:'line'},hasLegend:true,legend:{position:'bottom',textStyle:{...axisStyle,fontSize:21}},
    xAxis:{title:ko?'표본 순서':'Sample index',min:0,max:180,majorUnit:60,textStyle:axisStyle,majorGridlines:null},
    yAxis:{title:'T24 (z)',textStyle:axisStyle,majorGridlines:grid},
  });
  textbox(s,ko?`실제 개발 엔진 ${trace.unit}, 비행 ${trace.cycle}\n표시용 표준화, 모델 학습과 별도`:`Actual development engine ${trace.unit}, flight ${trace.cycle}\nStandardized for this display only`,80,583,700,63,22,theme.muted);
  textbox(s,ko?'개발 엔진 6개':'Six development engines',815,161,382,54,29,theme.navy,true);
  const units=[2,5,10,16,18,20], validation=[[5,16],[10,20],[2,18]];
  const fold=table(s,[['CV',...units.map(String)],...validation.map((v,i)=>[String(i+1),...units.map(u=>v.includes(u)?'V':'T')])],815,255,[64,52,52,52,52,52,52],195,21);
  fold.cells.block({row:0,column:0,rowCount:4,columnCount:7}).assign({margins:{left:4,right:4,top:8,bottom:8}});
  for(let row=1;row<=3;row++) for(let col=1;col<=6;col++) {
    const valid=validation[row-1].includes(units[col-1]);
    fold.getCell(row,col).fill=valid?theme.blue:'#F2F5F7';
    fold.getCell(row,col).text.style={typeface:font,fontSize:21,bold:valid,color:valid?'#FFFFFF':theme.muted,alignment:'center'};
  }
  textbox(s,ko?'T 학습     V 검증':'T training     V validation',816,470,380,42,23,theme.muted);
  textbox(s,ko?'시험 엔진 11, 14, 15\n확장 평가에서 재사용':'Test engines 11, 14, 15\nReused for extension evaluation',815,548,385,95,25,theme.ink);
  notes(s,ko
    ? '발표 시간 0:35-1:30. 왼쪽은 개발 엔진의 실제 T24 센서 기록 백팔십 개 표본입니다. 구간이 어디서 나뉘는지 보이도록 색을 달리했고, 모양을 읽기 위해 이 그림에서만 표준화했습니다. 모델의 표준화 방식과는 별개입니다. 최종 CNN은 같은 엔진의 같은 비행 안에서 육십 개 표본씩 사용하며 경계를 넘지 않습니다. 오른쪽 표에서 T는 학습, V는 검증 엔진입니다. 엔진 전체를 분리한 같은 세 구간을 모든 모델에 적용했습니다. 공식 개발 엔진은 여섯 개, 시험 엔진은 세 개입니다. 데이터는 실제 비행 조건을 반영한 열화 시뮬레이션입니다. 개발 비행 클래스는 모두 삼이며 시험은 일, 이, 삼을 포함합니다. 처음에 시험 결과를 이미 보았으므로 확장 시험은 탐색적 재사용 평가입니다.'
    : 'Timing 0:35-1:30. The left chart shows one hundred eighty actual T24 samples from a development engine. Colors distinguish consecutive sixty-sample windows. Standardization makes this display readable and is separate from model preprocessing. The final CNN uses windows within one engine and flight, never across those boundaries. On the right, T indicates training and V indicates validation. All model families share these three splits of entire engines. There are six official development engines and three test engines. The data simulate degradation under operating conditions drawn from real flights. All development engines have flight class three, while the tests include classes one, two and three. Since initial test results had already been inspected, the extension uses exploratory test evaluation.',[source.paper,source.sensitivity,source.cnn,`Sensor chart: first 180 rows of X_s_dev, channel T24, engine ${trace.unit}, cycle ${trace.cycle}. Z-score uses only the displayed 180 values for presentation. No model experiment was rerun.`]);

  s=base(p,ko?'회귀 모델과 CNN의 입력 경로':'Inputs to regression and the CNN',3,lang);
  textbox(s,ko?'통계 회귀':'Regression with statistics',310,143,520,45,28,theme.navy,true);
  const input=node(s,ko?'관측 입력\n18개 변수':'Observed\n18 channels',73,296,165,114,theme.pale,26);
  const stats=node(s,ko?'5개 통계\n90개 특징':'Five statistics\n90 features',316,207,201,105,theme.light,26);
  const regress=node(s,'Ridge\nGradient boosting',590,207,220,105,theme.light,23);
  const rulNode=node(s,ko?'남은\n비행 사이클':'Remaining\nflight cycles',1050,296,160,114,'#FDF0E6',26);
  link(s,input,stats);link(s,stats,regress);link(s,regress,rulNode);
  textbox(s,'1D CNN',310,355,500,42,28,theme.blue,true);
  const cnnNodes=[node(s,'Conv1d 16\nkernel 5\nReLU',313,420,140,112,theme.pale,23),node(s,'Conv1d 32\nkernel 5\nReLU',493,420,140,112,theme.pale,23),node(s,'Average\npooling',673,420,140,112,theme.pale,23),node(s,'Linear\nSoftplus',853,420,140,112,theme.pale,23)];
  link(s,input,cnnNodes[0]);cnnNodes.slice(1).forEach((n,i)=>link(s,cnnNodes[i],n));link(s,cnnNodes[3],rulNode,'right','bottom');
  textbox(s,ko?'60개 표본 구간':'60-sample\nwindow',76,429,190,75,22,theme.muted);
  textbox(s,ko?'CNN: 4,081개 파라미터, 12 epochs, RTX 4070 Ti SUPER':'CNN: 4,081 parameters, 12 epochs on RTX 4070 Ti SUPER',75,557,1135,47,26,theme.blue,true);
  textbox(s,ko?'민감도: 6개 구간 설정 × 2개 모델 × 2개 정책 = 24개 후보':'Sensitivity: six window settings × two models × two policies = 24 candidates',75,616,1135,42,23,theme.muted);
  notes(s,ko
    ? '발표 시간 1:30-2:25. 입력은 운항 조건 네 개와 관측 센서 열네 개입니다. 내부 열화 변수 T, 건강 상태 hs, 가상 센서 X_v와 식별자는 예측 변수에서 제외했습니다. 위 경로는 각 변수의 평균, 표준편차, 최소, 최대와 기울기를 계산해 아흔 개 특징을 만듭니다. Ridge와 Gradient boosting을 여섯 구간 설정과 두 출력 정책으로 비교했습니다. 스물네 후보지만 같은 모델의 원래 예측과 영 하한 예측은 학습 결과를 공유해 교차 검증 학습은 서른여섯 번입니다. 아래 CNN은 열여덟 채널의 육십 개 표본을 직접 받습니다. 합성곱 두 층 이후 시간 평균과 선형층을 통과하고 Softplus로 양수 RUL을 출력합니다. 사천팔십일 개 파라미터를 열두 에포크 동안 GPU로 학습했습니다.'
    : 'Timing 1:30-2:25. Inputs comprise four operating conditions and fourteen observed sensors. Latent parameters T, health state hs, virtual sensors X_v and identifiers are excluded. The upper path computes mean, standard deviation, minimum, maximum and slope for each channel, producing ninety features. Ridge and gradient boosting are compared across six window settings and two output policies. The twenty-four candidates require thirty-six cross-validation fits because raw and zero-clipped predictions share a fitted estimator. The lower path sends eighteen channels by sixty samples directly into the CNN. Two convolution layers feed temporal average pooling, a linear layer and Softplus for positive RUL. The network has four thousand eighty-one parameters and trains for twelve epochs on the GPU.',[source.sensitivity,source.cnn,'CNN architecture diagram follows the saved metrics.json architecture exactly. ReLU follows both convolutions.']);

  s=base(p,ko?'개발 검증에서 CNN의 오차가 가장 낮다':'The CNN has the lowest development error',4,lang);
  const meanCV=metrics.cv_summary.find(row=>row.model==='dummy_mean').macro_engine_cycle_rmse;
  chart(s,'bar',{
    position:{left:55,top:184,width:850,height:414},categories:ko?['평균 기준선','Ridge\n60/60','Ridge\n30/15 + clip','CNN\n60/60']:['Mean','Ridge\n60/60','Ridge\n30/15 + clip','CNN\n60/60'],
    series:[{name:ko?'개발 CV':'Development CV',values:[meanCV,originalRidgeCV,sensitivity.cv_macro_engine_cycle_rmse,cnn.cv_macro_engine_cycle_rmse].map(round),fill:theme.navy,valuesFormatCode:'0.00'},
      {name:ko?'시험 평가':'Test evaluation',values:[dummyRmse,originalRidge.macro_engine_rmse,sensitivity.test_metrics.cycles.macro_engine_rmse,selectedMetrics.macro_engine_rmse].map(round),fill:theme.orange,valuesFormatCode:'0.00'}],
    barOptions:{direction:'column',grouping:'clustered',gapWidth:70},hasLegend:true,legend:{position:'bottom',textStyle:axisStyle},
    xAxis:{textStyle:{...axisStyle,fontSize:21},majorGridlines:null},yAxis:{title:ko?'RMSE (사이클)':'RMSE (cycles)',min:0,max:25,majorUnit:5,textStyle:axisStyle,majorGridlines:grid},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{...axisStyle,fontSize:20,fill:theme.ink,bold:true}},
  });
  textbox(s,fmt(selectedMetrics.macro_engine_rmse),946,200,270,100,77,theme.blue,true);
  textbox(s,ko?'CNN 시험 RMSE':'CNN test RMSE',948,308,270,43,24,theme.ink);
  textbox(s,fmt(selectedMetrics.macro_engine_mae),946,410,270,90,61,theme.orange,true);
  textbox(s,ko?'CNN 시험 MAE':'CNN test MAE',948,501,270,42,24,theme.ink);
  textbox(s,ko?'모델 선택은 개발 CV만 사용. 시험 점수는 재사용 평가' : 'Model selection uses development CV only. Test scores reuse the test set',76,620,1120,43,24,theme.muted);
  notes(s,ko
    ? `발표 시간 2:25-3:20. 막대의 남색은 개발 교차 검증, 주황색은 시험 오차입니다. 모델 선택은 남색 막대만 사용했습니다. 초기 Ridge의 개발 RMSE는 ${fmt(originalRidgeCV)}, 민감도 실험의 최선 Ridge는 ${fmt(sensitivity.cv_macro_engine_cycle_rmse)}였으며, CNN은 ${fmt(cnn.cv_macro_engine_cycle_rmse)}로 가장 낮았습니다. 민감도 실험의 최선은 삼십 표본 구간과 십오 표본 간격, 영 하한 정책입니다. 다만 이 구간 설정의 이점은 여섯 엔진에서 비교한 작은 차이입니다. 최종 CNN의 재사용 시험 RMSE는 ${fmt(selectedMetrics.macro_engine_rmse)}, MAE는 ${fmt(selectedMetrics.macro_engine_mae)} 사이클입니다. 동일 비행의 구간 예측을 평균하고 엔진별 오차를 같은 비중으로 평균했습니다. 초기 Ridge의 음수 예측과 결과 파일은 보존했습니다. 이 점수들은 실제 항공기 정비 성능을 증명하지 않습니다.`
    : `Timing 2:25-3:20. Navy bars show development cross-validation and orange bars show test errors. Only the navy values drive model selection. Original Ridge development RMSE is ${fmt(originalRidgeCV)}, the sensitivity winner reaches ${fmt(sensitivity.cv_macro_engine_cycle_rmse)}, and the CNN reaches ${fmt(cnn.cv_macro_engine_cycle_rmse)}. The best regression setting uses thirty-sample windows, stride fifteen and zero clipping. Window-setting differences are modest within these six engines. On reused test data, the final CNN has RMSE ${fmt(selectedMetrics.macro_engine_rmse)} and MAE ${fmt(selectedMetrics.macro_engine_mae)} cycles. Predictions are averaged within each flight, and errors receive equal weight across engines. The original Ridge files and negative predictions remain preserved. These scores describe this simulation dataset rather than validating aircraft maintenance performance.`,[source.metrics,source.sensitivity,source.cnn]);

  s=base(p,ko?'시험 엔진의 정답과 CNN 예측':'Reference RUL and CNN predictions',5,lang);
  for(const [i,unit] of [11,14,15].entries()) {
    const x=65+i*398, points=cycleCurves.filter(row=>row.unit===unit).sort((a,b)=>a.cycle-b.cycle);
    const metric=selectedMetrics.per_engine.find(row=>row.unit===unit);
    textbox(s,ko?`엔진 ${unit}`:`Engine ${unit}`,x+15,145,355,45,29,theme.navy,true);
    textbox(s,`RMSE ${fmt(metric.rmse)}`,x+15,193,355,37,22,theme.muted);
    chart(s,'scatter',{
      position:{left:x,top:250,width:382,height:332},
      series:[{name:ko?'정답 RUL':'Reference RUL',xValues:points.map(r=>r.cycle),values:points.map(r=>r.target),line:{fill:theme.navy,width:3},marker:{symbol:'none'},valuesFormatCode:'0.00',xValuesFormatCode:'0'},
        {name:ko?'CNN 예측':'CNN prediction',xValues:points.map(r=>r.cycle),values:points.map(r=>round(r.prediction)),line:{fill:theme.orange,width:2.5},marker:{symbol:'none'},valuesFormatCode:'0.00',xValuesFormatCode:'0'}],
      scatterOptions:{style:'line'},hasLegend:true,legend:{position:'bottom',textStyle:{...axisStyle,fontSize:17}},
      xAxis:{title:ko?'비행 사이클':'Flight cycle',min:0,max:80,majorUnit:20,textStyle:{...axisStyle,fontSize:18},majorGridlines:null},
      yAxis:{title:'RUL',min:0,max:80,majorUnit:20,textStyle:{...axisStyle,fontSize:18},majorGridlines:grid},
    });
  }
  textbox(s,ko?'정답 Y를 읽지 않는 예측 경로에서도 202개 비행의 결과 생성':'The prediction path also produces results for 202 flights without reading Y',76,592,1130,42,27,theme.blue,true);
  textbox(s,ko?'정답 곡선은 평가에만 사용. 비행 종료 시점의 평균 예측' : 'Reference curves support evaluation. Predictions average windows at flight end',76,636,1130,28,19,theme.muted);
  notes(s,ko
    ? '발표 시간 3:20-4:10. 세 그래프는 시험 엔진 전체를 보여줍니다. 남색은 정답 RUL, 주황색은 저장한 CNN의 비행별 예측입니다. 축 범위를 동일하게 두어 엔진 사이의 차이를 비교할 수 있습니다. 엔진 십일에서 오차가 가장 크고, 십오에서 가장 작습니다. 예측은 수명 감소를 따라가지만 일부 구간에서 정답과 상당히 차이 납니다. 이 그래프는 실제 저장된 예측 파일을 사용했습니다. 별도의 추론 경로는 정답 배열 Y를 읽지 않고 이만 칠백구십육 개 구간과 이백이 개 비행의 예측을 생성했습니다. 그래프의 정답은 성능 평가를 위해 표시했으며 예측 입력이 아닙니다. 비행 안의 모든 구간 예측을 평균하므로 비행이 끝난 시점의 추정입니다.'
    : 'Timing 3:20-4:10. These charts show all three test engines. Navy is the reference RUL and orange is the saved CNN prediction for each flight. Common axis ranges make engine differences visible. Engine eleven has the largest error and engine fifteen the smallest. Predictions generally follow the decline in life, but substantial deviations remain in parts of the histories. Every curve comes from the saved prediction file. A separate inference path generates twenty thousand seven hundred ninety-six window predictions and two hundred two flight predictions without reading target Y. Targets appear here only for evaluation and never enter the prediction inputs. Averaging all windows within a flight makes this an estimate available at flight end.',[source.cnn,source.inference,'Curves: output/extended/cnn/test_cycle_predictions.csv. All official test engines are shown, using common 0–80 axes.']);

  s=base(p,ko?'4개 작업자에서 중앙 처리 시간이 가장 짧다':'Four workers give the shortest median time',6,lang);
  chart(s,'bar',{
    position:{left:55,top:186,width:835,height:384},categories:speedRows.map(r=>String(r.workers)),
    series:[{name:ko?'최소':'Minimum',values:speedRows.map(r=>round(Number(r.original.min_seconds))),fill:'#C3D2DB',valuesFormatCode:'0.00'},
      {name:ko?'중앙값':'Median',values:speedRows.map(r=>round(r.seconds)),fill:theme.blue,valuesFormatCode:'0.00',dataLabelOverrides:speedRows.map((r,i)=>({idx:i,showValue:true,position:'outEnd',textStyle:{...axisStyle,fontSize:20,bold:true,fill:theme.ink}}))},
      {name:ko?'최대':'Maximum',values:speedRows.map(r=>round(Number(r.original.max_seconds))),fill:theme.orange,valuesFormatCode:'0.00',dataLabelOverrides:[{idx:2,showValue:true,position:'outEnd',textStyle:{...axisStyle,fontSize:20,bold:true,fill:theme.orange}}]}],
    barOptions:{direction:'column',grouping:'clustered',gapWidth:70},hasLegend:true,legend:{position:'bottom',textStyle:axisStyle},
    xAxis:{title:ko?'작업자 수':'Workers',textStyle:axisStyle,majorGridlines:null},yAxis:{title:ko?'처리 시간 (초)':'Time (seconds)',min:0,max:2.6,majorUnit:.5,textStyle:axisStyle,majorGridlines:grid},
    dataLabels:{showValue:false},
  });
  textbox(s,`${fmt(best.speedup)}×`,940,201,275,100,73,theme.blue,true);
  textbox(s,ko?'4개 작업자\n중앙시간 기준':'Four workers\nMedian speedup',943,308,270,88,27,theme.ink);
  textbox(s,ko?'최대 2.29초\n실행 간 변동 존재':'Maximum 2.29 s\nRuns vary',943,440,272,84,25,theme.orange);
  textbox(s,ko?'3회 반복, OS 캐시 워밍업, 청크 120,000행, 구간/간격 60/60':'Three repeats, warm OS cache, 120,000-row chunks, window/stride 60/60',76,588,1130,42,23,theme.muted);
  textbox(s,ko?'해석의 범위: 엔진 9개 시뮬레이션, 시험 재사용, 외부·실항공기 검증 미완료':'Scope: nine simulated engines, reused tests, no external or real-aircraft validation',76,637,1130,45,22,theme.ink);
  notes(s,ko
    ? `발표 시간 4:10-5:00. 마지막은 CPU 통계 특징 추출의 반복 측정입니다. 최소와 중앙, 최대를 함께 표시했습니다. 여덟 코어 CPU와 삼십이 기가바이트 메모리에서 OS 캐시를 워밍업한 후 세 번씩 실행했습니다. 네 작업자의 중앙값은 ${fmt(best.seconds)}초로 한 작업자보다 ${fmt(best.speedup)}배 빨랐지만, 최대는 2.285초였습니다. 이 변동도 결과에 포함했습니다. 파일 읽기와 프로세스 비용을 포함하며 GPU 학습과는 별도로 측정했습니다. 저장장치 경합과 메모리 대역폭, Windows 시작 비용은 가능한 설명이지만 원인별 시간을 분리 측정하지는 않았습니다. 결과적으로 정답 없는 예측까지 연결된 구현을 완료했습니다. 다만 엔진 아홉 개의 시뮬레이션과 재사용 시험 결과이므로, 외부 자료와 실제 항공기에서 검증된 정비 모델이라고 주장하지 않습니다. 감사합니다.`
    : `Timing 4:10-5:00. The final chart includes minimum, median and maximum CPU extraction times. Each setting ran three times after warming the OS cache on an eight-core computer with thirty-two gigabytes of RAM. Four workers have a median of ${fmt(best.seconds)} seconds and a ${fmt(best.speedup)}-times speedup, but the maximum is 2.285 seconds. This variability remains visible. Timing includes file reads and process costs and is separate from GPU training. Storage contention, memory bandwidth and Windows startup are plausible explanations, but individual causes were not profiled. The completed implementation now reaches prediction without target labels. Its evidence remains limited to nine simulated engines and reused test outcomes, with no external or real-aircraft validation. Thank you.`,[source.benchmark,source.paper,source.cnn,'Four-worker measured times: minimum 1.4719171000033384 s, median 1.473291500005871 s, maximum 2.2850628000014694 s. Three measured repeats, warm OS cache.']);

  const candidate=path.join(build,`candidate_${lang}.pptx`);
  await (await PresentationFile.exportPptx(p)).save(candidate);
  const final=path.join(process.env.SLIDES_OUTPUT_DIR??output,`presentation_${lang}.pptx`);
  await fs.mkdir(path.dirname(final),{recursive:true});
  const result=await finalizePresentation({
    workspaceDir:workspace,candidatePath:candidate,finalPath:final,pythonExecutable:python,
    integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--validate-bullet-geometry','--require-native-table-slide','2'],
    explicitTotalSlideCount:6,requiredNativeTableOwnerSlides:[2],requiredNativeChartOwnerSlides:[2,4,5,6],materializeLiteralChartWorkbooks:true,
    fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,`validation_${lang}.json`),
  });
  const qa=path.join(output,'qa','slides',lang);await fs.mkdir(qa,{recursive:true});
  const render=spawnSync(process.execPath,[renderScript,final,qa],{encoding:'utf8'});
  if(render.status!==0) throw new Error(`Final-slide rendering failed: ${render.stderr||render.stdout}`);
  await fs.writeFile(path.join(qa,'metrics_used.json'),JSON.stringify({selected_model:selected,test:selectedMetrics,speedRows,trace,curve_source:'output/extended/cnn/test_cycle_predictions.csv',repeats:benchmarkManifest.repeats},null,2));
  console.log(JSON.stringify({language:lang,file:final,slides:6,receipt:result.receiptPath,qa}));
}

const languages=process.argv.slice(2);
if(languages.some(lang=>!['ko','en'].includes(lang))) throw new Error('Optional arguments must be ko and/or en');
for(const lang of languages.length?[...new Set(languages)]:['ko','en']) await makeDeck(lang);
