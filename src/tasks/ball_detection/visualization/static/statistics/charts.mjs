export function node(tag, text = '', className = '') {
  const el = document.createElement(tag);
  el.textContent = text;
  el.className = className;
  return el;
}
export function number(value) {
  return value === null || value === undefined ? '—' : Number(value).toLocaleString(undefined, {maximumFractionDigits: 4});
}
export function table(headings, rows) {
  const wrap = node('div', '', 'statistics-table-scroll');
  const t = node('table');
  const header = node('tr'); headings.forEach(h => header.append(node('th', h)));
  const thead = node('thead'); thead.append(header); t.append(thead);
  const body = node('tbody');
  rows.forEach(values => { const tr = node('tr'); values.forEach(value => {
    const td = node('td'); value instanceof Node ? td.append(value) : td.textContent = String(value); tr.append(td);
  }); body.append(tr); });
  t.append(body); wrap.append(t); return wrap;
}

const WORDS = {annotation:'注釈', instances:'球単位', frames:'frame単位', reviewed:'確認済み', unreviewed:'未確認', target:'担当区間', reference:'参照区間', reviewed_empty:'確認済み・球なし', multi_ball:'複数球', located:'座標あり', evidence:'区間選択の証拠', supervised:'座標教師', occluded:'遮蔽', boundary:'注釈等の境界', observed:'実測', interpolated:'補間', occlusion_estimated:'遮蔽推定', unresolved:'位置未確定', out_of_frame:'画面外', occluded_located:'遮蔽・座標あり', occluded_unlocated:'遮蔽・座標なし', notes_frames:'原文メモあり', original_available:'原本取得可能', counts:'件数', rates:'割合', mean:'平均', median:'中央値', min:'最小', max:'最大', gaps:'欠損', coordinate_gap:'座標欠損', observed_gap:'実測欠損', located_run:'連続座標あり', raw:'生の連続列', bounded:'境界で区切る', seconds:'秒', interpolation:'補間', run_frames:'連続frame数', run_seconds:'連続秒数', endpoint_distance:'端点間距離', endpoint_seconds:'端点間時間', spatial:'位置分布', motion:'動き', speed_px:'速度px/秒', speed_normalized:'正規化速度', step:'隣接移動量', abs_vx:'水平速度', abs_vy:'垂直速度', vertical_share:'垂直移動割合', acceleration:'加速度', turn:'方向変化', speed:'速度', relative_speed:'人物移動を除いた速度', spike:'一瞬の突出', score:'関節スコア', flagged:'飛び候補', person_observed:'人物観測', all_people_missing:'全人物欠損', translation_speed:'人物全体の移動', bone_change:'骨格長の急変', swap_advantage:'左右交換の対応改善', recovery_step:'欠損からの復帰変位', available_frames:'pose利用可能', left_wrist:'左手首', right_wrist:'右手首', left_shoulder:'左肩', right_shoulder:'右肩', left_elbow:'左肘', right_elbow:'右肘', left_hip:'左腰', right_hip:'右腰', left_knee:'左膝', right_knee:'右膝', left_ankle:'左足首', right_ankle:'右足首'};
export function label(key) {return key.split('/').map(word=>WORDS[word]??word).join(' / ');}
export function percent(value) {return value===null||value===undefined?'—':`${number(value*100)}%`;}
