const $ = id => document.getElementById(id);
const statusLabels = {
  auto: '仮入力', confirmed: '確定', unknown: '不明', not_point: 'ポイント外', multiple_points: '複数ポイント',
};
const reasons = ['unknown', 'not_point', 'multiple_points'];

function countLabel(count, points) {
  if (points === '試合終了') {
    const result = count.sets ? `セット ${count.sets.join('–')}` : `ゲーム ${count.games.join('–')}`;
    return `${result} / ${points}`;
  }
  const parts = [];
  if (count.sets) parts.push(`セット ${count.sets.join('–')}`);
  parts.push(`ゲーム ${count.games.join('–')}`, `ポイント ${points}`);
  return parts.join(' / ');
}

export class Scoring {
  constructor(bind, previewAndEdit) {
    this.previewAndEdit = previewAndEdit;
    bind('open-match', () => {
      $('match-settings').open = true;
      $('match-a').focus();
    });
    bind('match-type', () => this.toggleMatchFields(), 'change');
    bind('match-form', () => this.saveMatch(), 'submit');
    for (const id of ['score-status', 'before-source']) {
      bind(id, () => this.toggleScoreFields(), 'change');
    }
    bind('score-form', () => this.saveScore(), 'submit');
    bind('winner-a', () => this.saveScore('A'));
    bind('winner-b', () => this.saveScore('B'));
    bind('score-clear', () => this.previewAndEdit({action: 'set_score', name: this.clip.name, score: null}));
    bind('score-csv-form', async () => {
      const file = $('score-csv-file').files[0];
      const revision = this.project.revision;
      const csvText = new TextDecoder('utf-8', {fatal: true}).decode(await file.arrayBuffer());
      if (await previewAndEdit({action: 'import_scores', csv_text: csvText}, revision)) {
        $('score-csv-file').value = '';
      }
    }, 'submit');
  }

  load(project) {
    this.project = project;
    const match = project.match;
    const rules = match?.rules;
    $('match-settings').hidden = false;
    $('match-current').textContent = match
      ? `${this.side('A')} / ${this.side('B')} · ${rules.type === 'tennis' ? '硬式テニス' : 'ソフトテニス'} · 最初のサーブ ${match.first_server}`
      : '試合設定は未入力';
    $('match-form').hidden = project.read_only;
    $('match-a').value = match?.sides.A ?? 'A';
    $('match-b').value = match?.sides.B ?? 'B';
    $('match-first-server').value = match?.first_server ?? 'A';
    for (const side of ['A', 'B']) {
      $(`winner-${side.toLowerCase()}`).textContent = `${this.side(side)} の得点（${side}）`;
    }
    $('match-type').value = rules?.type ?? 'soft_tennis';
    $('match-games').value = rules?.match_games ?? 7;
    $('match-best-of').value = rules?.best_of ?? 3;
    $('match-no-ad').checked = rules?.no_ad ?? false;
    $('match-final-set').value = rules?.final_set ?? 'tie_break';
    this.toggleMatchFields();
    $('score-csv-form').hidden = project.read_only || !match;
  }

  toggleMatchFields() {
    const tennis = $('match-type').value === 'tennis';
    $('match-soft').hidden = tennis;
    $('match-games').disabled = tennis;
    $('match-hard').hidden = !tennis;
  }

  saveMatch() {
    const type = $('match-type').value;
    const rules = type === 'soft_tennis' ? {type, match_games: $('match-games').valueAsNumber} : {
      type, best_of: Number($('match-best-of').value),
      no_ad: $('match-no-ad').checked, final_set: $('match-final-set').value,
    };
    return this.previewAndEdit({
      action: 'set_match',
      match: {
        rules, sides: {A: $('match-a').value, B: $('match-b').value},
        first_server: $('match-first-server').value,
      },
    });
  }

  side(side) {
    const name = this.project.match?.sides[side];
    return !name || name === side ? side : `${side} · ${name}`;
  }

  row(clip) {
    const score = clip.score;
    const view = this.project.score_views[clip.name];
    const label = document.createElement('small');
    label.className = 'clip-score';
    label.classList.toggle('unscored', !score);
    if (!score) {
      label.textContent = '未入力';
      return label;
    }
    const parts = [statusLabels[score.status]];
    if (view.before_points !== null) parts.unshift(`開始前 ${countLabel(score.before, view.before_points)}`);
    if (score.winner) parts.push(`得点 ${this.side(score.winner)}`);
    if (view.chain_broken) parts.push('連鎖切れ');
    if (view.discontinuous) parts.push('不連続');
    if (view.expected_server) parts.push('サーブ順の矛盾');
    label.textContent = parts.join(' · ');
    return label;
  }

  select(clip) {
    this.clip = clip;
    $('score-panel').hidden = !clip;
    if (!clip) return;
    const score = clip.score;
    const view = this.project.score_views[clip.name];
    $('score-current').textContent = this.row(clip).textContent;
    if (score && (!view.chain_broken || score.server_source === 'manual')) {
      $('score-current').textContent += ` · サーブ ${this.side(score.server)}（${score.server_source === 'auto' ? '自動' : '手動'}）`;
    }
    $('score-after').hidden = !view?.after;
    $('score-after').textContent = view?.after ? `終了後 ${countLabel(view.after, view.after.points_label)}` : '';
    $('score-saved-note').hidden = !score?.note;
    $('score-saved-note').textContent = score?.note ?? '';
    const warnings = [];
    if (view?.chain_broken) warnings.push('連鎖切れ。開始前カウントを手入力してください。');
    if (view?.discontinuous) warnings.push('開始前と直前の終了後カウントが一致しません。');
    if (view?.expected_server) {
      warnings.push(`サーブ順の矛盾。直前からの計算では ${this.side(view.expected_server)}、保存値は ${this.side(score.server)} です。`);
    }
    $('score-warning').hidden = !warnings.length;
    $('score-warning').textContent = warnings.join(' ');
    $('open-match').hidden = this.project.read_only || !!this.project.match;
    $('score-form').hidden = this.project.read_only || !this.project.match;
    if ($('score-form').hidden) return;
    $('score-status').value = score?.status ?? 'auto';
    const broken = view?.chain_broken;
    $('before-source').value = broken ? 'anchor' : (score?.before_source ?? (view?.chain_available ? 'chain' : 'anchor'));
    $('before-source').querySelector('option[value="chain"]').disabled = broken === true;
    $('score-server').value = score?.server_source === 'manual' ? score.server : 'auto';
    $('score-note').value = score?.note ?? '';
    $('score-clear').disabled = !score;
    for (const key of ['sets', 'games', 'points']) {
      $('score-' + key + '-a').value = broken ? '' : (score?.before?.[key]?.[0] ?? 0);
      $('score-' + key + '-b').value = broken ? '' : (score?.before?.[key]?.[1] ?? 0);
    }
    const hard = this.project.match.rules.type === 'tennis';
    document.querySelectorAll('.score-sets').forEach(element => {
      element.hidden = !hard;
      if (element instanceof HTMLInputElement) element.disabled = !hard;
    });
    $('score-input-hint').textContent = hard
      ? '通常ゲームの 0・15・30・40 は 0・1・2・3、AD は 4 対 3。タイブレークはポイント数。'
      : 'デュースは 3 対 3、アドバンテージは 4 対 3。ファイナルゲームはポイント数。';
    this.toggleScoreFields();
  }

  toggleScoreFields() {
    const status = $('score-status').value;
    const notPoint = status === 'not_point';
    const before = $('score-before');
    before.hidden = before.disabled = notPoint || $('before-source').value === 'chain';
    $('before-source-label').hidden = notPoint;
    $('score-note').required = reasons.includes(status);
    $('score-winners').hidden = reasons.includes(status);
  }

  winnerButton(key) {
    if ($('score-panel').hidden || $('score-form').hidden || $('score-winners').hidden) return null;
    return key === 'a' ? $('winner-a') : $('winner-b');
  }

  saveScore(winner = null) {
    const status = winner ? 'confirmed' : $('score-status').value;
    const note = $('score-note').value;
    const server = $('score-server').value;
    if (!$('score-form').reportValidity()) return;
    const score = {
      status, note,
      winner: status === 'confirmed' ? (winner ?? this.clip.score?.winner ?? null) : null,
      server: server === 'auto' ? null : server,
      server_source: server === 'auto' ? 'auto' : 'manual',
      before_source: status === 'not_point' ? 'chain' : $('before-source').value,
    };
    if (status === 'confirmed' && !score.winner) throw new Error('得点側 A / B を選んでください。');
    if (status !== 'not_point') {
      score.before = null;
      if (score.before_source === 'anchor') {
        const keys = this.project.match.rules.type === 'tennis' ? ['sets', 'games', 'points'] : ['games', 'points'];
        score.before = Object.fromEntries(keys.map(key => [
          key, ['a', 'b'].map(side => $(`score-${key}-${side}`).valueAsNumber),
        ]));
      }
    }
    return this.previewAndEdit({action: 'set_score', name: this.clip.name, score});
  }
}
