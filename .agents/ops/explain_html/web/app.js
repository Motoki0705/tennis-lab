(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const channel = 'explain-reader-v1',
    idPattern = /^[A-Za-z][A-Za-z0-9_-]{0,79}$/,
    revisionPattern = /^[a-f0-9]{32}$/;
  const fragment = new URLSearchParams(location.hash.slice(1));
  const documentId = fragment.get('document_id'),
    token = fragment.get('token');
  history.replaceState(null, '', location.pathname);
  const base = `/api/documents/${documentId}`,
    sessionKey = `explain-html:${documentId}:pending`;
  let displayed = null,
    current = null,
    historyRows = [],
    blocks = new Map(),
    activeFrame = null,
    staging = null;
  let selected = { kind: 'document' },
    readingState = null,
    pending = null,
    lastAttempt = null;
  let pollTimer = null,
    pollBusy = false,
    networkOkay = false,
    connectionNotice = false,
    authenticationLost = false,
    pollingDelay = 1000;
  const uuid = () => crypto.randomUUID().replaceAll('-', '');
  const requestId = () => `request_${uuid()}`;
  const showStatus = (text, state = '') => {
    if ($('status').textContent !== text) $('status').textContent = text;
    $('status').dataset.state = state;
  };
  const notice = (text, fromConnection = false) => {
    $('notice').textContent = text;
    connectionNotice = fromConnection;
  };
  const connected = (yes) => {
    networkOkay = yes;
    if (yes && connectionNotice) notice('');
    const label = yes ? '接続済み' : '接続を確認できません';
    if ($('connection').textContent !== label) $('connection').textContent = label;
    controls();
  };
  class ApiError extends Error {
    constructor(error) {
      super(error.message || error.code);
      this.detail = error;
    }
  }
  async function api(path = '', body) {
    const response = await fetch(base + path, {
      method: body === undefined ? 'GET' : 'POST',
      credentials: 'omit',
      cache: 'no-store',
      headers: {
        'X-Explain-Token': token,
        ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    const value = await response.json();
    if (!response.ok)
      throw new ApiError(
        value.error || { code: 'invalid_response', message: '応答を確認できません。' },
      );
    connected(true);
    return value;
  }
  function controls() {
    const preview = displayed && current && displayed.revision_id !== current.revision_id;
    $('reconnect').hidden = networkOkay && pending?.kind !== 'undo';
    $('send').disabled =
      !displayed ||
      Boolean(staging || preview || pending || $('refresh').dataset.conflict) ||
      !networkOkay ||
      selected?.kind === 'missing';
    $('cancel').disabled = !pending || pending.kind !== 'edit';
    $('undo').disabled =
      !displayed?.parent_revision || Boolean(pending || staging || preview) || !networkOkay;
    $('export').disabled = !displayed || Boolean(staging);
    $('preview-note').hidden = !preview;
    $('refresh').hidden = !preview && !$('refresh').dataset.conflict;
    for (const button of $('history').querySelectorAll('button'))
      button.disabled = Boolean(pending || staging) || !networkOkay;
  }
  function indexRevision(revision) {
    if (
      !revision ||
      !revisionPattern.test(revision.revision_id) ||
      revision.document.document_id !== documentId
    )
      throw new Error('Invalid revision');
    const template = document.createElement('template');
    template.innerHTML = revision.document.html;
    return new Map(
      [...template.content.querySelectorAll('[data-content-id]')].map((el) => {
        const id = el.dataset.contentId,
          figures = revision.document.figures.filter((figure) => figure.content_id === id);
        const assets = revision.document.assets.filter(
          (asset) =>
            figures.some((figure) => figure.asset_ids.includes(asset.asset_id)) ||
            el.outerHTML.includes(asset.path),
        );
        const heading = el.querySelector('h1,h2,h3,h4,h5,h6,summary,figcaption');
        return [
          id,
          {
            text: el.textContent,
            label: (heading?.textContent || id).trim(),
            signature: JSON.stringify([el.outerHTML, figures, assets]),
          },
        ];
      }),
    );
  }
  function targetOptions() {
    const old = selected,
      options = [new Option('ページ全体', 'document')];
    for (const [id, block] of blocks) options.push(new Option(block.label, `content:${id}`));
    if (old?.kind === 'selection')
      options.push(
        new Option(
          `選択した文章 · ${blocks.get(old.content_id)?.label || old.content_id}`,
          'selection',
        ),
      );
    if (old?.kind === 'missing') options.push(new Option('対象を選び直してください', 'missing'));
    $('target').replaceChildren(...options);
    $('target').value =
      old?.kind === 'content' ? `content:${old.content_id}` : old?.kind || 'document';
    $('selection').textContent =
      old?.kind === 'selection'
        ? `「${old.exact}」`
        : old?.kind === 'content'
          ? `内容ID: ${old.content_id}`
          : old?.kind === 'missing'
            ? old.reason || '節の削除・変更により以前の対象を使えません。対象を選び直してください。'
            : '本文の見出し・図を選ぶか、文章の一部を選択できます。';
  }
  function post(frame, message) {
    frame.node.contentWindow.postMessage(
      {
        channel,
        document_id: documentId,
        revision: frame.revision.revision_id,
        view: frame.view,
        ...message,
      },
      '*',
    );
  }
  function showRevision(revision) {
    if (displayed?.revision_id === revision.revision_id) {
      controls();
      return Promise.resolve();
    }
    const nextBlocks = indexRevision(revision),
      unchanged = [...nextBlocks]
        .filter(([id, block]) => blocks.get(id)?.signature === block.signature)
        .map(([id]) => id);
    if (staging) {
      staging.node.remove();
      clearTimeout(staging.timer);
      staging.reject(new Error('Superseded view'));
    }
    const node = document.createElement('iframe');
    node.title = '説明文書。見出しや図を選ぶと指示の対象にできます';
    node.setAttribute('sandbox', 'allow-scripts');
    node.referrerPolicy = 'no-referrer';
    node.className = 'staging';
    node.src = `${base}/revisions/${revision.revision_id}/content/reader.html`;
    const completion = new Promise((resolve, reject) => {
      staging = { node, revision, view: uuid(), blocks: nextBlocks, unchanged, resolve, reject };
      staging.timer = setTimeout(() => {
        if (staging?.node !== node) return;
        staging = null;
        node.remove();
        controls();
        reject(new Error('文書を表示できませんでした。状態を再確認してください。'));
      }, 15000);
    });
    $('reading').append(node);
    controls();
    return completion;
  }
  function validTarget(target) {
    if (!target || !blocks.has(target.content_id)) return false;
    if (target.kind === 'content') return Object.keys(target).sort().join() === 'content_id,kind';
    if (
      target.kind !== 'selection' ||
      Object.keys(target).sort().join() !== 'content_id,end,exact,kind,prefix,start,suffix'
    )
      return false;
    const text = blocks.get(target.content_id).text;
    return (
      Number.isSafeInteger(target.start) &&
      Number.isSafeInteger(target.end) &&
      target.start >= 0 &&
      target.end > target.start &&
      typeof target.exact === 'string' &&
      target.exact.trim() &&
      typeof target.prefix === 'string' &&
      typeof target.suffix === 'string' &&
      text.slice(target.start, target.end) === target.exact &&
      text.slice(0, target.start).endsWith(target.prefix) &&
      text.slice(target.end).startsWith(target.suffix)
    );
  }
  function validState(state, known) {
    return (
      state &&
      Number.isFinite(state.x) &&
      Number.isFinite(state.y) &&
      state.x >= 0 &&
      state.y >= 0 &&
      (!state.anchor ||
        (known.has(state.anchor.content_id) && Number.isFinite(state.anchor.offset))) &&
      Array.isArray(state.blocks) &&
      state.blocks.length === known.size &&
      new Set(state.blocks.map((block) => block.content_id)).size === known.size &&
      state.blocks.every(
        (block) =>
          known.has(block.content_id) &&
          Array.isArray(block.details) &&
          block.details.every((value) => typeof value === 'boolean') &&
          Array.isArray(block.inputs) &&
          block.inputs.every(
            (input) => typeof input.value === 'string' && typeof input.checked === 'boolean',
          ),
      )
    );
  }
  addEventListener('message', (event) => {
    const message = event.data;
    if (
      event.origin !== 'null' ||
      !message ||
      message.channel !== channel ||
      message.document_id !== documentId
    )
      return;
    if (
      staging &&
      event.source === staging.node.contentWindow &&
      message.revision === staging.revision.revision_id
    ) {
      if (message.type === 'ready')
        post(staging, {
          type: 'connect',
          state: readingState,
          unchanged: staging.unchanged,
          changes: staging.revision.changes
            .filter((change) => change.kind !== 'removed')
            .map((change) => change.content_id),
        });
      if (
        message.type === 'state' &&
        message.view === staging.view &&
        message.restored === true &&
        validState(message.state, staging.blocks)
      ) {
        const next = staging;
        clearTimeout(next.timer);
        if (
          (selected?.kind === 'content' && !next.blocks.has(selected.content_id)) ||
          (selected?.kind === 'selection' && !next.unchanged.includes(selected.content_id))
        )
          selected = { kind: 'missing' };
        activeFrame?.node.remove();
        $('document')?.remove();
        next.node.id = 'document';
        next.node.className = '';
        activeFrame = next;
        displayed = next.revision;
        blocks = next.blocks;
        readingState = message.state;
        staging = null;
        $('empty').hidden = true;
        $('revision').textContent = `表示版 ${displayed.revision_id.slice(0, 8)}`;
        $('revision').dataset.revision = displayed.revision_id;
        if (message.missing_anchor)
          notice(
            '読んでいた節が削除・再構成されました。元の高さ付近を表示しています。変更箇所へはボタンで移動できます。',
          );
        targetOptions();
        changesPanel();
        controls();
        next.resolve();
      }
      return;
    }
    if (
      !activeFrame ||
      event.source !== activeFrame.node.contentWindow ||
      message.revision !== displayed.revision_id ||
      message.view !== activeFrame.view
    )
      return;
    if (message.type === 'selection' && validTarget(message.target)) {
      selected = message.target;
      targetOptions();
      controls();
    } else if (
      message.type === 'selection' &&
      !message.target &&
      ['multiple_blocks', 'dynamic_selection'].includes(message.error)
    ) {
      selected = {
        kind: 'missing',
        reason:
          '同じ節の文章を選択してください。操作で変わるグラフの値は、図そのものを対象に選べます。',
      };
      targetOptions();
      controls();
    } else if (message.type === 'state' && validState(message.state, blocks))
      readingState = message.state;
  });
  function changesPanel() {
    const changes = displayed?.changes || [];
    $('change-panel').hidden = !changes.length;
    $('summary').textContent = displayed?.change_summary || '';
    $('changes').replaceChildren(
      ...changes.map((change) => {
        if (change.kind === 'removed' || !blocks.has(change.content_id)) {
          const label = document.createElement('span');
          label.textContent = `${change.content_id}（削除済み）`;
          return label;
        }
        const button = document.createElement('button');
        button.textContent = `${blocks.get(change.content_id).label}へ移動`;
        button.dataset.contentId = change.content_id;
        button.addEventListener('click', () =>
          post(activeFrame, { type: 'navigate', content_id: change.content_id }),
        );
        return button;
      }),
    );
  }
  async function loadHistory() {
    historyRows = (await api('/revisions')).revisions;
    $('history').replaceChildren(
      ...historyRows.map((row) => {
        const li = document.createElement('li');
        li.dataset.revision = row.revision_id;
        const label = document.createElement('div');
        label.textContent = `${row.revision_id.slice(0, 8)} · ${row.change_summary}`;
        const view = document.createElement('button');
        view.textContent = '読む';
        view.addEventListener('click', () =>
          run(async () => {
            await showRevision((await api(`/revisions/${row.revision_id}`)).revision);
          }),
        );
        const restore = document.createElement('button');
        restore.textContent = 'この版を復元';
        restore.addEventListener('click', () => restoreRevision(row.revision_id));
        li.append(label, view, restore);
        return li;
      }),
    );
    controls();
  }
  function savePending(value) {
    if (value) sessionStorage.setItem(sessionKey, JSON.stringify(value));
    else sessionStorage.removeItem(sessionKey);
    pending = value;
    controls();
    $('elapsed').textContent = '';
  }
  function failure(error) {
    if (error instanceof ApiError) {
      notice(`${error.detail.code}: ${error.detail.message}`);
      if (error.detail.code === 'unauthorized') {
        authenticationLost = true;
        connected(false);
        showStatus(
          'セッションが変わりました。サービスの起動URLを開き直してください。',
          'disconnected',
        );
      }
      if (error.detail.code === 'stale_revision') {
        $('refresh').dataset.conflict = 'true';
        $('refresh').hidden = false;
        notice(
          '表示版が古いため反映できませんでした。現在版を確認してから、指示を明示的に送信してください。',
        );
        controls();
      }
    } else {
      connected(false);
      notice(
        '接続または応答を確認できません。本文は読み続けられます。要求IDで状態を照合し、指示を自動再送しません。',
        true,
      );
    }
  }
  async function run(operation) {
    try {
      await operation();
    } catch (error) {
      failure(error);
    }
  }
  function schedulePoll(delay = pollingDelay) {
    clearTimeout(pollTimer);
    if (pending && !authenticationLost) pollTimer = setTimeout(reconcile, delay);
  }
  async function handleJob(job) {
    if (!pending || job.request_id !== pending.request_id || pending.applying) return;
    if (job.document_id !== documentId || job.base_revision !== pending.base_revision)
      throw new Error('Invalid request response');
    const labels = {
      queued: '受付済み',
      running: '説明を編集中',
      validating: '内容と表示を検証中',
    };
    if (labels[job.state]) {
      showStatus(
        `${labels[job.state]}。数分かかることがあります。このまま読み続けられます。処理は中止できます。`,
        job.state,
      );
      schedulePoll();
      return;
    }
    if (job.state === 'succeeded' && revisionPattern.test(job.revision_id)) {
      const operation = pending;
      operation.applying = true;
      try {
        current = (await api()).revision;
        await showRevision(current);
        await loadHistory();
        savePending(null);
      } catch (error) {
        operation.applying = false;
        throw error;
      }
      showStatus(
        current.revision_id === job.revision_id
          ? '検証済みの変更を本文へ反映しました。'
          : '要求は完了しています。さらに更新された現在版を表示しています。',
        'succeeded',
      );
      $('retry').hidden = true;
    } else if (['failed', 'cancelled', 'interrupted'].includes(job.state) && job.error) {
      savePending(null);
      showStatus(
        {
          failed: '反映できませんでした。表示版を保持しています。',
          cancelled: '処理を中止しました。表示版を保持しています。',
          interrupted: '処理が中断されました。指示は再送していません。',
        }[job.state],
        job.state,
      );
      failure(new ApiError(job.error));
      $('retry').hidden = !lastAttempt || !job.error.retryable;
    } else throw new Error('Invalid job state');
  }
  async function reconcile() {
    if (!pending || pollBusy) return;
    clearTimeout(pollTimer);
    pollBusy = true;
    try {
      if (pending.kind === 'edit') await handleJob(await api(`/requests/${pending.request_id}`));
      else {
        await loadHistory();
        const published = historyRows.find((row) => row.request_id === pending.request_id);
        if (published) {
          notice('');
          current = (await api()).revision;
          await showRevision(current);
          savePending(null);
          showStatus('復元済みの版を確認しました。', 'succeeded');
        } else {
          notice(
            '復元の結果はまだ確定できません。要求IDを保持して公開履歴を照合しています。自動再送は行いません。状態を再確認できます。',
          );
          showStatus('復元の確定を照合中です。表示版はこのまま読めます。', 'reconciling');
        }
      }
      pollingDelay = 1000;
    } catch (error) {
      failure(error);
      if (error instanceof ApiError && error.detail.code === 'request_not_found') {
        showStatus('この要求IDは登録されていません。指示は自動再送していません。', 'not_found');
        savePending(null);
        $('retry').hidden = !lastAttempt;
      } else if (!(error instanceof ApiError) || error.detail.retryable)
        pollingDelay = Math.min(pollingDelay * 2, 15000);
    } finally {
      pollBusy = false;
      schedulePoll();
    }
  }
  async function submit(instruction, target) {
    if ($('send').disabled) return;
    const body = {
      document_id: documentId,
      request_id: requestId(),
      base_revision: displayed.revision_id,
      target,
      instruction,
    };
    lastAttempt = { instruction, target };
    notice('');
    $('retry').hidden = true;
    savePending({
      kind: 'edit',
      request_id: body.request_id,
      base_revision: body.base_revision,
      started: Date.now(),
    });
    showStatus('指示を送信しています。本文はこのまま読めます。', 'submitting');
    try {
      await handleJob(await api('/requests', body));
    } catch (error) {
      failure(error);
      if (error instanceof ApiError) {
        savePending(null);
        showStatus('指示は受け付けられませんでした。', 'rejected');
      } else schedulePoll();
    }
  }
  async function restoreRevision(revisionId) {
    if (!current || pending || staging || !networkOkay) return;
    const body = {
      document_id: documentId,
      request_id: requestId(),
      base_revision: current.revision_id,
      revision_id: revisionId,
    };
    notice('');
    savePending({
      kind: 'undo',
      request_id: body.request_id,
      base_revision: body.base_revision,
      started: Date.now(),
    });
    showStatus('過去版を新しい版として復元しています。', 'restoring');
    try {
      current = (await api('/undo', body)).revision;
      await showRevision(current);
      await loadHistory();
      savePending(null);
      showStatus('過去版を新しい版として復元しました。ここから編集を続けられます。', 'succeeded');
    } catch (error) {
      failure(error);
      if (error instanceof ApiError) savePending(null);
      else schedulePoll();
    }
  }
  $('edit-form').addEventListener('submit', (event) => {
    event.preventDefault();
    if ($('instruction').value.trim()) run(() => submit($('instruction').value, selected));
  });
  $('target').addEventListener('change', () => {
    const value = $('target').value;
    if (value === 'document') selected = { kind: 'document' };
    else if (value.startsWith('content:'))
      selected = { kind: 'content', content_id: value.slice(8) };
    targetOptions();
    controls();
  });
  $('cancel').addEventListener('click', () =>
    run(async () => {
      if (!pending || pending.kind !== 'edit') return;
      try {
        await handleJob(
          await api(`/requests/${pending.request_id}/cancel`, {
            document_id: documentId,
            request_id: pending.request_id,
          }),
        );
      } catch (error) {
        failure(error);
        schedulePoll();
      }
    }),
  );
  $('retry').addEventListener('click', () => {
    if (lastAttempt) run(() => submit($('instruction').value || lastAttempt.instruction, selected));
  });
  $('reconnect').addEventListener('click', () =>
    run(async () => {
      if (pending) await reconcile();
      else {
        current = (await api()).revision;
        await showRevision(current);
        await loadHistory();
      }
    }),
  );
  $('refresh').addEventListener('click', () =>
    run(async () => {
      current = (await api()).revision;
      await showRevision(current);
      await loadHistory();
      delete $('refresh').dataset.conflict;
      notice('現在版を表示しています。内容と対象を確認してから指示を送信してください。');
      controls();
    }),
  );
  $('undo').addEventListener('click', () => restoreRevision(displayed.parent_revision));
  addEventListener('online', () => {
    if (pending) schedulePoll(0);
  });
  addEventListener('offline', () => {
    connected(false);
    notice('接続が切れました。本文は読み続けられます。復旧後は要求IDで照合します。', true);
  });
  setInterval(() => {
    if (pending) {
      const seconds = Math.max(0, Math.floor((Date.now() - pending.started) / 1000));
      $('elapsed').textContent =
        `経過 ${Math.floor(seconds / 60)}分${seconds % 60}秒 · 要求 ${pending.request_id.slice(-8)}`;
    }
  }, 1000);
  function zip(files) {
    const encoder = new TextEncoder(),
      chunks = [],
      directory = [];
    let offset = 0,
      directorySize = 0;
    const crc32 = (bytes) => {
      let crc = -1;
      for (const byte of bytes) {
        crc ^= byte;
        for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
      }
      return (crc ^ -1) >>> 0;
    };
    for (const [name, bytes] of files) {
      const encoded = encoder.encode(name),
        crc = crc32(bytes),
        local = new Uint8Array(30 + encoded.length),
        l = new DataView(local.buffer);
      l.setUint32(0, 0x04034b50, true);
      l.setUint16(4, 20, true);
      l.setUint16(6, 0x800, true);
      l.setUint32(14, crc, true);
      l.setUint32(18, bytes.length, true);
      l.setUint32(22, bytes.length, true);
      l.setUint16(26, encoded.length, true);
      local.set(encoded, 30);
      l.setUint16(12, 33, true); // 1980-01-01, supported by offline ZIP readers.
      const central = new Uint8Array(46 + encoded.length),
        c = new DataView(central.buffer);
      c.setUint32(0, 0x02014b50, true);
      c.setUint16(4, 20, true);
      c.setUint16(6, 20, true);
      c.setUint16(8, 0x800, true);
      c.setUint32(16, crc, true);
      c.setUint32(20, bytes.length, true);
      c.setUint32(24, bytes.length, true);
      c.setUint16(28, encoded.length, true);
      c.setUint32(42, offset, true);
      central.set(encoded, 46);
      c.setUint16(14, 33, true);
      chunks.push(local, bytes);
      directory.push(central);
      offset += local.length + bytes.length;
      directorySize += central.length;
    }
    const end = new Uint8Array(22),
      e = new DataView(end.buffer);
    e.setUint32(0, 0x06054b50, true);
    e.setUint16(8, files.length, true);
    e.setUint16(10, files.length, true);
    e.setUint32(12, directorySize, true);
    e.setUint32(16, offset, true);
    return new Blob([...chunks, ...directory, end], { type: 'application/zip' });
  }
  $('export').addEventListener('click', () =>
    run(async () => {
      const revision = displayed.revision_id;
      $('export-status').textContent = '静的な説明一式を準備しています。';
      const snapshot = await api('/exports', { document_id: documentId, revision_id: revision });
      if (!revisionPattern.test(snapshot.snapshot_id) || snapshot.revision_id !== revision)
        throw new Error('Invalid export');
      const files = [];
      for (const name of snapshot.files) {
        if (
          !/^(index\.html|evidence\.json|figures\.json|manifest\.json|assets\/[A-Za-z][A-Za-z0-9_-]{0,79}\.[A-Za-z0-9]+)$/.test(
            name,
          )
        )
          throw new Error('Invalid export path');
        const response = await fetch(`/api/exports/${snapshot.snapshot_id}/${name}`, {
          credentials: 'omit',
          cache: 'no-store',
        });
        if (!response.ok) throw new Error('Export download failed');
        files.push([name, new Uint8Array(await response.arrayBuffer())]);
      }
      const url = URL.createObjectURL(zip(files)),
        link = document.createElement('a');
      link.href = url;
      link.download = `${documentId}-${revision.slice(0, 8)}.zip`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      $('export-status').textContent =
        '保存しました。ZIPを展開してindex.htmlを開けます。操作UIを含まない静的版です。';
    }),
  );
  run(async () => {
    if (
      !idPattern.test(documentId || '') ||
      !token ||
      token.length < 32 ||
      fragment.getAll('token').length !== 1 ||
      fragment.getAll('document_id').length !== 1
    ) {
      showStatus(
        'サービスが表示した起動URLを開いてください。再起動後は新しいURLが必要です。',
        'disconnected',
      );
      return;
    }
    const saved = sessionStorage.getItem(sessionKey);
    if (saved) {
      const record = JSON.parse(saved);
      if (
        !['edit', 'undo'].includes(record.kind) ||
        !idPattern.test(record.request_id) ||
        !revisionPattern.test(record.base_revision) ||
        !Number.isFinite(record.started)
      )
        throw new Error('Invalid saved request');
      pending = record;
    }
    current = (await api()).revision;
    await showRevision(current);
    await loadHistory();
    if (pending) {
      showStatus('前の要求IDで状態を照合しています。再送は行いません。', 'reconciling');
      await reconcile();
    } else showStatus('本文を読みながら、説明を追加できます。', 'ready');
  });
})();
