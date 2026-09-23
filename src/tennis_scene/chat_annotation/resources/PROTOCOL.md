# テニス動画の注釈（{{KIT_VERSION}}）

## 対象と全フレームの確認

主に撮影しているコートの試合・練習に参加するプレーヤーと、そこでプレー中のボールを注釈してください。
サーブトスを含み、人数・球数は固定しません。非プレーヤー、隣接コートの選手・球、予備球・球拾い、コート形状は対象外です。参加者や対象コートを判別できない場合は不明点を記録します。

動画名・解像度・総フレーム数Nは添付動画から取得してください。
先頭から末尾まで、前後の参考区間も含めた全Nフレームを省略せず確認してください。
framesには表示順の0〜N−1を各1件、昇順で記録します。代表フレームだけの確認や、未確認区間へのコピー・補間で代替しないでください。
各フレームの両対象を確認した場合だけreviewed=trueとします。対象不在を確認した場合だけ空配列を使います。
未確認フレームもreviewed=falseで残し、notesへ理由を記録してください。位置が不明な場合と未確認を区別し、座標を捏造しません。

## 注釈の定義

座標は元画像のpixel xy（左上原点、右+x、下+y）。bboxは[x_min,y_min,x_max,y_max]。
プレーヤーは隠れた身体を含む全身bboxで、ラケット・影を除外します。
bbox_sourceはobserved / inferred / unresolved。遮蔽はoccluded、画面切れはtruncatedをtrueにし、どちらも全身bboxはinferredです。
画像外に延びるbboxにはtruncated=trueが必要です。全身を定位できなければbbox_xyxy=null、bbox_source=unresolvedとします。

ボールのcenter_pxは中心座標です。statusはvisible（直接観測）、occluded（遮蔽）、interpolated（短区間の内挿）、out_of_frame（画面外）、unresolved（不明）。
画面外・不明は座標をnullにします。遮蔽は根拠がある場合のみ推定座標を使い、定位不能ならnullです。座標がある球は画像内に収めます。
内挿は、全区間を確認した同一球の両端visible間だけに限り、両端のフレーム番号をinterpolation_frames=[始点,終点]に記録します。それ以外はnullです。
両端の実時刻差は{{BALL_MAX_GAP_SECONDS}}秒以下とし、実時刻による線形内挿を使います。
打球・バウンド・カット・プレー開始/終了があるフレームはinterpolation_break=trueとし、そのフレームを含む区間の内挿は禁止します。
観測値の上書き、画面外への補間、外挿は禁止です。可変FPSを固定FPSとして扱いません。

同じ対象のtrack_idを維持し、同一フレームの同種対象で重複させません。カット後に同一性が不明なら新しいIDを使います。
不明点はそのフレームのnotes、全体の問題はissuesに記録します。問題がなければnotes=""、issues=[]です。
全フレームを確認して未解決事項もない場合だけstatus=completed。それ以外はpartialとし、理由を残します。
部分完了も同じ2ファイルを返してください。入力不一致や実行不能は理由を説明し、存在しない成果物を提示しません。

## 注釈JSON例

以下は1フレームの例です。実際には入力の全Nフレームを記録します。clip_idは添付MP4のファイル名から拡張子だけを除いた名前です。
例の全項目を必須とし、width・height・frame_countは入力と一致させます。未知の座標はnullとし、NaN、Infinity、重複キーは使いません。

```json
{
  "schema_version": "tennis_chat_annotation.v2",
  "clip_id": "source__run__clip",
  "width": 1920,
  "height": 1080,
  "frame_count": 1,
  "status": "completed",
  "issues": [],
  "frames": [{
    "frame_index": 0,
    "reviewed": true,
    "players": [{
      "track_id": "p1",
      "bbox_xyxy": [100, 200, 180, 500],
      "bbox_source": "observed",
      "occluded": false,
      "truncated": false
    }],
    "balls": [{
      "track_id": "b1",
      "center_px": [900, 400],
      "status": "visible",
      "interpolation_frames": null
    }],
    "interpolation_break": false,
    "notes": ""
  }]
}
```

## 成果物

ZIP直下にはoverlay_{clip_id}.mp4とannotation_{clip_id}.jsonの2ファイルだけを入れます。
overlayは全Nフレーム・元の画角・順序・表示時間を保持し、JSONのbbox・球中心・IDを対応するフレームへ描きます。
観測と推定・補間を見分けられるようにし、フレーム番号、凡例、未確認・未解決状態を表示します。null座標は描画しません。
返却前に全フレームの欠落・重複、座標、入力との対応、JSONとoverlayの一致を確認してください。
ZIPのダウンロードリンクを提示します。応答文の形式・行数、作業手順、実装方法は自由です。
