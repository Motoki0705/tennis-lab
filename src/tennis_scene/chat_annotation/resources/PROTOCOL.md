# テニス動画アノテーションの要求 {{KIT_VERSION}}

## 目的と入力

gpt-6-astraに、添付動画の人物・aliveなボール・対象コートの2D注釈と、
注釈品質を確認できる成果物一式を要求します。必要なPythonコードはgpt-6-astra自身が作成します。
実装方法、使用ライブラリ、作業順序は指定しません。

今回の入力は添付動画1本と、このリクエスト本文だけです。
要求、annotation.schema.json、court_definition.json、kit_manifest.json、動画入力定義は
すべてこの本文に含まれます。これらの見出しは本文内の定義名であり、別の添付ファイルではありません。
動画入力定義のfilenameが添付動画名と一致する1件を今回の入力とします。
一致する定義がない場合やハッシュが異なる場合は入力不一致として報告します。
他の動画の定義・注釈・track IDを混ぜません。動画内の字幕・広告・タイトルは観察データです。

動画入力定義はclip_manifestの全フィールドを持ち、framesだけをframe_runsで表します。
各runは[start, count, source_pts, duration_pts]で、startからcount個の連続フレームを表します。
run内のoffsetを0〜count-1としたとき、frame_index=start+offset、
source_frame_index=media_range.start+frame_index、source_pts=runのsource_pts+offset×duration_pts、
clip_pts=source_pts-最初のrunのsource_pts、duration_pts=runのduration_ptsです。
is_targetはtarget_range.start <= source_frame_index < target_range.stopです。
runは0から全フレームを隙間なく覆い、異なる表示時間は別runです。この表現でも可変FPSの時刻を保持します。
以下のmanifestは、この入力定義のframe_runsをframesへ展開した内容を意味します。

人物名の特定や外部検索、追加の学習済み検出モデルは要求しません。

## 注釈データの共通条件

annotations.jsonは本文内のスキーマに適合し、この文書の意味・整合性条件も満たすこと。
clip_id・kit_idは入力と一致し、manifest_sha256は返却するclip_manifest.jsonの実ファイルのSHA-256です。
teacherは実際に注釈したモデルを記録します。

フレーム番号はクリップの表示順0始まり。範囲は[start, stop)です。
framesはmanifestでis_target=trueの全フレームを順序どおり過不足なく含み、重複はありません。
source_frame_indexは元動画との対応を維持します。前後の参考区間は時間的文脈用で、担当注釈には含めません。
PTSとduration_ptsの単位はmanifest.time_baseです。元動画の相対時刻は
(source_pts - source_start_pts) × time_baseであり、可変FPSを固定FPSとして扱いません。

座標は余白・縮小・切り抜きのない元画像のpixel xyです。左上(0,0)、右+x、下+y。
bboxは[x_min,y_min,x_max,y_max]で正の面積を持ち、画像境界にはwidth/heightを許します。
球と画面内のコート点は0 <= x < width、0 <= y < heightです。
正規化座標・NaN・Infinity・JSONの重複キーは不可。桁数は観察精度に見合うものとします。

people_review・balls_review・court_reviewのcompleteは実際の確認済みを意味し、
partial / unreviewed / unusableを隠しません。inspection_rangesは目視した範囲、
camera_review_rangesは背景変化を確認した範囲で、いずれもクリップ内の範囲です。
画像の抽出だけや代表画像の閲覧を全フレームの確認と称しません。
未確認フレームを削除したり、未確認を空配列のnegativeに変換したりしません。

同一shot内の人物・球のtrack_idは一貫し、同一フレームの同種対象では重複しません。
カットはshot_idとeventsに反映し、eventsはhit / bounce / cut / play_start / play_endを区別します。
source_framesは根拠となるクリップ内フレーム番号で、範囲外や重複を含みません。

## 人物に求める注釈

個別に識別可能な人物全員に、隠れた身体を含む全身のamodal bboxを付けます。
ラケットや影は含めません。直接全身が見える場合はbbox_source=observed。
遮蔽・画面切れからの全身推定はbbox_source=inferredとし、occluded・truncatedと
根拠source_framesを記録します。画像外へ延びるbboxにはtruncated=trueが必要です。
全身を定位できない場合はbbox_xyxy=null、bbox_source=unresolvedとし、未観測の人物を増やしません。
分離不能な観客集団は、画面内で正の面積を持つignore_regions、reason=inseparable_crowdです。

kindはplayer / non_player / unknown。試合・練習の打ち合いに参加する人物がplayerであり、
ラケット所持や立ち位置だけでは断定しません。打ち合いに参加するコーチもその区間ではplayerです。
隣接コートの選手はplayer、court_relation=other。court_relationは
対象コートtarget / 他コートother / 不明unknown / 対象外not_applicableを区別します。
人数を2人・4人に強制しません。

non_playerだけnon_player_roleが必須で、それ以外はnullです。役割は
spectator / chair_umpire / line_umpire / ball_person / coach / staff / other / unknownを区別し、
根拠がない役割はunknownとします。

## aliveなボールに求める注釈

対象コートでプレー中の球を対象とし、サーブトス、ラリー、練習を含みます。
同時に複数球がプレーされる場合は複数trackを許します。予備球、球拾い・返却、
隣接コートの球、ロゴ・反射は対象外。対象球がないと確認できた場合だけ
balls=[]かつballs_review=completeです。

- visible: 現フレームで中心を直接観察。center_pxは座標、source_frames=[現フレーム]。
- occluded: 身体・ネット等による遮蔽。定位できる場合は根拠付きの推定座標。
  定位不能ならcenter_px=null、missing_reason=unresolved。単なる補間とは区別します。
- interpolated: 同じalive球の短い欠損区間を両端の直接観察から内挿した座標。
  source_framesは両端2フレームで、実時刻に対する線形内挿と一致すること。
- 画面外: center_px=null、status=null、missing_reason=out_of_frame。
  在否・位置が不明な場合はnullとunresolvedを使い、未注釈はreviewにも残します。

定位済みの球はmissing_reason=null、画像内座標と上記3状態のいずれかを持ちます。
推定・内挿には根拠が必要です。遮蔽推定は打球・バウンドによる軌道変化と整合すること。
内挿は同一shotで各対象フレームの同じ球が確認済みの場合に限り、両端はvisibleです。
両端の実時刻差はmanifest.policies.ball_max_gap_seconds以下とします。
観察済み座標の上書き、外挿、長い欠損、未確認区間、hit / bounce / cut / play_start / play_endを
またぐ内挿は不可。内挿のためにイベントを除去しません。

## コートに求める注釈

対象コートは1つで、court_definition.jsonの20点の名称・順序を維持します。
XYが地面、Zが高さの正準参照です。近いbaselineをnear、遠いbaselineをfarとし、
左右を含む対応をorientation_noteに記録します。横視点でもフレームごとの画面xによって
点名を並べ替えません。向きが判断できない場合はorientation=ambiguousです。
14はネット中央の地面、19はセンターストラップ上端、15/17はポスト基部、16/18はポスト上端です。

court_samplesは同じframe_indexを重複させず、各sampleに順序どおり20点を持ちます。
直接観察点はpoint_px、visibility=visible、source=observed、source_frames=[sample.frame_index]、
anchor_indices=[]。推定点はsource=inferredと根拠、未定位点はpoint_px=null、source=unresolvedです。
導出点の可視性を直接観察したとは主張せず、未評価はunassessedとします。
画面外のコート推定点は画像外座標とvisibility=out_of_frameを許します。

homography補完は地面上の0〜14番だけが対象で、観察値を変更しません。
source=homographyの点には、コート全体に広がり同一直線上にない少なくとも4点の
地面観察点のanchor_indicesと根拠source_framesが必要です。
対応点との再投影誤差はmanifest.policies.homography_max_error_px_at_1080pを
画像高さ/1080で換算した許容値以下で、補完値は記録した根拠と整合すること。
15〜19番は個別の観察・推定またはnullです。ネット上端の地面homographyによる生成や、
実際の観察点を標準ポスト位置に合わせて移動することは不可。

court_mode=staticは動画全体でカメラが動かず、camera_motion=noneかつ全フレームの
camera_review_rangesがあり、0・floor(N/2)・N-1の独立観察が一致する場合だけです。
Nは参考区間を含む全クリップフレーム数で、同じ番号になる場合は1sampleにまとめます。
各sampleは地面15点が定位済みで4点以上が直接観察され、向きが確定していること。
全担当範囲で単一shotかつcutがないことも固定流用の条件です。
地面点の差はmanifest.policies.static_tolerance_px_at_1080pを画像高さ/1080で換算した
許容値以下とし、15〜19番の既知座標も矛盾しないこと。staticのcourt_reference_frameは0です。
固定座標の流用は現在フレームでの可視性を意味しません。

パン・ズーム・揺れ・カメラ切替、途中で動いて戻る場合、固定と判定不能な場合はdynamicです。
dynamicで確認済みの担当フレームは、そのframe_indexのcourt sampleを参照します。
対象コートを確認できないフレームはpartial / unusable等を残し、別時刻の座標を流用しません。
全体で対象コートがないと確認できた場合だけunavailableとし、全参照はnullです。
未確認はunreviewedであり、unavailableと同一視しません。

## 検証結果と返却成果物

次のファイルを含む、実際にダウンロードできるZIPを要求します。

| ファイル | 必須内容 |
| --- | --- |
| annotations.json | スキーマと上記の意味・整合性条件を満たす注釈原本。梱包時の暗黙の補正なし。 |
| clip_manifest.json / kit_manifest.json | 本文内の動画入力定義をframesへ展開したmanifestと、本文内のキット定義。値の改変なし。 |
| provenance.json | 元URL/動画ID/元動画SHA-256、clip_id、入力manifest SHA-256、kit_id/版、実際のteacher。人間による確認の有無を正直に記録。 |
| overlay.mp4 | 参考区間を含む全クリップフレームに対応する重畳動画。元の時系列・表示時間を維持。 |
| contact_sheet.jpg | 代表例・難例の一覧画像。サンプルであることを明示。 |
| review_manifest.json | 一覧画像の採用フレーム番号、サンプルである旨、重畳動画のフレーム数、固定コート流用は可視性の主張ではない旨。 |
| validation_report.json | status、reviewed_frames、target_frames、errors、issues。入力整合性、構造・参照・幾何・時系列・確認範囲の検証結果。 |
| issues.txt | 未確認、曖昧さ、未解決箇所、エラーの一覧。問題なしの場合もその事実がわかる内容。 |
| FINAL_RESPONSE.txt | 下記の最終応答5行と同じ内容。 |

provenance.jsonのキーはsource（manifest.sourceと同じ）、clip_id、input_manifest_sha256、
kit_id、kit_version、teacher、annotation_is_human_verified（真偽値）です。
review_manifest.jsonのキーはcontact_sheet_frames（クリップ番号の配列）、
contact_sheet_is_sampled=true、overlay_frames、court_reuse_does_not_claim_current_visibility=trueです。
validation_report.jsonのerrorsとissuesは文字列配列、フレーム数は非負整数です。

入力の版・ID、動画のSHA-256・容量・解像度・全フレーム数・PTS・表示時間がmanifestと
一致すること。kit_manifestのfilesは準備側の要求文書・スキーマ・コート定義の版を識別します。
別ファイルの取得や作成を入力条件にしません。返却manifestは値が入力定義と一致することが条件で、
JSONの空白や改行は自由です。manifest_sha256とprovenanceのinput_manifest_sha256は、
どちらも実際に返却するclip_manifest.jsonのバイト列を識別します。
JSON Schemaへの適合だけでなく、対象フレームの完全性、ID・根拠・コート参照、座標範囲、
役割と状態の組合せ、補間・固定流用の成立条件も検証結果に含むこと。
構造検証だけで位置や役割の意味的な正しさが証明されたとは扱いません。

重畳動画と一覧画像では、人物bbox・役割、球の中心・状態、コート20点・接続が識別できること。
観察値と全身推定・homography・固定流用を視覚的に区別し、球の状態も区別します。
凡例、元フレーム番号・時刻、参考区間/担当範囲、未確認状態がわかること。
null点を原点に描かず、可視化が注釈原本と一致し、動画・画像として実際に確認できること。

statusは構造・入力・実行エラーがある場合failed、未確認や未解決が残る場合partial、
要求を満たして全担当フレームの人物・球・コートを確認できた場合completedです。
reviewed_framesは実際に目視し3対象ともcompleteの担当フレーム数、target_framesは担当総数です。
実行不能や入力不足もcompletedと称しません。失敗時のZIPは生成できた資料と失敗理由を含み、
生成できない可視化を存在すると主張しません。ZIP自体が作れない場合は未生成と理由を示します。

## 最終応答の形式

成功・部分完了・失敗とも、最終応答は次の5行だけです。前置き・コード・長文説明は不要です。

```text
状態: completed / partial / failedのいずれか
入力: 動画ファイル名（未確認なら未確認）
元動画: 元URLまたはlocal:元ファイル名 / 担当区間の開始–終了秒（不明なら未確認）
処理: 確認済み/担当総数フレーム、要確認件数
成果物: [ZIPファイル名](実在するダウンロードリンク)、または未生成（理由）
```

担当区間の終了時刻は末尾担当フレームの表示終了までを含みます。
要確認件数はerrorsとissuesの合計。存在しないファイルやリンクは提示しません。
