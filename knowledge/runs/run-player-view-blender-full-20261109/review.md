# C3 人間確認用の動画と設定

**人間の定性受入は未了。** 対象は以下の保存済みclipの最終動画。

入力: `clip000_20261103`、人物ID `0` / `1`、元frame `[0,1010)`、FPS `59.94006`。
2人分とも同じ最終renderer・GPU denoising設定で新規描画し、旧388frameは採用していない。

|動画|視点|frame数|解像度|FPS|長さ（秒）|
|---|---|---:|---|---:|---:|
|[動画](videos/player_0.mp4) `player_0.mp4`|ID0|1010|1280×720|59.94006000|16.850167|
|[動画](videos/player_1.mp4) `player_1.mp4`|ID1|1010|1280×720|59.94006000|16.850167|
|[動画](videos/comparison.mp4) `comparison.mp4`|左右同期 ID0 / ID1|1010|2560×720|59.94006000|16.850167|
|[動画](videos/comparison_short.mp4) `comparison_short.mp4`|同左、元frame [0,240)|240|2560×720|59.94006000|4.004000|

比較動画の各画面には人物ID・元frame番号・cameraの有効/欠測を表示。

仮設定: 1280×720、水平画角90度、球半径0.05m、本人mesh全体非表示。
cameraはC2の保存列を読み、頭部姿勢のroll/pitchも保持する。注視・安定化・欠測補間は追加していない。
相手は灰色SMPL、球は緑の通常Principled material、alpha1・emission0。背景は簡素なコートとネット。
16 Cycles samples、Blender5.2.1 LTS、NVIDIA L4/OPTIX、OpenImageDenoiseのGPU使用を明示。
cameraの保存設定: `{"width": 1280, "height": 720, "horizontal_fov_degrees": 90.0, "face_offset_head_m": [0.0, 0.08, 0.12], "near_m": 0.05, "far_m": 100.0}`。

|検証|ID0|ID1|
|---|---:|---:|
|監査frame数|1010|1010|
|camera有効|1007|993|
|相手mesh描画許可|992|992|
|球描画許可|854|849|
|PC2全頂点の最大差（m）|0.0|0.0|

描画許可数はmaskに基づく値で、画角や遮蔽による画面内の見え方は人間が確認します。

ID0 camera欠測元frame: `[0, 1, 731]`。

ID1 camera欠測元frame: `[0, 1, 735, 741, 742, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761]`。

全frameをdecodeし、解像度・FPS・長さ・PTSを確認済み。
入力bundle/保存cameraのdigest、全2020rowのID/source/cache index・camera/人物/球mask・姿勢・球位置・本人非表示を照合済み。
入力/設定・各動画・最終描画記録のdigestの対応は[manifest](videos/manifest.json)と[独立検証](full_validation.json)に保持。

人間の確認観点は視点の向きと追従、人物と球の見え方、元frameとの時間対応、欠測表示。
対象動画と人間本人の判断は親がissueへ対応付けて記録する。agentの目視・検証やCIは人間の受入に置き換えない。

再生成は[固定入力とコマンド](reproduction.md)を参照。
Blender実描画source ref: `993c7e2310a76ebc7145ca33b9698be5b4996ddc`。
