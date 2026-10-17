# 保存形式・shard共有・RGB cacheの比較

同じ全入力・読出し行順・attempt数・pass数から12方式を測定する。小さなfixtureの結果から
運用方式を自動採用しない。実動画の測定も通常の資源scopeを使い、停止中の予算を迂回しない。

| 軸 | 比較する値 |
|---|---|
| 形式 | taskのnative store、同じJPEGと教師を格納したImageRecord store |
| shard | attemptごとのcopy、元の不変shardへのhardlink |
| cache | 毎passで復号、attemptごとのRGB保持、全attemptでRGB共有 |

元形式のadapterは[composition層](../../../../annotation/storage_comparison.py)が所有する。
ball frames v1/v2、player frames v1/v2、共通ImageRecordに対応する。ImageRecord入力には
そのdescriptor全文を指定する。共通測定層はlabelの意味を解釈しない。
変換形式は**比較専用**で、既存training readerにそのまま渡せる公開datasetではない。

## 固定と実行

1. `StoreSpec(adapter, root, descriptor)` を固定rootに保存しhashで参照する。
   rootは既存storeのrole相対ディレクトリ。共有するshardはadapterが列挙する。
2. `InspectStorageOperation(specification, limits)` を標準CLIの `submit` へ渡す。
   全ファイルのhash/size、shard区分、全frame数とmetadata digestを `CapturedStore` に固定する。
   未知のadapter、symlink、special file、上限超過は拒否する。
3. `ComparisonRequest(source, rows, attempts, passes, limits,
   maximum_written_bytes, converted_shard_bytes)` を保存する。sourceはcaptureの参照。
   rowsは重複しない0始まりの読出し行順、attemptsは1〜50、passesは2〜100。
   変換はsample行だけでなく全frameを含む。
4. `CompareStorageOperation(request)` を `submit` し、完了receipt内のreportを読む。
   plan/runtimeに `inspect_storage` と `compare_storage` のCPU/I/O枠とcostを明示する。
   CLIの使い方は[標準入口](../../../../annotation/README.md)を参照する。

limitsには走査entry数、source file bytes、frame数、各frameのpixel数を指定する。
file/frame上限は受け入れる素材量の上限であり、readerの最大PSS保証ではない。
PSS・割当容量・空き容量・timeoutはstage supervisorが別に管理する。
`maximum_written_bytes` は新しく書く画像・index・cache等のpayloadの保守的な予約上限。
hardlinkの共有payloadは重ねて予約しない。directory・receipt・inventoryの容量は資源scopeに
含める。npz indexとnpy headerは上界を予約するため、予約量は実測容量と一致しない。

保存先は `campaigns/<campaign>/jobs/<job>/comparison/`。sourceへのwrite、move、
unlinkは行わない。hardlink作成によるsource inodeのlink数とctimeの変化は生じる。
失敗したattemptの途中成果は保持する。新しいjob attemptから再測定し、成功済み操作の再送は
既存receiptを返す。hardlink不可能時のcopy代替、壊れたcacheの再復号代替は行わない。

## 同一性と計測の解釈

全行のkey・JPEG bytes・教師digestとstore metadataを変換前後、各attemptで照合する。
nativeの教師には全frame/instance列とclip metadataを含める。配列はdtype・shape・bytesで
保存するので、空bbox、NaNの位置、playerのconfirmed_empty、ballの教師maskも維持する。
sample行では復号RGBのshape・dtype・画素digestを全passで照合する。cache keyはsource
capture、row identity、JPEG/教師/画素digest、OpenCV/NumPy版とRGB契約を結び付ける。

reportは変換時間、attemptごとのcopy/link時間・reader構築時間、各passのread時間、
復号回数とcache hit数を記録する。初回cache missの時間にはcache書込を含む。
各sampleの画素hash照合はread timerの外で行う。
全方式は固定順で逐次実行し、hash確認やcloneがfilesystem cacheを温める。
OS cacheの破棄やfsyncによる永続化待ちは行わないため、初回passをcold read、
結果を並列throughputや耐久書込の測定とは呼ばない。

各方式のinventoryは「形式のtemplate + その方式の全attempt/cache」の和集合を数える。
native templateは元store、変換templateは新規ImageRecordであり、両方の総保持量を求める
場合は別途全体inventoryを取る。sourceと変換templateのinventoryも保存する。
logical bytes、inode重複除外後のallocated bytes、files、hardlink数を確認できる。
方式間で共通のtemplateを持つため、各方式の容量を単純加算しない。
これはGCの削減可能量や実解放量の保証ではない。

reportの `adopted_policy` は常にnull。実素材・実attempt数での結果、利用readerとの整合、
予算と保持方針を確認した後、採用理由を別の意思決定記録に残す。
