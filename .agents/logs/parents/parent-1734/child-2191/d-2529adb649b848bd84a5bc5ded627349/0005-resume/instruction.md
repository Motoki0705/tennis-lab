## 親の指示: 正式レビューのmajor 1件（PR #2222）への対応

指摘: 既定のclipモードでも、追加された `mode`・`video_shared` の設定がcourtの設定digestに入る（configuration.py:509 → definition.py:162 のcourt identity、orchestrator.py:74 の publication_identity）。その結果、次の影響がある。
- 既存のcourt cacheと、それに依存する成果物が再計算になる。
- load指定は、identityの不一致で止まる。
- 既存の annotation.json がstaleになり、明示したoverwriteが必要になる。

PR本文は、数値の互換と、共有モードのidentity変更しか説明していない。

親の判断: #2200 と同じく、**意図した変更として受け入れ、PR本文に明記する**（コードは変えない）。理由は、meiji_4camの全件処理はこれから行うもので、既存のcacheは評価用だけであること。
- PR本文に「既定clipモードのcacheへの影響」の節を追加する。「数値は不変」と「cacheは互換でない」を区別して書く。そのうえで、再計算の範囲・load拒否・annotation.jsonのstaleとoverwriteの必要性、src全体のfingerprintによる既存の全node無効化（このPRに固有ではない）を書く。
- 必要なら、実際の確認（既定YAMLでのdigestの差など）をCPUで行い、その結果を本文に引用する。本文だけを変える場合、commitは不要。
merge・closeは行わない。
