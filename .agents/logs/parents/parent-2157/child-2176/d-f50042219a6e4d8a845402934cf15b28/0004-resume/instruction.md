# PR #2184 のCI失敗の修正（#2176）

HEAD daeb87a16 のCI（run 43819061297）で、`CPU tests (9)` の1件がERRORになり、集約の `Python` も失敗しました。直前のHEAD 4017f9a1c では15件すべて成功しています。

```
ERROR tests/e2e/agents_ops/test_parent_judgement_workflow.py::test_parent_can_verify_artifacts_from_an_ended_failed_turn - FileNotFoundError: [Errno 2] No such file or directory: 'maintenance.lock'
```

tmpディレクトリの削除（shutil.rmtree）中に `maintenance.lock` が消えており、テスト終了時点で別プロセス（切り離した処理・runner・dispatcherなど）がまだ動いていて同じディレクトリを触った競合に見えます。ただし今回の変更（`targeted_history` の継承、手動修正起動の取得制限など）が原因かどうかは未確認です。

## 依頼
1. 原因を特定してください。今回のPRの変更でプロセスの終了・後片付けの順序が変わったのか、もともと存在する競合（mainでも起こりうる）なのかを、根拠付きで報告してください。
2. 原因に応じて修正してください。テスト終了前に起動したプロセスの終了を確認する・ロックの扱いを直すなど、競合そのものを無くしてください。再実行・sleep・例外の握りつぶしで隠さないでください。
3. 修正後、該当テストを繰り返し実行（例: `-n auto` で `--count` 相当や複数回）して安定を確認し、関連するagents_opsのe2eも実行してください。

push後に、原因・修正内容・テスト結果・新しいHEADを報告して終了してください。CIの完了は待たなくてよい。

## CIログ（抜粋）
```
tests/support/agents_ops/directory_seed.py:29: in restore
    shutil.rmtree(destination)
../../_temp/uv-python-dir/cpython-3.11.18-linux-x86_64-gnu/lib/python3.11/shutil.py:752: in rmtree
    _rmtree_safe_fd(fd, path, onerror)
../../_temp/uv-python-dir/cpython-3.11.18-linux-x86_64-gnu/lib/python3.11/shutil.py:672: in _rmtree_safe_fd
    _rmtree_safe_fd(dirfd, fullname, onerror)
../../_temp/uv-python-dir/cpython-3.11.18-linux-x86_64-gnu/lib/python3.11/shutil.py:672: in _rmtree_safe_fd
    _rmtree_safe_fd(dirfd, fullname, onerror)
../../_temp/uv-python-dir/cpython-3.11.18-linux-x86_64-gnu/lib/python3.11/shutil.py:672: in _rmtree_safe_fd
    _rmtree_safe_fd(dirfd, fullname, onerror)
../../_temp/uv-python-dir/cpython-3.11.18-linux-x86_64-gnu/lib/python3.11/shutil.py:703: in _rmtree_safe_fd
    onerror(os.unlink, fullname, sys.exc_info())
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

topfd = 17
path = '[local path]'
onerror = <function rmtree.<locals>.onerror at 0x79ebcbb60900>

    def _rmtree_safe_fd(topfd, path, onerror):
        try:
            with os.scandir(topfd) as scandir_it:
                entries = list(scandir_it)
        except OSError as err:
            err.filename = path
            onerror(os.scandir, path, sys.exc_info())
            return
        for entry in entries:
            fullname = os.path.join(path, entry.name)
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_dir = False
            else:
                if is_dir:
                    try:
                        orig_st = entry.stat(follow_symlinks=False)
                        is_dir = stat.S_ISDIR(orig_st.st_mode)
                    except OSError:
                        onerror(os.lstat, fullname, sys.exc_info())
                        continue
            if is_dir:
                try:
                    dirfd = os.open(entry.name, os.O_RDONLY | os.O_NONBLOCK, dir_fd=topfd)
                    dirfd_closed = False
                except OSError:
                    onerror(os.open, fullname, sys.exc_info())
                else:
                    try:
                        if os.path.samestat(orig_st, os.fstat(dirfd)):
                            _rmtree_safe_fd(dirfd, fullname, onerror)
                            try:
                                os.close(dirfd)
                            except OSError:
                                # close() should not be retried after an error.
                                dirfd_closed = True
                                onerror(os.close, fullname, sys.exc_info())
                            dirfd_closed = True
                            try:
                                os.rmdir(entry.name, dir_fd=topfd)
                            except OSError:
                                onerror(os.rmdir, fullname, sys.exc_info())
                        else:
                            try:
                                # This can only happen if someone replaces
                                # a directory with a symlink after the call to
                                # os.scandir or stat.S_ISDIR above.
                                raise OSError("Cannot call rmtree on a symbolic "
                                              "link")
                            except OSError:
                                onerror(os.path.islink, fullname, sys.exc_info())
                    finally:
                        if not dirfd_closed:
                            try:
                                os.close(dirfd)
                            except OSError:
                                onerror(os.close, fullname, sys.exc_info())
            else:
                try:
>                   os.unlink(entry.name, dir_fd=topfd)
E                   FileNotFoundError: [Errno 2] No such file or directory: 'maintenance.lock'
```
