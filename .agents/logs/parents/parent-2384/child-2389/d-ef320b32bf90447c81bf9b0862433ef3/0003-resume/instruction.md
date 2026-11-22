## 自動進行の修正ターン

このターンは同じモデル・同じsessionのeffort=highで行う修正。
失敗情報を調べ、自分で直せるか親AIの判断が必要かを決める。
直せる場合は実装・意味のある検証・pre-commit・commit・pushを行い、その直後にターンを終了する。
CIの待機は呼出側CLIが担当する。workerはpush後にCIを待たない。
親の判断が必要ならoutcomeをneeds_parentにし、parent_requestへ具体的な依頼を書く。
最終報告は次のschemaを満たすJSONオブジェクト1つだけにする。
repairedのparent_requestはnull、needs_parentでは非空文字列を必須にする。
head_shaは報告時の完全な40桁commit。verificationには実施した検証と結果・未実施を記す。
修正の識別情報は {"schema_version": 1, "repair_id": "[private identifier]", "pr": 2398}
報告schemaは {"type": "object", "additionalProperties": false, "required": ["schema_version", "repair_id", "pr", "outcome", "head_sha", "summary", "verification", "parent_request"], "properties": {"schema_version": {"type": "integer", "const": 1}, "repair_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"}, "pr": {"type": "integer", "minimum": 1}, "outcome": {"enum": ["repaired", "needs_parent"]}, "head_sha": {"type": "string", "pattern": "^[a-f0-9]{40}$", "maxLength": 40}, "summary": {"type": "string", "pattern": "[local path]"}, "verification": {"type": "array", "items": {"type": "string", "pattern": "[local path]"}}, "parent_request": {"type": ["string", "null"], "pattern": "[local path]"}}}

{
  "reason": "conflict",
  "head": "36688ce117ad29423edc5509365f16a750219bc4"
}