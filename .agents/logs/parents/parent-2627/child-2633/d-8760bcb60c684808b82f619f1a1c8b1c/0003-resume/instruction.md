## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261126T113407Z_23e0bc41263b_i2633-viewpoint-inputs",
      "name": "i2633-viewpoint-inputs",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 754.047,
      "log": "[local path]",
      "log_excerpt": "Verifying the frozen synthetic publication\ns': 36}, 'side_on': {'images': 0, 'videos': 0}, 'unclassified': {'images': 0, 'videos': 0}}, 'test': {'all': {'images': 4372, 'videos': 249}, 'challenging': {'images': 439, 'videos': 106}, 'low_angle': {'images': 224, 'videos': 41}, 'oblique': {'images': 194, 'videos': 78}, 'cropped': {'images': 162, 'videos': 46}, 'side_on': {'images': 0, 'videos': 0}, 'unclassified': {'images': 0, 'videos': 0}}}\nB00 selections: [('narrow', 128), ('broad', 128)]\nB01 selections: [('narrow', 128), ('broad', 128)]\nB02 selections: [('narrow', 128), ('broad', 128)]\nB03 selections: [('narrow', 128), ('broad', 128)]\nEffective bins: {'narrow': {'azimuth_deg': 7.3309038798029444, 'height_m': 3.393350632545594, 'hfov_deg': 1.6749038816003934}, 'broad': {'azimuth_deg': 7.828279444733008, 'height_m': 4.733523177052368, 'hfov_deg': 2.884336224398423}}\nPrepared [local path]"
    }
  ]
}
