| checkpoint | source / camera | observed | recall@8 | recall@1 | 候補外 | 誤候補が上位 |
|---|---|---:|---:|---:|---:|---:|
| ft-e13 | chat_annotation | 6622 | 86.107% | 81.773% | 13.893% | 4.334% |
| ft-e13 | meiji | 23007 | 74.251% | 52.627% | 25.749% | 21.624% |
| ft-e13 | meiji/cam0 | 7556 | 65.802% | 38.208% | 34.198% | 27.594% |
| ft-e13 | meiji/cam1 | 7999 | 81.885% | 59.570% | 18.115% | 22.315% |
| ft-e13 | meiji/cam2 | 7452 | 74.624% | 59.796% | 25.376% | 14.828% |
| ft-e13 | tracknet | 1538 | 98.570% | 98.440% | 1.430% | 0.130% |
| mixed-e0 | chat_annotation | 6622 | 87.466% | 81.592% | 12.534% | 5.874% |
| mixed-e0 | meiji | 23007 | 86.956% | 69.579% | 13.044% | 17.377% |
| mixed-e0 | meiji/cam0 | 7556 | 82.186% | 62.295% | 17.814% | 19.891% |
| mixed-e0 | meiji/cam1 | 7999 | 92.562% | 77.047% | 7.438% | 15.514% |
| mixed-e0 | meiji/cam2 | 7452 | 85.776% | 68.948% | 14.224% | 16.828% |
| mixed-e0 | tracknet | 1538 | 98.309% | 97.529% | 1.691% | 0.780% |
| mixed-e11 | chat_annotation | 6622 | 89.293% | 84.129% | 10.707% | 5.165% |
| mixed-e11 | meiji | 23007 | 90.220% | 77.968% | 9.780% | 12.253% |
| mixed-e11 | meiji/cam0 | 7556 | 85.204% | 70.566% | 14.796% | 14.637% |
| mixed-e11 | meiji/cam1 | 7999 | 95.074% | 84.086% | 4.926% | 10.989% |
| mixed-e11 | meiji/cam2 | 7452 | 90.097% | 78.905% | 9.903% | 11.192% |
| mixed-e11 | tracknet | 1538 | 98.895% | 98.114% | 1.105% | 0.780% |

距離≤20 source px、K=8/NMS=5/patch=5/subpixel。全率の分母はobserved frame。
誤候補が上位 = 正解候補が存在しtop-1が誤り。同scoreはdecoder順を保持し、strict score版と同率件数をJSONに併記。
camera IDのないsourceはcamera別値を作らない。

| checkpoint | sha256 |
|---|---|
| ft-e13 | `cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892` |
| mixed-e0 | `7b9a202b7753edc9edc200271edfad1b09bdee59073bdbaffb2709cbf942aa9b` |
| mixed-e11 | `6a9c0ef21a19241638ae279131f9b7211c47c7fb6ddb4b0753fe762d9026961f` |

選定: `mixed-e11`。Meiji video_000だけで決定。
mixed e11 > e0: `True`。
