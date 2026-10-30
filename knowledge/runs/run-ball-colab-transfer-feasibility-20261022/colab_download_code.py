import pathlib,json,subprocess,time,datetime,os,hashlib,shutil
out=pathlib.Path("outputs/ball_detection/transfer_probe/20261022-colab")
samples=[{'repeat': 1, 'bytes': 544686517, 'files': [{'name': 'clip-00053.bin', 'bytes': 13038519, 'sha256': '78f61d1c1735275437a464c70e670325b3294b8f3a98e70b4a27a1e9b162765b'}, {'name': 'clip-00177.bin', 'bytes': 40210890, 'sha256': '99e474737bfbefe343b2e1d83edeee9444e571b81a7957eb0ca492c0162a2bcd'}, {'name': 'clip-01101.bin', 'bytes': 57555604, 'sha256': '22e26aa4a66558b9e4c141b401ffbf1f401acdcdc7a0a261d002f48a9fcb8c2e'}, {'name': 'clip-01217.bin', 'bytes': 93777319, 'sha256': 'a9f70042c9bd576e9290a7260bfd49d087f2dfcc5a83cffee05d3e43703ba18e'}, {'name': 'clip-01327.bin', 'bytes': 143876798, 'sha256': '592500e30436973ebb81b0ea75654d00ccf77dbd938ade1b39b45464d0fcbcd1'}, {'name': 'clip-01537.bin', 'bytes': 196227387, 'sha256': '4915bc97df6b0b6df016b9ead2c526d287bd04f361cc35894f2d4f0eab77130c'}]}, {'repeat': 2, 'bytes': 586327937, 'files': [{'name': 'clip-00069.bin', 'bytes': 20409027, 'sha256': 'ffe9d31d61d0fe2fffe8bfb4428ff11fdfd34dbbadfe159984221e18db2ac9b5'}, {'name': 'clip-00176.bin', 'bytes': 57982790, 'sha256': 'ca793993d1277a4ed38818bd1beced96efefabedd72f5f8f0bf1cb00f26a1ca8'}, {'name': 'clip-01419.bin', 'bytes': 59009601, 'sha256': '28ca66064efa525b644ebb120c660dffedd64a7f22eec4e6f71fb2e1afa19454'}, {'name': 'clip-00277.bin', 'bytes': 99839991, 'sha256': '7bef445b53638f09c62a4f535c388f3517399875f68ac676d53ba0ef6291f901'}, {'name': 'clip-01179.bin', 'bytes': 147250362, 'sha256': 'a3762ec8213edd46b82a5a7ffc9f325f09a59b59a0535e4d7a1ad270a67a4f67'}, {'name': 'clip-00999.bin', 'bytes': 201836166, 'sha256': 'd0b9e6b5f5980dfa8678da4af7f8352cb0791c01fe9a5ea4ef3dea8487a20bcc'}]}, {'repeat': 3, 'bytes': 650089269, 'files': [{'name': 'clip-00047.bin', 'bytes': 29965467, 'sha256': 'c71893c8e0084c1a75c34b29a48eb8f6dd78d9bd97663fde2bc5e9c919774899'}, {'name': 'clip-00223.bin', 'bytes': 95484618, 'sha256': '282817b6b49f4b56e54d380c14a4fb373c93cf6c336693dd4f8020e5c5844571'}, {'name': 'clip-01346.bin', 'bytes': 59748971, 'sha256': 'fd41df01f2233fa54eb012de20f29bde286ad814d655d312661362cafb94b854'}, {'name': 'clip-01776.bin', 'bytes': 103561783, 'sha256': 'eb5757344c7ef7e8274659baa70f4210b60dc5ae33a5405ae843b658c6c8c114'}, {'name': 'clip-00519.bin', 'bytes': 152076164, 'sha256': '518ef51a7d753e1dd4235d129ec900ab4003f1035900870602e4a8a019622ce6'}, {'name': 'clip-00667.bin', 'bytes': 209252266, 'sha256': '3426f8d82269e7781a6e5620623208fd37901de9bd16c3732f9ff1b56f581903'}]}]
for sample in samples:
 n=sample['repeat']
 source=f"staging/ball-detection-transfer-probe-20261022/sample-{n}"
 destination=pathlib.Path(f"data/ball_detection/transfer-probe-20261022/sample-{n}")
 cmd=['python3','.agents/skills/tennis-drive/scripts/drive.py','--timeout-seconds','900','download',source,str(destination),'--verify']
 env=os.environ.copy();env['RCLONE_LOG_LEVEL']='INFO';env['RCLONE_STATS_LOG_LEVEL']='NOTICE'
 started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
 with (out/f'download-{n}.stdout.json').open('w') as stdout,(out/f'download-{n}.stderr.log').open('w') as stderr:
  p=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=env)
 elapsed=time.perf_counter()-t
 result={'schema_version':1,'direction':'drive_to_colab','repeat':n,'started_at':started,'elapsed_seconds':elapsed,'bytes':sample['bytes'],'end_to_end_MB_per_second':sample['bytes']/elapsed/1e6,'returncode':p.returncode,'command':cmd,'disk_usage_after':shutil.disk_usage('/content')._asdict(),'timing_includes':'path uniqueness checks, transfer, post-readback, local/remote hash verification; frozen SHA256 check below timed separately'}
 checks=[];t=time.perf_counter()
 if p.returncode==0:
  for row in sample['files']:
   path=destination/row['name']
   actual=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
   checks.append({'file':row['name'],'expected_sha256':row['sha256'],'actual_sha256':actual,'expected_bytes':row['bytes'],'actual_bytes':path.stat().st_size,'match':actual==row['sha256'] and path.stat().st_size==row['bytes']})
 result['frozen_sha256_checks']=checks;result['frozen_hash_seconds']=time.perf_counter()-t
 (out/f'download-{n}.timing.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='frozen_sha256_checks'}),flush=True)
 if p.returncode or not all(c['match'] for c in checks): raise RuntimeError('Transfer or content validation failed')
