# environment — 環境の癖

このマシン・OS・ハードウェアの癖。1行に1エントリを書き、リンク文字列はエントリの `title` と一致させる。形式は [memory skill](../../skills/memory/SKILL.md#形式) を参照。

- [WSL2 のホストRAM不足では、学習プロセスがtracebackなしで黙って殺される](wsl2-host-ram-silent-kill.md)
- [WSL2 でGPU推論中に nvidia-smi を短い間隔で呼ぶと、推論とドライバ照会がそろって止まり、画面も固まる](wsl2-nvidia-smi-polling-stalls-gpu.md)
- [WSL2 で 16GB の VRAM を使い切ると、エラーにならずホスト全体が周期的に固まる](wsl2-vram-overflow-freezes-host.md)
- [WSL2 の Blender EEVEE は、環境変数を指定しないと黙って CPU（llvmpipe）で描画する](wsl2-blender-eevee-gpu-needs-d3d12-env.md)
- [WSL2 のdistroが再起動しても boot_id は変わらず、旧PID名前空間のプロセスはすべて消える](wsl2-distro-restart-keeps-boot-id.md)
