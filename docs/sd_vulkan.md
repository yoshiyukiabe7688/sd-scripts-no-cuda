# SD1.5 / SDXL を CUDA 無しで学習する（Vulkan バックエンド）

Anima と同じく、自作デバイス `vk`（[`torch-vulkan`](https://github.com/yoshiyukiabe7688/torch-vulkan)）で、SD1.5 と SDXL の LoRA 学習を Vulkan 上で実行します。
学習スクリプト（`train_network.py` / `sdxl_train_network.py`）は無改造で、環境変数 `SD_SCRIPTS_BACKEND=vk` を付けるだけです。
ビルド・必要な Vulkan 機能・環境変数は [anima_vulkan.md](anima_vulkan.md) と同じです。

## 検証済みの環境と結果
学習データは人物 12 枚（512px、キャプション付き）、LoRA rank 16、bf16（`--full_bf16`）、勾配チェックポイントあり、300 step。
学習した LoRA を **stable-diffusion.cpp**（master-1053-5a1ba5d）で読み込み、未学習の 3 場面（駅のホーム・雪の街・神社）を LoRA あり／なしで同一シードで生成して確認しました。
LoRA なしは場面ごとに別の人物になり、ありはキャラクター（銀髪ツインテール・赤いリボン・青い目）が 3 場面とも再現されます（目視）。

| モデル | GPU | 速度（512px） | メモリ |
|---|---|---|---|
| SD1.5 | Radeon RX 9070 XT（Windows） | 約 0.7 秒/step（300 step で約 4 分） | — |
| SD1.5 | Radeon RX 550 4GB（Linux、Mesa RADV） | 約 8 秒/step（300 step で約 41 分） | GPU メモリ 約 2.7GB（定常）、最大 3.4GB |
| SDXL | Radeon RX 9070 XT（Windows） | 約 1.2〜1.3 秒/step（300 step で約 6.5 分） | — |
| SDXL（`--int8_base`） | Radeon RX 550 4GB（Linux、Mesa RADV） | 約 15 秒/step（300 step で約 1 時間 15 分） | GPU メモリ 定常 約 3.2GB、最大 3.7GB（4GB の約 77〜90%） |

- 実機で確認したのは上の 4 つの組み合わせだけです。SDXL の UNet は bf16 で約 5.2GB あり 4GB の GPU には載らないので、4GB 級では `--int8_base`（下の節）が必要です。
- SD1.5 の UNet（乱数重み、LoRA 付き）の fwd+bwd を CPU の fp32 と比べると、fp32 では出力の相対誤差 3e-6、LoRA 勾配 2e-5 以内、bf16 では出力 1.4%、LoRA 勾配の中央値 2.5% です（bf16 の丸め誤差の範囲）。

## 使い方
`torch-vulkan` をビルドし、`PYTHONPATH` に通します（[anima_vulkan.md](anima_vulkan.md) の手順 1）。

```powershell
$env:SD_SCRIPTS_BACKEND = "vk"
$env:PYTHONPATH = "C:\Users\<you>\dev\torch-vulkan"
python train_network.py `
  --pretrained_model_name_or_path=<v1-5-pruned-emaonly.safetensors> `
  --dataset_config=<dataset.toml> --output_dir=<out> --output_name=<name> --save_model_as=safetensors `
  --network_module=networks.lora --network_dim=16 --network_alpha=16 `
  --learning_rate=1e-4 --optimizer_type=AdamW --max_train_steps=300 `
  --mixed_precision=bf16 --full_bf16 --save_precision=bf16 `
  --cache_latents --cache_latents_to_disk --sdpa --network_train_unet_only `
  --max_data_loader_n_workers=0 --gradient_checkpointing --clip_skip=1
```

SDXL は `sdxl_train_network.py` に替え、`--clip_skip=1` の代わりに `--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk` を付けます。それ以外は同じです。
データセットの解像度は、TOML の `resolution`（例 512）で指定します。

## 重みのみ INT8 で VRAM を半分にする（`--int8_base`、オプトイン）
`--int8_base` を付けると、凍結しているベースの Linear / Conv2d の重みだけを int8（64 要素ごとの bf16 scale）で保持します。LoRA・Norm・bias は元の dtype のままで、指定しなければ従来どおり bf16 です。
計算は使う直前に bf16 へ戻して行うので、速度はほとんど変わりません（SD1.5 の 9070 XT で 300 step: bf16 252 秒、int8 255 秒）。

```powershell
python sdxl_train_network.py ... --int8_base
```

| モデル | UNet の重み（vk 上の確保量） bf16 | `--int8_base` |
|---|---|---|
| SD1.5 | 1639 MiB | 845 MiB |
| SDXL | 4897 MiB | 2526 MiB |

- **SDXL が RX 550（4GB）で学習できます**（512px、batch 1、rank 16、勾配チェックポイントあり）。300 step を 15.05 秒/step で完走し、loss は 0.32 → 0.12 前後に下がりました。学習した LoRA を sd.cpp で 3 場面生成すると、LoRA ありはキャラクター（銀髪ツインテール・赤いリボン・青い目）が再現され、なしは別人でした（目視。合格）。
  - GPU メモリは定常 約 3.2GB、起動直後に最大 約 3.7GB まで上がります。上限を強制した実験では、`VKT_MEM_LIMIT_MB=3072` は通り、2560 は確保に失敗しました（必要量は 2.5〜3GB の間）。4GB の GPU でも、表示用や他のプロセスが多く使っている場合は余裕が少ないので、注意してください。
  - 同じ GPU で `--int8_base` なしの bf16 を試すと、OOM にはなりませんが 24 秒/step と遅くなりました（重みが VRAM に収まらず、システムメモリへ溢れているためと見ています。ドライバ側の挙動は未確認）。上限を 3072 MiB に強制すると確保に失敗します。
  - CPU フォールバックは全体の約 3%（`linalg_vector_norm` と `mse_loss`）でした。
- 劣化: 重みの相対誤差は中央値で約 0.6%。SD1.5・SDXL とも、LoRA の loss は量子化なしとほぼ同じで、3 場面ともキャラクターが再現されました（9070 XT で検証。比較画像はそれぞれ bf16 と並べて目視）。本物の int8 は scale を bf16 に丸めるため、fake 量子化（scale が fp32）や bf16 とは**ビット一致しません**。
- `--int8_base_skip <正規表現>` で、特定のモジュール名だけ元の dtype のまま残せます。通常（交換型）のブロックスワップ（`--blocks_to_swap`）とは併用できません。Anima と `--vram_budget_mb` の併用は [anima_vulkan.md](anima_vulkan.md) を参照してください。
- `--blocks_to_swap` / `--vram_budget_mb` を SDXL・SD1.5 に付けた検証はしていません（UNet 系のスワップ対応は未確認）。

## 実装の概要（torch-vulkan 側）
SD1.5 / SDXL の UNet が学習で使う演算のうち、Anima で未実装だったのは 3 系統だけです（46 種の ATen 演算のうち）。
- **conv2d**（forward / backward）: im2col + 既存の行列積（協調行列 / タイル型）。NCHW・groups=1・dilation=1。1x1・stride 1 は im2col を省く。ベース重みは凍結なので、backward は入力の勾配が中心。重みの勾配は LoRA の 1x1 conv だけ。
- **GroupNorm**（forward / backward）: 1 つのワークグループが 1 つの (n, group) を処理する。起動時に `maxComputeWorkGroupInvocations` が 1024 以上であることを確認する。
- **nearest upsample**（整数倍のみ、forward / backward）。

## 制限・注意
- groups ≠ 1 や dilation ≠ 1 の conv、整数倍でない upsample は、CPU にフォールバックします（結果は合います）。conv LoRA（`conv_dim`）は未検証です。
- VAE の畳み込みは、潜在のキャッシュ作成（CPU）を前提にしています。学習中のサンプル画像生成は、conv が GPU で動くようになったので、以前より速くなる見込みですが、未測定です。
- 画像サイズが 512px を超える場合のメモリと速度は未測定です。SDXL の標準（1024px）は未検証です。
- このほか [anima_vulkan.md](anima_vulkan.md) の「制限・注意」（bf16 の勾配の丸め誤差、Windows の AMD ドライバでの fp16 など）がそのまま当てはまります。
