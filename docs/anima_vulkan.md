# Anima / Anima-Turbo を CUDA 無しで学習する（Vulkan バックエンド）

CUDA が使えない環境（AMD GPU など）で、Anima / Anima-Turbo の LoRA 学習を **Vulkan** 上で実行する手順です。
PyTorch の標準 Vulkan は推論専用なので、学習用の自作デバイス `vk`（PrivateUse1）を別プロジェクト [`torch-vulkan`](https://github.com/yoshiyukiabe7688/torch-vulkan)（公開・MIT ライセンス）として用意しています。
学習スクリプト（`anima_train_network.py`）は無改造で、環境変数 `SD_SCRIPTS_BACKEND=vk` を付けるだけで GPU を使います。
CPU のみでも学習できます（環境変数なし。bf16 で約 10 秒/step）。

## 検証済みの環境と結果
- Windows 11 / Ryzen 7 7800X3D / **Radeon RX 9070 XT**（Vulkan 1.4、ドライバ 26.10.1）、PyTorch 2.8.0+cpu。
- Anima-base v1.0 と Anima-Turbo v1.1 の LoRA 学習（512px、batch 1、bf16、rank 16）: **約 1.4 秒/step**（CPU bf16 の約 7 倍）。
- 学習した LoRA を **stable-diffusion.cpp**（master-1053-5a1ba5d、Vulkan ビルド）で読み込んで生成できることを確認済み（840/840 テンソルが適用）。
  オリジナルキャラクター（銀髪ツインテール・赤いリボン・青い目）を 12 枚で学習し、トリガーワード付きの未学習の場面で、LoRA ありはキャラクターが再現され、なしでは再現されない。base と Turbo の両方で確認。
- Linux（Ubuntu、Mesa RADV）/ **Radeon RX 550 4GB**（Polaris。fp16 演算・協調行列なし）: Anima-base の LoRA 学習（384px、300 step、`--blocks_to_swap 24`）が完走し、
  生成した人物 LoRA でも同様にキャラクターが再現されることを確認（詳細は「古い GPU 向けの調整」）。
- Windows 11 / **Radeon RX 640**（Polaris 23、4GB、AMD 公式ドライバ 26.5.2、Vulkan 1.3.260。fp16 演算・協調行列なし）: 384px・8 step の学習が完走
  （`--blocks_to_swap 24` で約 22 秒/step、`--vram_budget_mb` 指定または auto で約 19 秒/step）。
  このドライバは `maxPushConstantsSize` に 128 バイト（Vulkan の保証最小値）を報告するため、要素演算の push constant を 128 バイトに収めている。
- Windows 11 / **Intel Arc 140V**（内蔵、メモリ共有、協調行列の拡張はあるがカーネルが使う形が無いのでタイル型）: 384px・8 step の学習が完走（スワップなしで約 12.7 秒/step）。
  `VKRT_DEVICE=Arc`（番号か名前の一部）で使うデバイスを選ぶ（未指定なら最初の dGPU）。
- **実機で確認したのは上の 4 機種だけ**です。それ以外の GPU（NVIDIA を含む）は未検証です。RX 640 と Arc の確認は 384px・8 step までで、300 step の学習と、生成した人物 LoRA の確認はしていません。

## 必要なもの
- Vulkan 対応 GPU とドライバ。次の機能が必須です（無いデバイスでは起動時に、足りない機能名つきのエラーになります）:
  `bufferDeviceAddress`、`shaderInt64`、`shaderInt16`、`shaderInt8`、`storageBuffer8BitAccess`、`storageBuffer16BitAccess`。
  `maxPushConstantsSize` は 128 バイト以上（Vulkan の保証最小値）。
  fp16 演算（`shaderFloat16`）と協調行列（`VK_KHR_cooperative_matrix`。カーネルが使う 16x16x16 の形があるときだけ有効）は任意です。協調行列があると行列積が速くなります（無ければ自動でタイル型 GEMM に切り替わる）。
- `torch-vulkan` のビルド: Visual Studio 2022（MSVC）、Rust（cargo）、Vulkan SDK（シェーダを再生成する場合のみ。生成済み SPIR-V はコミット済み）。
- `torch-vulkan` は sd-scripts とは別リポジトリです。<https://github.com/yoshiyukiabe7688/torch-vulkan> を任意の場所に clone してください（例: `C:\Users\<you>\dev\torch-vulkan`）。

## 手順
1. `torch-vulkan` をビルドする（`run_build.cmd`。MSVC 環境を読み込み、cargo で Rust ランタイムを、続いて C++ シムをビルドする）。
2. **潜在とテキスト出力のキャッシュは CPU で作る**（VAE と Qwen3 は学習の演算に含まれず、vk では動かさない）。
   `SD_SCRIPTS_BACKEND` を設定せずに、`--cache_latents_to_disk --cache_text_encoder_outputs_to_disk` 付きで 1 step だけ実行する。
3. キャッシュができたら、同じ設定で `vk` を使って学習する。

```powershell
$env:SD_SCRIPTS_BACKEND = "vk"
$env:PYTHONPATH = "C:\Users\<you>\dev\torch-vulkan"   # torch_vulkan を import できるように
python anima_train_network.py `
  --pretrained_model_name_or_path=<anima-base-v1.0.safetensors か anima-turbo-v1.1.safetensors> `
  --qwen3=<qwen_3_06b_base.safetensors> --vae=<qwen_image_vae.safetensors> `
  --dataset_config=<dataset.toml> --output_dir=<out> --output_name=<name> --save_model_as=safetensors `
  --network_module=networks.lora_anima --network_dim=16 --network_alpha=16 `
  --learning_rate=3e-4 --optimizer_type=AdamW --max_train_steps=500 `
  --mixed_precision=bf16 --full_bf16 --save_precision=bf16 `
  --timestep_sampling=sigmoid --discrete_flow_shift=1.0 --attn_mode=torch `
  --cache_latents --cache_latents_to_disk --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk `
  --network_train_unet_only --max_data_loader_n_workers=0
```

## 少ない VRAM の GPU で動かす（`--vram_budget_mb`）
VRAM が少ない GPU（4GB 級。表示用に取られる分を除いた 3GB 前後）でも学習できます。`--vram_budget_mb` に使ってよい VRAM（MiB）を渡すと、
**収まるように勾配チェックポイントとブロックスワップ（H2D 専用）の設定を自動で決めます**（`--gradient_checkpointing` や `--blocks_to_swap` を自分で指定する必要はありません）。

```powershell
python anima_train_network.py ... --vram_budget_mb 3072     # 3GB に収める
python anima_train_network.py ... --vram_budget_mb auto     # デバイスが報告する使用可能量（表示用などを除いた分）から決める
```

- 仕組み: ① 勾配チェックポイントで活性化を約 5GB → 約 0.3GB に減らす。② 凍結しているベース重み（Anima で約 3.7GB）の一部を CPU に置き、
  GPU 上の 2 枠のリングバッファへ順番に転送して使う（書き戻しはしない）。どちらも**計算結果は変わりません**（予算なしの学習と LoRA がビット一致することを確認済み）。
- 予算ごとの構成（Anima-base、512px、batch 1、bf16、rank 16 の実測。使用量は GPU に確保したメモリ）:

| 予算 | スワップするブロック数（28 中） | 実測のピーク確保量 | 速度 |
|---|---|---|---|
| なし（16GB） | 0（チェックポイントも無し） | 約 9.0GB | 約 1.06 秒/step |
| 3072 MiB | 15 | 約 2.7GB | 約 1.9 秒/step |
| 2048 MiB | 23 | 約 1.65GB | 約 2.0 秒/step |
| 1536 MiB | 26（上限） | 上限を強制すると、実学習では確保に失敗する（スラブ内の断片化）。**未達** | — |

- `vk` で数値を指定したときは、その値を**上限として強制**します（超えると OOM）。1536 MiB のように、収めるのが厳しい上限では、断片化で確保に失敗することがあります。2048 MiB 以上で確認しています。大きい GPU で小さい GPU を模擬して、収まるかを試すのに使えます。
  `auto` は強制しません（OS が許す予算を VK_EXT_memory_budget で取得して決めるだけ）。
- `--blocks_to_swap` を自分で指定したときは `--vram_budget_mb` は無視されます。`--cpu_offload_checkpointing` / `--unsloth_offload_checkpointing` とは併用できません。
- 見積もりは 512px・rank 16 の実測を元にしています（解像度と rank に応じて補正）。余裕を持たせてあるので、OOM になる場合は予算を下げてください。
- メインメモリに、スワップするブロック分の重み（最大で約 3.5GB）が必要です。

### さらに小さくする: `--int8_base`（重みのみ INT8、オプトイン）
`--int8_base` を付けると、**凍結しているベースの Linear / Conv2d の重みだけ**を int8（64 要素ごとの bf16 scale、対称量子化）で保持します（メモリは約 1.03 バイト/要素。bf16 の約半分）。
指定しなければ従来どおり bf16 のままです。LoRA・Norm・bias・埋め込みは元の dtype のままで、計算は使う直前に bf16 へ戻して行います（学習で動くのは LoRA だけで、量子化誤差は固定の定数として働きます）。

```powershell
python anima_train_network.py ... --int8_base --vram_budget_mb 2048
```

- **`--vram_budget_mb` と併用すると、プランナーが int8 のサイズで見積もる**ので、スワップするブロックが減ります（Anima-base、512px、batch 1、rank 16、9070 XT で実測。bf16 は上の表）:

| 予算 | スワップするブロック数（28 中） | 速度 | 結果 |
|---|---|---|---|
| 3072 MiB | 14〜15 | 約 1.4 秒/step | 完走、loss は bf16 と同一 |
| 2048 MiB | 15（bf16 では 23） | 約 1.4 秒/step | 同上 |
| 1536 MiB | 22（bf16 では強制すると確保に失敗） | 約 1.4 秒/step | 完走（6 step）、loss は同一 |

  bf16 のスワップ構成（約 1.9 秒/step）より速くなるのは、転送するデータが減るためです。
- 劣化: 重みの相対誤差は中央値で約 0.6%（bf16 自体の丸め誤差の 1.5〜3 倍程度）。LoRA の loss は量子化なしとほぼ同じで、生成も崩れませんでした（SD1.5 / SDXL は 3 場面でキャラクター再現を確認。Anima は雪の街の 2 枚で顔が少し崩れましたが、シード由来か量子化由来かは未切り分けです）。**ビット一致はしません**（bf16 のベースとは別の計算になります）。
- `--int8_base_skip <正規表現>` で、特定のモジュール名だけ元の dtype のまま残せます。
- 必須の Vulkan 機能は変わりません（`shaderInt8` / `storageBuffer8BitAccess` は元から必須）。dequant は専用カーネルで行います（[2048, 8192] の bf16 出力で約 0.05 ms。温まった状態での測定）。
- 通常（交換型）のブロックスワップとは併用できません（エラーになります）。H2D 専用スワップ（vk の既定、`--vram_budget_mb`）は int8 の buffer も流します。

## stable-diffusion.cpp で使う
出力した LoRA を `--lora-model-dir` のフォルダに置き、プロンプトに `<lora:名前:1.0>` を付ける。
Turbo は 8〜12 step・CFG 1.0、base は 25 step 前後・CFG 5〜6 が目安。

```
sd-cli.exe --diffusion-model anima-base-v1.0.safetensors --vae qwen_image_vae.safetensors --llm qwen_3_06b_base.safetensors ^
  --lora-model-dir <LoRA のフォルダ> -p "masterpiece, best quality, safe, ohwx, 1girl, solo, <服のパーツ> <lora:name:1.0>" ^
  -W 512 -H 512 --steps 25 --cfg-scale 5.0 --sampling-method euler -o out.png
```

## 学習データのキャプションのコツ
- Anima のテキストエンコーダは小さい（0.6B）ので、曖昧な指示だとデータセットの特徴（肌色成分など）が混ざりやすい。
  **着せたい服のパーツはすべて言語化する**（例: `white short-sleeved blouse, red bowtie, navy blue pleated skirt, white knee-high socks, brown loafers`）。
- 覚えさせたいキャラクターの特徴（髪・目など）はキャプションに書かず、トリガーワード（例: `ohwx`）に覚えさせる。
- 画像が成人向けに偏らないよう、プロンプトには `safe` を付け、ネガティブに `nsfw, nude` などを入れる。

## 古い GPU（VRAM 4GB 級）向けの調整
RX 550 4GB（Polaris、Linux の Mesa RADV）で実際に動かして確認した内容です。GTX 1060 級も想定していますが、**実機での検証は RX 550 のみ**です。
この世代の GPU には協調行列も fp16 演算も無く、行列積は最大でも約 0.5 TFLOPS です。そのため RX 9070 XT（約 1.4 秒/step）より大幅に遅く、**学習データの解像度を下げるのが最も効果の大きい高速化**です。

### 解像度を下げると速くなる
Anima の DiT は画像をトークンに分割して処理するので、トークン数は解像度の 2 乗に比例します（512px で 1024、384px で 576）。
**学習が遅いときは、まず解像度を 512 → 384 や 320 に下げてください。** 画像の枚数や中身はそのままで構いません。

| 学習解像度 | トークン数 | 秒/step | 300 step の目安 | GPU メモリ最大 |
|---|---|---|---|---|
| 512 | 1024 | 36.2 | 約 3.0 時間 | 約 3.2GB |
| 448 | 784 | 30.7 | 約 2.6 時間 | 約 3.2GB |
| 384 | 576 | 21.9 | 約 1.8 時間 | 約 1.6GB |
| 320 | 400 | 17.5 | 約 1.5 時間 | 未確認 |

（RX 550 4GB、batch 1、rank 16、bf16、`--blocks_to_swap 24`、勾配チェックポイントあり。）

- 解像度は、データセット設定（TOML）の `resolution` に **16 の倍数**で指定します。元画像が大きくても、この解像度に縮小されます。
  ```toml
  [[datasets]]
  resolution = 384
  batch_size = 1
  ```
- 潜在のキャッシュ（`--cache_latents_to_disk`）は、1 つの `.npz` に**潜在サイズごとのキー**（`latents_48x48` など）で追記されます。
  解像度を変えて再実行すると、足りない解像度の分だけが作られ、既存の分は上書きされません。
- 小さい解像度で学習した LoRA は、その解像度に近い画像でもっともよく効きます。生成は 512px 前後でも使えますが、品質は自分で確認してください。

### step 数の目安
**512x512 で 300 step 程度**でも、人物 LoRA は「そこそこ」動きます（9070 XT では約 7 分、RX 550 では上の表のとおり）。
`--max_train_steps` で自由に調整できます。時間が足りないときは、step 数を減らすより先に解像度を下げてください。

### 必須の設定
- **`--blocks_to_swap 24` を付ける**（RX 550 で検証した設定）。`--vram_budget_mb`（前の節）でも同様のスワップを自動で決められます。RX 550 では 3072 MiB で 4 step、RX 640 では 3072 / 2048 / auto で 6〜8 step が完走しました（RX 640 では `--blocks_to_swap 24` より速い）。 Anima の重み（約 4GB）は 4GB の GPU に載らないので、ブロックを CPU に置き、使うときだけ GPU に転送します。
  CUDA 以外では、この指定は自動で通常のスワップ（凍結重みを CPU の原本から転送するだけで、書き戻しはしない方式）になります。
- `--gradient_checkpointing` を併用する（活性化を約 5GB → 0.3GB に減らす）。
- LoRA の `--network_dim` は 16 のまま。これを上げると GPU メモリと時間が増えます。

### VAE は自動で fp32 になる（AVX512_BF16 / AMX の無い CPU）
潜在キャッシュを作る VAE の畳み込みは CPU で動きます。bf16 の畳み込みは、AVX512_BF16 / AMX の無い CPU だと極端に遅い実装に落ちます（12 枚で 30 分以上。fp32 なら約 45 秒）。
そのため、CPU がこれらに対応していない場合は、**`--no_half_vae` 相当を自動で選びます**（`library/device_utils.py` の `prefer_fp32_for_host_compute`）。指定は要りません。
`SD_SCRIPTS_BACKEND=cpu` を指定すると、`vk` を使わず CPU だけで動かせます（切り分け用）。

### GPU が途中で止まる（`ERROR_DEVICE_LOST`）ときは
古い amdgpu ドライバ（Linux）は、約 10 秒を超える GPU 処理を異常とみなして破棄します（Windows の TDR は約 2 秒）。
`torch-vulkan` は 1 回の submit を dispatch 64 個までに制限して回避しています。それでも出る場合は、環境変数 `VKRT_MAX_BATCH`（既定 64）をさらに小さくしてください。

### Linux でのビルド
`torch-vulkan` の `build.py` は Linux に対応しています（g++ と Rust が必要。Rust ランタイムは `libvkrt.so` になり、rpath 付きで同梱されます）。
Python 環境は Windows と同じで、PyTorch の CPU 版（2.8.0+cpu）と `requirements.txt` です。

### 実測した負荷（RX 550、学習中）
GPU 使用率 100%、温度 77〜79°C、電力 25〜32W。CPU 使用率は 0〜3%、RAM は 9〜21GB。
GPU メモリが 96% に迫る構成（512px など）では、特定の step が極端に遅くなる事例を 1 度だけ観測しています（原因は未特定）。余裕を見て、384px 以下を勧めます。

## 融合アテンション（自動）
`F.scaled_dot_product_attention` は、vk では融合カーネル（flash attention 型）で実行されます。QK^T の logits を fp32 のまま保持し、N x N の score 行列をメモリに作りません（forward と backward の両方）。
学習スクリプトの変更は要りません。

- **協調行列が使える GPU（RX 9070 XT など、bf16 / f16）は常に融合版**を使います。従来の bmm + softmax より速く、メモリも少なくなります（RX 9070 XT、bf16、fwd+bwd）:

| 形 | 従来 | 融合 | ピークメモリ |
|---|---|---|---|
| Anima 型（16 heads、S=1024、D=128） | 3.65 ms | 1.55 ms | 236 → 48 MiB |
| SD1.5 型（8 heads、S=4096、D=40） | 92.8 ms | 3.9 ms | 1564 → 32 MiB |

  Anima の実学習（500 step）は 440 秒で完走し、学習した LoRA で人物が再現されることを sd.cpp で確認しました。
- **協調行列の無い GPU（RX 550 など）は、score 行列が大きい（256 MiB 以上。SD1.5・SDXL の 512px など）ときだけ**融合版（スカラー版）を使います。従来より少し遅い（SD1.5 で約 4%）代わりにメモリが大きく減ります（RX 550 の SD1.5: VRAM 最大 3933 → 2650 MB。4GB の 96% に達していたのが下がります）。score 行列が小さいとき（Anima の 512px など）は従来経路です。
- 対応は 4 次元 [B,H,S,D]、D <= 128、f32 / bf16 / f16、マスク・dropout・causal 無しです。それ以外（Anima の LLM アダプタのマスク付き attention、D > 128 など）は従来の経路で動きます。
- **精度**: 実 Anima の attention の入力で比べると、出力と勾配の誤差は従来経路・CPU の bf16 と同水準（0.3〜1%）です。**精度の改善策ではありません**。モデル全体の LoRA 勾配の誤差は、bf16 逆伝播の不安定さで大きく揺れ（相対誤差が入力によって 0.2〜3）、融合版も同じ揺れの範囲です。

## 環境変数
| 変数 | 意味 |
|---|---|
| `SD_SCRIPTS_BACKEND=vk` | 外部バックエンド `vk` を使う（`library/device_utils.py`）。未指定なら従来どおり CUDA/XPU/MPS/CPU |
| `VKT_NO_COOP=1` | 協調行列カーネルを使わない（タイル型 GEMM のみ。速度は落ちるが、勾配の丸め誤差の挙動が CPU に近い） |
| `VKT_FUSED_ATTN=0/1` | 融合アテンション（`F.scaled_dot_product_attention` の vk 実装）を使わない（0）／常に使う（1）。未指定は auto（下の節）。`VKT_NO_FUSED_ATTN=1` は 0 と同じ |
| `VKT_STATS=1` | 終了時に dispatch/転送/フォールバックの時間と回数を表示 |
| `VKT_TRACE_FALLBACK=1` | CPU フォールバックした演算の引数と、vk と CPU の混在を表示（デバッグ用） |
| `VKRT_PROFILE=1` | カーネル別の GPU 時間を計測（1 dispatch ごとに同期するので遅くなる） |
| `VKRT_VALIDATION=1` | Vulkan 検証レイヤを有効にする（開発用） |
| `VKRT_DEVICE=<番号 or 名前の一部>` | 使う Vulkan デバイスを選ぶ（例: `VKRT_DEVICE=Arc`）。未指定なら最初の dGPU。一致しなければ利用可能なデバイス一覧つきのエラー |
| `VKT_MEM_LIMIT_MB=N` | GPU メモリの使用量に上限（MiB）を設け、小さい VRAM の GPU を模擬する（`--vram_budget_mb N` を指定すると自動で設定される） |

## 制限・注意
- **Windows の AMD 公式ドライバ（RX 640 で確認）では、fp16 の値が正しく返らない**（仮数部を失う。原因は未特定）。Anima の学習は bf16 なので影響しませんが、fp16 を使う設定（`--mixed_precision=fp16` など）は使わないでください。
- **GPU で動くのは DiT の学習（fwd/bwd）とオプティマイザ**です。VAE のエンコード・デコード（畳み込み）とテキストエンコーダは CPU で動きます。
  学習中のサンプル画像生成（`--sample_every_n_steps`）は動作しますが、VAE デコードが CPU のため遅くなります。
- bf16 の勾配には、CPU bf16 でも出る程度の丸め誤差があります（前半ブロックの LoRA 勾配は fp32 基準に対して相対 L2 誤差 0.3〜0.5 程度）。
  協調行列カーネルは、タイル型と各呼び出しで一致しており、協調行列に固有の問題は見つかっていません。ただし入力によっては、vk のアテンション
  （ロジットを bf16 に丸めてから softmax する）が原因で勾配が外れ値的にずれることがあります（調査: `torch-vulkan/tools/gradcmp/`）。
  LoRA は正しく学習できており生成も合格していますが、厳密さを優先するなら `VKT_NO_COOP=1` を使ってください（約 1.4 → 2.0 秒/step）。
- 1 回の確保は 2GB 以下（Vulkan の `maxMemoryAllocationSize`）。512px の学習では問題になりません。
- 対象は Anima / Anima-Turbo のみ。SDXL など畳み込み主体のモデルの学習は未対応です（畳み込みの GPU カーネルが無い）。
