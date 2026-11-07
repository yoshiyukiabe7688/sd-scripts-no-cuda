# Anima / Anima-Turbo を CUDA 無しで学習する（Vulkan バックエンド）

CUDA が使えない環境（AMD GPU など）で、Anima / Anima-Turbo の LoRA 学習を **Vulkan** 上で実行する手順です。
PyTorch の標準 Vulkan は推論専用なので、学習用の自作デバイス `vk`（PrivateUse1）を別プロジェクト `torch-vulkan` として用意しています。
学習スクリプト（`anima_train_network.py`）は無改造で、環境変数 `SD_SCRIPTS_BACKEND=vk` を付けるだけで GPU を使います。
CPU のみでも学習できます（環境変数なし。bf16 で約 10 秒/step）。

## 検証済みの環境と結果
- Windows 11 / Ryzen 7 7800X3D / **Radeon RX 9070 XT**（Vulkan 1.4、ドライバ 26.10.1）、PyTorch 2.8.0+cpu。
- Anima-base v1.0 と Anima-Turbo v1.1 の LoRA 学習（512px、batch 1、bf16、rank 16）: **約 1.4 秒/step**（CPU bf16 の約 7 倍）。
- 学習した LoRA を **stable-diffusion.cpp**（master-1053-5a1ba5d、Vulkan ビルド）で読み込んで生成できることを確認済み（840/840 テンソルが適用）。
  オリジナルキャラクター（銀髪ツインテール・赤いリボン・青い目）を 12 枚で学習し、トリガーワード付きの未学習の場面で、LoRA ありはキャラクターが再現され、なしでは再現されない。base と Turbo の両方で確認。

## 必要なもの
- Vulkan 対応 GPU とドライバ。協調行列（`VK_KHR_cooperative_matrix`）があると行列積が速くなる（無ければ自動でタイル型 GEMM に切り替わる）。
- `torch-vulkan` のビルド: Visual Studio 2022（MSVC）、Rust（cargo）、Vulkan SDK（シェーダを再生成する場合のみ。生成済み SPIR-V はコミット済み）。
- `torch-vulkan` は sd-scripts のリポジトリ外にあります（例: `C:\Users\<you>\dev\torch-vulkan`）。

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

## 環境変数
| 変数 | 意味 |
|---|---|
| `SD_SCRIPTS_BACKEND=vk` | 外部バックエンド `vk` を使う（`library/device_utils.py`）。未指定なら従来どおり CUDA/XPU/MPS/CPU |
| `VKT_NO_COOP=1` | 協調行列カーネルを使わない（タイル型 GEMM のみ。速度は落ちるが、勾配の丸め誤差の挙動が CPU に近い） |
| `VKT_STATS=1` | 終了時に dispatch/転送/フォールバックの時間と回数を表示 |
| `VKT_TRACE_FALLBACK=1` | CPU フォールバックした演算の引数と、vk と CPU の混在を表示（デバッグ用） |
| `VKRT_PROFILE=1` | カーネル別の GPU 時間を計測（1 dispatch ごとに同期するので遅くなる） |
| `VKRT_VALIDATION=1` | Vulkan 検証レイヤを有効にする（開発用） |

## 制限・注意
- **GPU で動くのは DiT の学習（fwd/bwd）とオプティマイザ**です。VAE のエンコード・デコード（畳み込み）とテキストエンコーダは CPU で動きます。
  学習中のサンプル画像生成（`--sample_every_n_steps`）は動作しますが、VAE デコードが CPU のため遅くなります。
- 協調行列カーネルは、同じ入力のタイル型と各呼び出しで一致する（出力丸めの範囲）一方、fp32 を基準にした 1 step 後の勾配の誤差が、タイル型よりやや大きい（相対 L2 誤差 0.36 対 0.27。CPU bf16 は 0.28）ことを観測しています。
  原因は特定できていません。bf16 学習としては実用上問題なく LoRA は正しく学習できますが、厳密さを優先するなら `VKT_NO_COOP=1` を使ってください（約 1.4 → 2.0 秒/step）。
- 1 回の確保は 2GB 以下（Vulkan の `maxMemoryAllocationSize`）。512px の学習では問題になりません。
- 対象は Anima / Anima-Turbo のみ。SDXL など畳み込み主体のモデルの学習は未対応です（畳み込みの GPU カーネルが無い）。
