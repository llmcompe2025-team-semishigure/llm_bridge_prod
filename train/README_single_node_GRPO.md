# GRPO (Group Relative Policy Optimization) for GSM8K - 日本語版

## 概要

この実装は、公式VERLフレームワークを使用してGSM8K数学推論タスク向けのシングルノードおよびマルチノードGRPOトレーニングを提供します。GRPO（Group Relative Policy Optimization）は、main_ppoトレーナーで`algorithm.adv_estimator=grpo`を設定することで実装されます。

## シングルノード使用方法

### 前提条件

1. `README_install_uv.md`に記載されている通りuv仮想環境を設定
2. GSM8KデータとLlama-3.2-1B-Instructモデルが`/home/Competition2025/P04/shareP04/`に配置済み
3. wandbクレデンシャルを設定
4. スクリプト実行前に`.venv`が有効化されていることを確認

### シングルノードGRPOの実行

```bash
# uv環境を最初に有効化
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

# moduleのロード
module load cuda/12.4
module load nccl/2.22.3
module load cudnn/9.6.0
source .venv/bin/activate

# スクリプトを実行
ulimit -v unlimited
unset ROCR_VISIBLE_DEVICES

cd ~/train/scripts/singlenode_grpo
python single_node_grpo.py

```