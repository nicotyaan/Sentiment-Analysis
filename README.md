#Sentiment Aalysis 

🎬 映画レビュー感情分析（Sentiment Analysis）
このプロジェクトは PyTorch と Hugging Face Transformers を使用して、
映画レビューを 「ポジティブ / ネガティブ」 に分類する感情分析モデルを構築するものです。
事前学習済みモデル BERT をファインチューニングし、
学習・評価・推論 API・Docker による再現環境まで一通り揃えています。


🏋️‍♂️ モデルの学習
1. データ準備
 と  を以下の形式で配置：

2. 学習実行

学習済みモデルは  に保存されます。

📊 モデル評価

精度・F1 スコアなどが出力されます。

🔍 推論（ローカル）


🌐 推論 API（FastAPI）

エンドポイント
• 	


🐳 Docker で実行
1. ビルド

2. 実行


📈 今後の改善案
• 	RoBERTa / DeBERTa など他モデルの比較
• 	日本語データセット対応
• 	モデル蒸留による軽量化
• 	MLOps（MLflow / Weights & Biases）導入

🇺🇸 README (English Version) — Sentiment Analysis with BERT
🎬 Movie Review Sentiment Analysis
This project performs binary sentiment classification (Positive / Negative) on movie reviews using PyTorch and Hugging Face Transformers.
We fine-tune a pre-trained BERT model and provide:
• 	Training script
• 	Evaluation script
• 	Inference script
• 	FastAPI inference server
• 	Docker environment for full reproducibility

📁 Project Structure


🚀 Technologies Used
• 	Python 3.10+
• 	PyTorch
• 	Hugging Face Transformers
• 	FastAPI
• 	Docker
• 	Uvicorn

📦 Setup
1. Create virtual environment


🏋️‍♂️ Training the Model
1. Prepare dataset
Place  and  under :

2. Run training

The trained model will be saved under .

📊 Evaluation

Outputs accuracy, F1 score, etc.

🔍 Inference (Local)


🌐 Inference API (FastAPI)

Endpoint
• 	


🐳 Run with Docker
1. Build

2. Run


📈 Future Improvements
• 	Compare with RoBERTa / DeBERTa
• 	Add multilingual datasets
• 	Model distillation for faster inference
• 	Integrate MLflow or Weights & Biase
