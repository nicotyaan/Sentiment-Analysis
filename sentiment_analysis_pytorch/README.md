# Sentiment Analysis with PyTorch

PyTorch と Hugging Face Transformers を使った映画レビュー感情分析のサンプルリポジトリ。  
BERT をファインチューニングしてレビューをポジティブ / ネガティブに分類します。学習・評価・推論API（FastAPI）を含み、Dockerで再現可能です。

## 使い方（簡易）
1. 依存関係をインストール  
   `pip install -r requirements.txt`
2. 学習  
   `python src/train.py`
3. 評価  
   `python src/evaluate.py models/best_model.pt`
4. 推論API起動  
   `python src/inference.py` または `uvicorn src.inference:app --host 0.0.0.0 --port 8000`

## アピールポイント
- PyTorch と Hugging Face を用いた実務的な NLP ワークフロー
- 学習・評価・API化までの一貫した実装
- Dockerfile による再現性
