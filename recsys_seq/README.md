# recsys_seq

`recsys_cf` 스타일을 그대로 따라가면서(입력/출력, scores.parquet, submission.csv, config.yaml) 
두 가지 확장 모델을 추가한 폴더입니다.

- **sasrec_content**: SASRec + item side-info(genre/director/writer)를 item embedding에 흡수
- **hybrid**: ALS 후보 생성 + content cosine re-rank (two-stage)

## Data
필요 파일 (동일 폴더):
- train_ratings.csv (user,item,time)
- genres.tsv (item,genre)
- directors.tsv (item,director)
- writers.tsv (item,writer)

## Run
```bash
pip install -r requirements.txt

python run.py --train_path /path/to/train_ratings.csv --meta_dir /path/to/train_dir --model als
python run.py --train_path /path/to/train_ratings.csv --meta_dir /path/to/train_dir --model sasrec_content --device cpu
python run.py --train_path /path/to/train_ratings.csv --meta_dir /path/to/train_dir --model hybrid

python evaluate.py --train_path /path/to/train_ratings.csv --scores_path ./experiments/hybrid/scores.parquet --k 10
```

Outputs are written to `./experiments/<model>/`.
