.PHONY: data clean

data: scripts/data_prep.py
	python3 scripts/data_prep.py

train:
	python3 scripts/train_model.py

evaluate:
	python3 scripts/evaluate_model.py

deploy:
	python3 scripts/deploy_model.py
clean:
	rm -f data/processed_iris.csv
