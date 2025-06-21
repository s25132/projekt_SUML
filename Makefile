install:
	@pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	@black *.py

train:
	@python train.py

eval:
	@echo "## Model Metrics" > report.md
	@cat ./results/classification_report.txt >> report.md

	@cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add .  # dodaje wszystkie nowe i zmodyfikowane pliki
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login: 
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub: 
	huggingface-cli upload spahbod/Titanic ./app --repo-type=space --commit-message="Sync app files"

deploy: hf-login push-hub

all: install format train eval update-branch deploy