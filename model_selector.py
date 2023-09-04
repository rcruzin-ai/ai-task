# model_selector.py

class ModelSelector:
    def __init__(self):
        self.model_paths = {
            "twitter-roberta-base-sentiment-latest": {
                "path": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "logo": "logo/cardiffnlp.png"
            },
            "bertweet-base-sentiment-analysis": {
                "path": "finiteautomata/bertweet-base-sentiment-analysis",
                "logo": "logo/finiteautomata.png"
            },
            "twitter-xlm-roberta-base-sentiment": {
                "path": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                "logo": "logo/cardiffnlp.png"
            },
            "twitter-roberta-base-sentiment": {
                "path": "cardiffnlp/twitter-roberta-base-sentiment",
                "logo": "logo/cardiffnlp.png"
            }
        }

    def get_model_path(self, model_name):
        return self.model_paths[model_name]['path']

    def get_model_logo(self, model_name):
        return self.model_paths[model_name]['logo']

    def get_model_names(self):
        return list(self.model_paths.keys())
