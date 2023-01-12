import yaml
import pandas as pd

def preprocess_dataframe(df, mode, configs):
    assert mode in ["train","test"]
    if mode == "train":
        df_train = df.copy()
        empty_title = ((df_train['title2_zh'].isnull()) \
               | (df_train['title1_zh'].isnull()) \
               | (df_train['title2_zh'] == '') \
               | (df_train['title2_zh'] == '0'))
        df_train = df_train[~empty_title]

        MAX_LENGTH = int(configs["Preprocess_config"]["max_length"])
        df_train = df_train[~(df_train.title1_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
        df_train = df_train[~(df_train.title2_zh.apply(lambda x : len(x)) > MAX_LENGTH)]

        SAMPLE_FRAC = float(configs["Preprocess_config"]["frac"])
        df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=9527)

        df_train = df_train.reset_index()
        df_train = df_train.loc[:, ['title1_zh', 'title2_zh', 'label']]
        df_train.columns = ['text_a', 'text_b', 'label']

        SAVE_PATH = str(configs["Preprocess_config"]["preprocess_train_path"])
        df_train.to_csv(SAVE_PATH, sep="\t", index=False)
        return df_train
    elif mode == "test":
        SAVE_PATH = str(configs["Preprocess_config"]["preprocess_test_path"])
        df_test = df.copy()
        df_test = df_test.loc[:, ["title1_zh", "title2_zh", "id"]]
        df_test.columns = ["text_a", "text_b", "Id"]
        df_test.to_csv(SAVE_PATH, sep="\t", index=False)

        df_test = df_test.fillna("")
        return df_test
    
def read_configs(path="./config.yaml"):
    with open(path, 'r') as file:
        try:
            configs = yaml.load(file, Loader=yaml.FullLoader)
        except Exception as e:
            print(f"Cannot read config yaml file, Error: {e}")

    return configs