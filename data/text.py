import re
import string
import math


def tokenized_text_normalize(text: str):
    #     text=  re.sub(r'http(\S)+', ' ',text)
    #     text=  re.sub(r'https(\S)+', ' ',text)
    #     text=  re.sub(r'http ...', ' ',text)
    #     text = re.sub(r'@[\S]+',' ',text)
    #     text = text.strip(string.punctuation+" ")
    #     text = re.sub("\*", " ", text)
    #     text = text.strip(string.punctuation+" ")
    text = re.sub("\*", " ", text)
    text = re.sub("#", " ", text)
    text = text.strip(r"# *")
    text = re.sub(" _ ", " ", text)
    text = re.sub("\.+", "\.", text)
    text = re.sub("…", "\.", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"- - -", " ", text)
    text = re.sub("_ ", " ", text)
    text = re.sub(" +", " ", text)
    return text


def extract_num(text):
    if type(text) == str:
        if text == "unknown":
            return math.nan
        return re.findall("\d+", text)[0]
    return text


def vlsp_impute(df, columns=["timestamp_post", "num_like_post", "num_comment_post", "num_share_post"]):

    for col in columns:
        df[col] = df[col].apply(extract_num)
        df[col] = df[col].astype(float)
        df[col] = df[col].fillna(-df[col].mean())

    return df
