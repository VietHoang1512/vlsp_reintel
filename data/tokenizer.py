import logging
import re

from vncorenlp import VnCoreNLP

# Disable all child loggers of urllib3, e.g. urllib3.connectionpool
logging.getLogger("urllib3").propagate = False


class VnCoreTokenizer:
    def __init__(self, path="vncorenlp/VnCoreNLP-1.1.1.jar"):
        self.rdrsegmenter = VnCoreNLP(path, annotators="wseg", max_heap_size="-Xmx500m")

    def tokenize(self, text: str, return_sentences=False) -> str:
        sentences = self.rdrsegmenter.tokenize(text)
        if return_sentences:
            return [" ".join(sentence) for sentence in sentences]
        # print(sentences)
        output = ""
        for sentence in sentences:
            output += " ".join(sentence) + " "

        return self._strip_white_space(output)

    def _strip_white_space(self, text):
        text = re.sub("\n+", "\n", text).strip()
        text = re.sub(" +", " ", text).strip()
        return text


if __name__ == "__main__":
    tokenizer = VnCoreTokenizer("../vncorenlp/VnCoreNLP-1.1.1.jar")
    print(tokenizer.tokenize("Tôi là Hoàng sinh viên đại học Bách Khoa Hà Nội. Lớp Công nghệ thông tin"))
