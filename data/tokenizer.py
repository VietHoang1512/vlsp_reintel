from vncorenlp import VnCoreNLP
import logging

# Disable all child loggers of urllib3, e.g. urllib3.connectionpool
logging.getLogger("urllib3").propagate = False


class VnCoreTokenizer():
    def __init__(self, path="vncorenlp/VnCoreNLP-1.1.1.jar"):
        self.rdrsegmenter = VnCoreNLP(path,
                                      annotators="wseg", max_heap_size='-Xmx500m')

    def tokenize(self, text: str) -> str:
        sentences = self.rdrsegmenter.tokenize(text)
        output = ""
        for sentence in sentences:
            output += " ".join(sentence)
        return output

if __name__ == "__main__":
    tokenizer = VnCoreTokenizer("../vncorenlp/VnCoreNLP-1.1.1.jar")
    print(tokenizer.tokenize("Tôi là Hoàng sinh viên đại học Bách Khoa Hà Nội"))