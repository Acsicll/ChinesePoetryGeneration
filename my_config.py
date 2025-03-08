import os
import logging
from collections import defaultdict
from dataclasses import dataclass

# to compatible with windows and linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VOCAB_DIR = os.path.join(DATA_DIR, "vocab")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

SENTENCES_PATH = os.path.join(DATA_DIR, "sentences.txt")
CCPC_DATA_PATH = os.path.join(DATA_DIR, "ccpc_data_7.json")
DATA_PATH = os.path.join(DATA_DIR, "song_iambics.txt")
T2S_DATA_PATH = os.path.join(DATA_DIR, "t2s_song_iambics.txt")
CLASSIFIED_DIR_PATH = os.path.join(DATA_DIR, "classified")

SIMPLE_VOCAB_PATH = os.path.join(VOCAB_DIR, "simple_vocab.json")
SIMPLE_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "simple_processed_data.pt")
WORD2VEC_VOCAB_PATH = os.path.join(VOCAB_DIR, "word2vec_vocab.json")
WORD2VEC_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "word2vec_processed_data.pt")

# config value
MAX_LENGTH = 100

# special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TKOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# stopwords
# TODO: add more stopwords
STOP_WORDS = frozenset(
    ["的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
     "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
     "你", "会", "着", "没有", "看", "好", "自己", "这"]
)

# poetry themes
# TODO: add more poetry themes
POETRY_THEMES = defaultdict(set, {
    "自然": {"山", "水", "云", "雨", "江", "河", "海", "村", "漠"},
    "爱情": {"情", "爱", "相思", "恋"},
    "战争": {"战", "军", "兵", "戈", "剑", "塞", "帐"},
    "人生": {"欢", "人生", "人间", "老", "少", "年", "时", "世", "道"},
    "政治": {"国", "政", "朝", "古", "宫", "治", "理"},
})


@dataclass
class HyperParams:
    model_type: str = "LSTM"
    input_dim: int = 5000
    output_dim: int = 5000
    enc_emb_dim: int = 256
    dec_emb_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    enc_dropout: float = 0.5
    dec_dropout: float = 0.5
    batch_size: int = 64
    clip: float = 1
    learning_rate: float = 3e-4


LOG_PATH = os.path.join(BASE_DIR, "logs", "running.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),  # 输出到文件
        logging.StreamHandler()  # 继续打印到终端
    ]
)
