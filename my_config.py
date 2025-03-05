# path strings
VOCAB_PATH = "./data/vocab.json"
PROCESSED_PATH = "./data/processed_data.pt"
SENTENCES_PATH = "./data/sentences.txt"
CCPC_DATA_PATH = "./data/ccpc_data_7.json"
DATA_PATH = "./data/song_iambics.txt"
T2S_DATA_PATH = "./data/t2s_song_iambics.txt"
CLASSIFIED_DIR_PATH = "./data/classified/"

# config value
MAX_LENGTH = 100

# special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TKOKENS = [PAD_TOKEN,SOS_TOKEN,EOS_TOKEN,UNK_TOKEN]

# stopwords
STOP_WORDS = set(
    ["的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
     "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
     "你", "会", "着", "没有", "看", "好", "自己", "这"])

# poetry themes
POETRY_THEMES = {
    "自然" : ["山", "水", "云", "雨", "江", "河", " 海", "村", "漠"],
    "爱情" : ["情", "爱", "相思", "恋"],
    "战争" : ["战", "军", "兵", "戈", "剑","塞","帐"],
    "人生" : ["欢","人生 ","人间","老","少","年","时","世","道"],
    "政治" : ["国","政","朝","古","宫","治","理"]
}


class HyperParams:
    input_dim = 5000
    output_dim = 5000
    enc_emb_dim = 256
    dec_emb_dim = 256
    hidden_dim = 1024
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5
    batch_size= 64
    clip = 1