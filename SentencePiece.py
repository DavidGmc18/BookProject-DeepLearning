from transformers import LlamaTokenizerFast
import tempfile
import sentencepiece as spm
import tensorflow as tf
from tensorflow.python.data.ops.map_op import _MapDataset
from tensorflow.python.data.ops.prefetch_op import _PrefetchDataset

def _write_dataset_to_file(dataset, file_path):
    tfd = isinstance(dataset, (_MapDataset, _PrefetchDataset))
    with open(file_path, 'w') as f:
        for sample in dataset.as_numpy_iterator() if tfd else dataset:
            sample = sample.decode("utf-8") if tfd else sample
            f.write(sample + '\n')

def _train_spm(dataset, vocab_size, max_sentence_length=4192):
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        _write_dataset_to_file(dataset, temp_file.name)
        spm.SentencePieceTrainer.train(f"--input={temp_file.name} --model_prefix=tokenizers/temp/spm --vocab_size={vocab_size} --max_sentence_length={max_sentence_length} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=<pad> --unk_piece=<unk> --bos_piece=<bos> --eos_piece=<eos>")

def _convert_spm_to_hf(name, vocab_size):
   SentencePieceTokenizer(vocab_file="tokenizers/temp/spm.model", pad_token="<pad>", fuse_unk=False).save_pretrained(f"tokenizers/{name}-{vocab_size}")

def train_hf_tokenizer(name, dataset, vocab_size, max_sentence_length=4192):
    _train_spm(dataset, vocab_size, max_sentence_length)
    _convert_spm_to_hf(name, vocab_size)


class SentencePieceTokenizer(LlamaTokenizerFast):
    def __init__(self, vocab_file=None, tokenizer_file=None, clean_up_tokenization_spaces=False, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>", 
                 add_bos_token=False, add_eos_token=False, use_default_system_prompt=False, legacy=None, add_prefix_space=None, **kwargs):
        super().__init__(vocab_file, tokenizer_file, clean_up_tokenization_spaces, unk_token, bos_token, eos_token, add_bos_token, add_eos_token, 
                         use_default_system_prompt, legacy, add_prefix_space, **kwargs)
        self.padding_side = "right"
        self.truncation_side = "right"

    @classmethod
    def from_pretrained(cls, name: str, **kwargs) -> "SentencePieceTokenizer":
        return super().from_pretrained(name, **kwargs)

    def __call__(self, text, output_seq_length=None, add_special_tokens=True):
        if add_special_tokens:
            text = tf.strings.join([self.bos_token, text, self.eos_token])

        if len(tf.shape(text)) == 1:
            return tf.py_function(func=lambda x: self._tokenize_batch(x, output_seq_length), inp=[text], Tout=tf.int32)
        else:
            return tf.py_function(func=lambda x: self._tokenize(x, output_seq_length), inp=[text], Tout=tf.int32)

    def _tokenize(self, text, output_seq_length=None):
        return self.encode(text.numpy().decode("utf-8"), padding="max_length" if output_seq_length else True, truncation=bool(output_seq_length), 
                           max_length=output_seq_length, return_tensors="tf", return_attention_mask=False, 
                           return_token_type_ids=False)[0]
 
    def _tokenize_batch(self, text, output_seq_length=None):
        text = [s.decode("utf-8") for s in text.numpy()]
        return self.batch_encode_plus(text, padding="max_length" if output_seq_length else True, truncation=bool(output_seq_length), 
                                      max_length=output_seq_length, return_tensors="tf", return_attention_mask=False, 
                                      return_token_type_ids=False)["input_ids"]