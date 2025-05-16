# Requriments - transformers, tokenizers
# Right now, the Smiles Tokenizer uses an exiesting vocab file from rxnfp that is fairly comprehensive and from the USPTO dataset.
# The vocab may be expanded in the near future

import collections
import re
from typing import List
from logging import getLogger
import torch

logger = getLogger(__name__)
"""
SMI_REGEX_PATTERN: str
    SMILES regex pattern for tokenization. Designed by Schwaller et. al.

References

.. [1]  Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
        ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
        1572-1583 DOI: 10.1021/acscentsci.9b00576

"""

# SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|
# #|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]+)"""

SMI_REGEX_PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]+)"



class MOFTokenizerGPT(object):
    """
        Creates the SmilesTokenizer class for autoregressive models.
    """
    def __init__(self,
                 vocab_file: str = '',
                 pad_token='[PAD]',
                 mask_token='[MASK]',
                 bos_token='[BOS]',
                 eos_token='[EOS]',
                 unk_token='[UNK]',
                 max_len=512,
                 add_special_tokens=True,
                 truncation=True,
                 use_topology=True):
      self.max_len = max_len
      self.vocab = load_vocab(vocab_file)
      self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
      self.basic_tokenizer = BasicSmilesTokenizer(regex_pattern=SMI_REGEX_PATTERN)
      self.topo_tokenizer = TopoTokenizer()
      self.pad_token = pad_token 
      self.mask_token = mask_token
      self.bos_token = bos_token
      self.eos_token = eos_token
      self.unk_token = unk_token
      self.add_special_tokens = add_special_tokens
      self.truncation = truncation
      self.use_topology = use_topology

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def vocab_list(self):
        return list(self.vocab.keys())
    
    @property
    def pad_token_id(self):
       return self.vocab[self.pad_token]
    
    @property
    def mask_token_id(self):
        return self.vocab[self.mask_token]
    
    @property
    def bos_token_id(self):
        return self.vocab[self.bos_token]
    
    @property
    def eos_token_id(self):
        return self.vocab[self.eos_token]

    def get_vocab(self):
        return dict(self.vocab)

    def tokenize(self, text):
        """
          Tokenize str into list of tokens
        Parameters
        ----------
        inputs:
            text: str, Input string to tokenize
        Returns:
            List of tokens
        """
        smiles, topo = text.split('&&')
        smiles_tokens = [token for token in self.basic_tokenizer.tokenize(smiles)]
        if not self.use_topology:
          return smiles_tokens
        topo_tokens = self.topo_tokenizer.tokenize(topo)
        split_tokens = smiles_tokens + ['&&'] + topo_tokens
        return split_tokens
    
    def tokenize_smiles(self, text):
        """
          Tokenize str into list of tokens (only SMILES)
        Parameters
        ----------
        inputs:
            text: str, Input string to tokenize
        Returns:
            List of tokens
        """
        smiles_tokens = [token for token in self.basic_tokenizer.tokenize(text)]
        return smiles_tokens

    def convert_token_to_id(self, token):
        """
        Converts a token (str) in an id using the vocab.
        Parameters
        ----------
        token: str
            Token to convert to id.
        Returns:
            int: id of the token in the vocab.
        """
        return self.vocab.get(token, 
                              self.vocab.get(self.unk_token))
    
    def convert_id_to_token(self, index):
        """
        Converts an index (integer) into a token (str) using the vocab.
        Parameters
        ----------
        index: int
            Index to convert to token.
        Returns:
            str: token corresponding to the index in the vocab.
        """
        return self.ids_to_tokens.get(index, 
                                      self.unk_token)
    
    def convert_tokens_to_ids(self, tokens: List[str]):
        """
        Converts a token (str) in an id using the vocab.
        Parameters
        ----------
        tokens: List[str]
            Tokens to convert to ids.
        Returns:
            List[int]: ids of the tokens in the vocab.
        """
        ids = []
        for token in tokens:
            ids.append(self.convert_token_to_id(token))
        return ids
    
    def convert_ids_to_tokens(self, ids: List[int]):
        """
        Converts an id to str using the vocab.
        Parameters
        ----------
        ids: List[int]
            Ids to convert to tokens.
        Returns:
            List[str]: tokens corresponding to the ids in the vocab.
        """
        tokens = []
        for index in ids:
            tokens.append(self.convert_id_to_token(index))
        return tokens
    
    def convert_tokens_to_string(self, tokens: List[str]):
        """
        Converts a sequence of tokens (string) in a single string.
        Parameters
        ----------
        tokens: List[str]
            Tokens to convert to string.
        Returns:
            str: Converted string from tokens.
        """
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string
    
    def build_input_with_special_tokens(self, tokens):
        """
        Build model inputs from a sequence by appending eos_token_id 
        Parameters
        ----------
        tokens: List[int]
            List of input tokens.
        Returns:
            List[int]: List of input tokens with EOS token appended.
        """
        return tokens + [self.eos_token_id] 

    def pad_batched_tokens(self, 
                           batched_token_ids, 
                           batched_masks,
                           batched_target_token_ids,
                           padding=True,
                           ignore_index=-100):
        """
        Pad batch of input tokens with pad_token_id
        Parameters
        ----------
        batched_token_ids: List[List[int]]
            Batch of input tokens.
        batched_masks: List[List[int]]  
            Batch of input masks.
        Returns:
            List[List[int]]: Batch of padded input tokens.
            List[List[int]]: Batch of padded input masks. 
        """
        if padding:
            batch_max_len = max([ids.shape[0] for ids in batched_token_ids])
            padded_ids = torch.zeros((0, batch_max_len), dtype=torch.long)
            padded_masks = torch.zeros((0, batch_max_len), dtype=torch.long)
            padded_target_ids = torch.zeros((0, batch_max_len), dtype=torch.long)
            for token_ids, masks, target_token_ids in zip(batched_token_ids, 
                                                          batched_masks,
                                                          batched_target_token_ids):
                curr_padded_ids = torch.cat((token_ids,
                                             self.vocab[self.pad_token] * torch.ones(batch_max_len - token_ids.shape[0],
                                                                                     dtype=torch.long)))
                padded_ids = torch.cat((padded_ids, curr_padded_ids.unsqueeze(0)), dim=0)
                curr_padded_masks = torch.cat((masks,
                                               torch.zeros(batch_max_len - masks.shape[0],
                                                           dtype=torch.long)))
                padded_masks = torch.cat((padded_masks, curr_padded_masks.unsqueeze(0)), dim=0)
                curr_padded_target_ids = torch.cat((target_token_ids,
                                                    ignore_index * torch.ones(batch_max_len - target_token_ids.shape[0],
                                                                              dtype=torch.long)))
                padded_target_ids = torch.cat((padded_target_ids, curr_padded_target_ids.unsqueeze(0)), dim=0)

            return padded_ids, padded_masks, padded_target_ids
        else:
            return batched_token_ids, batched_masks
        
    def encode(self, 
               text):
      """
      Converts a string in a sequence of ids (integer), using the tokenizer and vocabulary.
      Same as doing ``self.convert_tokens_to_ids(self.tokenize(text))``.
      Parameters
      ----------
      text: 
        str Text to encode.
      add_special_tokens: 
        bool, whether or not to add bos and eos tokens.
      
      Returns
      -------
      List[int]: List of ids (integer) corresponding to the tokenized text.
      """
      tokens = self.tokenize(text)
      if self.truncation:
         if self.add_special_tokens:
            if len(tokens) > self.max_len-2:
                tokens = tokens[:self.max_len-2]
            tokens = [self.bos_token] + tokens + [self.eos_token]
         else:
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]

      # if self.truncation and len(tokens) > self.max_len:
      #    if self.add_special_tokens:
      #       tokens = tokens[:self.max_len-2] # -2 for bos and eos
      #    else:
      #       tokens = tokens[:self.max_len]
      
      # if self.add_special_tokens:
      #     tokens = [self.bos_token] + tokens + [self.eos_token]
      if len(tokens) > self.max_len:
          print(f"Token length: {len(tokens)}")
          raise ValueError("Too many tokens in input sequence. The maximum sequence length is {}".format(self.max_len))
      return self.convert_tokens_to_ids(tokens)
           

class BasicSmilesTokenizer(object):
  """

    Run basic SMILES tokenization using a regex pattern developed by Schwaller et. al. This tokenizer is to be used
    when a tokenizer that does not require the transformers library by HuggingFace is required.

    Examples
    --------
    >>> from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
    >>> tokenizer = BasicSmilesTokenizer()
    >>> print(tokenizer.tokenize("CC(=O)OC1=CC=CC=C1C(=O)O"))
    ['C', 'C', '(', '=', 'O', ')', 'O', 'C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1', 'C', '(', '=', 'O', ')', 'O']


    References
    ----------
    .. [1]  Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
            ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
            1572-1583 DOI: 10.1021/acscentsci.9b00576

    """

  def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
    """ Constructs a BasicSMILESTokenizer.
        Parameters
        ----------

        regex: string
            SMILES token regex

        """
    self.regex_pattern = regex_pattern
    self.regex = re.compile(self.regex_pattern)

  def tokenize(self, text):
    """ Basic Tokenization of a SMILES.
        """
    tokens = [token for token in self.regex.findall(text)]
    return tokens


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  with open(vocab_file, "r", encoding="utf-8") as reader:
    tokens = reader.readlines()
  for index, token in enumerate(tokens):
    token = token.rstrip("\n")
    vocab[token] = index
  return vocab

class TopoTokenizer(object):
  """

  Run basic SMILES tokenization using a regex pattern developed by Schwaller et. al. This tokenizer is to be used
  when a tokenizer that does not require the transformers library by HuggingFace is required.

  Examples
  --------
  >>> from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
  >>> tokenizer = BasicSmilesTokenizer()
  >>> print(tokenizer.tokenize("CC(=O)OC1=CC=CC=C1C(=O)O"))
  ['C', 'C', '(', '=', 'O', ')', 'O', 'C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1', 'C', '(', '=', 'O', ')', 'O']


  References
  ----------
  .. [1]  Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
          ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
          1572-1583 DOI: 10.1021/acscentsci.9b00576

  """

  def __init__(self):
    return

  def tokenize(self, text):
    """ Basic Tokenization of a SMILES.
        """
    topo_cat = text.split('.')
    if len(topo_cat)<2:
      topos = topo_cat[0]
      topos = topos.split(',')
      tokens = topos
    else:
      topos, cat = topo_cat[0], topo_cat[1]
      topos = topos.split(',')
      tokens = topos + [cat]
    return tokens


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  with open(vocab_file, "r", encoding="utf-8") as reader:
    tokens = reader.readlines()
  for index, token in enumerate(tokens):
    token = token.rstrip("\n")
    vocab[token] = index
  return vocab