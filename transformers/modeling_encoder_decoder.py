# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classes to support Encoder-Decoder architectures """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os

import torch
from torch import nn

from .modeling_auto import AutoModel, AutoModelWithLMHead
from examples.utils_summarization import fit_to_block_size

logger = logging.getLogger(__name__)


class PreTrainedEncoderDecoder(nn.Module):
    r"""
        :class:`~transformers.PreTrainedEncoderDecoder` is a generic model class that will be
        instantiated as a transformer architecture with one of the base model
        classes of the library as encoder and (optionally) another one as
        decoder when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.
    """

    def __init__(self, encoder, decoder):
        super(PreTrainedEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def from_pretrained(
        cls,
        encoder_pretrained_model_name_or_path=None,
        decoder_pretrained_model_name_or_path=None,
        *model_args,
        **kwargs
    ):
        r""" Instantiates an encoder and a decoder from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you need to first set it back in training mode with `model.train()`

        Params:
            encoder_pretrained_model_name_or_path: information necessary to initiate the encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path: information necessary to initiate the decoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/decoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

                You can specify kwargs sepcific for the encoder and decoder by prefixing the key with `encoder_` and `decoder_` respectively. (e.g. ``decoder_output_attention=True``). The remaining kwargs will be passed to both encoders and decoders.

        Examples::

            model = PreTrainedEncoderDecoder.from_pretained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert
        """

        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as a whole.
        # We let the specific kwargs override the common ones in case of conflict.
        kwargs_common = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("encoder_")
            and not argument.startswith("decoder_")
        }
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder = kwargs_common.copy()
        kwargs_encoder.update(
            {
                argument[len("encoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        kwargs_decoder.update(
            {
                argument[len("decoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("decoder_")
            }
        )

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            encoder = AutoModel.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder
            )
        encoder.config.is_decoder = False

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            decoder = AutoModelWithLMHead.from_pretrained(
                decoder_pretrained_model_name_or_path, **kwargs_decoder
            )
        decoder.config.is_decoder = True

        model = cls(encoder, decoder)

        return model

    def save_pretrained(self, save_directory):
        """ Save a Seq2Seq model and its configuration file in a format such
        that it can be loaded using `:func:`~transformers.PreTrainedEncoderDecoder.from_pretrained`

        We save the encoder' and decoder's parameters in two separate directories.
        """
        if not os.path.exists(os.path.join(save_directory, "encoder")):
            os.makedirs(os.path.join(save_directory, "encoder"))
        if not os.path.exists(os.path.join(save_directory, "decoder")):
            os.makedirs(os.path.join(save_directory, "decoder"))
        self.encoder.save_pretrained(os.path.join(save_directory, "encoder"))
        self.decoder.save_pretrained(os.path.join(save_directory, "decoder"))

    def forward(self, encoder_input_ids, decoder_input_ids, fdebug=None, **kwargs):
        """ The forward pass on a seq2eq depends what we are performing:

        - During training we perform one forward pass through both the encoder
          and decoder;
        - During prediction, we perform one forward pass through the encoder,
          and then perform several forward passes with the encoder's hidden
          state through the decoder to decode a full sequence.

        Therefore, we skip the forward pass on the encoder if an argument named
        `encoder_hidden_state` is passed to this function.

        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.
        kwargs_common = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("encoder_")
            and not argument.startswith("decoder_")
        }
        #print(kwargs_common)
        #fdebug=kwargs_common['fdebug']
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder = kwargs_common.copy()
        kwargs_encoder.update(
            {
                argument[len("encoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        kwargs_decoder.update(
            {
                argument[len("decoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("decoder_")
            }
        )

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        #print('encoder_hidden_states={}'.format(encoder_hidden_states))
        if fdebug is not None:
            torch.set_printoptions(profile="full")
            fdebug.write('{}'.format(encoder_hidden_states)+'\n')

        if encoder_hidden_states is None:
            encoder_outputs = self.encoder(encoder_input_ids, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[
                2
            ]  # output the last layer hidden state
        else:
            encoder_outputs = ()

        # Decode
        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get(
            "attention_mask", None
        )
        decoder_outputs = self.decoder(decoder_input_ids, **kwargs_decoder)

        return decoder_outputs + encoder_outputs

    def decoding(self, encoder_input_ids, decoder_input_ids, fdebug=None, **kwargs):
        """ The forward pass on a seq2eq depends what we are performing:

        - During training we perform one forward pass through both the encoder
          and decoder;
        - During prediction, we perform one forward pass through the encoder,
          and then perform several forward passes with the encoder's hidden
          state through the decoder to decode a full sequence.

        Therefore, we skip the forward pass on the encoder if an argument named
        `encoder_hidden_state` is passed to this function.

        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.
        kwargs_common = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("encoder_")
            and not argument.startswith("decoder_")
        }
        #fdebug=kwargs['fdebug']
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder = kwargs_common.copy()
        kwargs_encoder.update(
            {
                argument[len("encoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        kwargs_decoder.update(
            {
                argument[len("decoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("decoder_")
            }
        )

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        if encoder_hidden_states is None:
            encoder_outputs = self.encoder(encoder_input_ids, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[
                0
            ]  # output the last layer hidden state
        else:
            encoder_outputs = ()

        #print('encoder input ids')
        #print(encoder_input_ids)
        #print('encoder hidden states')
        #print(encoder_hidden_states)
        torch.set_printoptions(profile="full")
        if fdebug is not None:
            fdebug.write('{}'.format(encoder_hidden_states) + '\n')
        # Decode
        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get(
            "attention_mask", None
        )
        #add attention_msk to kwarfs_decoder
        #decoder_input_ids=decoder_input_ids[:,:1]
        decoder_input_shape=decoder_input_ids.size()
        #print('debug decoder_input_ids={}'.format(decoder_input_shape))


        for step in range(10):
            produced_decoder_attn_mask=torch.cat([torch.ones([decoder_input_shape[0],step+1],dtype=torch.int32, device=decoder_input_ids.device)
                                                     ,torch.zeros([decoder_input_shape[0],decoder_input_shape[1]-(step+1)], dtype=torch.int32, device=decoder_input_ids.device)],dim=1)
            #print('produced_decoder_attn_mask = {}'.format(produced_decoder_attn_mask))

            kwargs_decoder["attention_mask"]=produced_decoder_attn_mask
            decoder_outputs = self.decoder(decoder_input_ids, **kwargs_decoder)
            decoder_ids=decoder_outputs[1].argmax(dim=-1)
            decoder_ids=decoder_ids[:,step]
            #print('decoder_input_ids shape = {}'.format(decoder_input_ids.size()))
            #print('decoder_output_ids shape = {}'.format(decoder_ids.size()))
            #print(decoder_input_ids[:,step+1])
            #print('########################################')
            #print(decoder_ids)
            decoder_input_ids[:,step+1]=decoder_ids
            #print('decoder input ids:')
            #print(decoder_input_ids)


        return decoder_input_ids


class Model2Model(PreTrainedEncoderDecoder):
    r"""
        :class:`~transformers.Model2Model` instantiates a Seq2Seq2 model
        where both of the encoder and decoder are of the same family. If the
        name of or that path to a pretrained model is specified the encoder and
        the decoder will be initialized with the pretrained weight (the
        cross-attention will be intialized randomly if its weights are not
        present).

        It is possible to override this behavior and initialize, say, the decoder randomly
        by creating it beforehand as follows

            config = BertConfig.from_pretrained()
            decoder = BertForMaskedLM(config)
            model = Model2Model.from_pretrained('bert-base-uncased', decoder_model=decoder)
    """

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def __init__(self, *args, **kwargs):
        super(Model2Model, self).__init__(*args, **kwargs)
        self.tie_weights()

    def tie_weights(self):
        """ Tying the encoder and decoders' embeddings together.

       We need for each to get down to the embedding weights. However the
        different model classes are inconsistent to that respect:
        - BertModel: embeddings.word_embeddings
        - RoBERTa: embeddings.word_embeddings
        - XLMModel: embeddings
        - GPT2: wte
        - BertForMaskedLM: bert.embeddings.word_embeddings
        - RobertaForMaskedLM: roberta.embeddings.word_embeddings

        argument of the XEmbedding layer for each model, but it is "blocked"
        by a model-specific keyword (bert, )...
        """
        # self._tie_or_clone_weights(self.encoder, self.decoder)
        #self.decoder.bert.embeddings=self.encoder.embeddings
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):

        if (
            "bert" not in pretrained_model_name_or_path
            or "roberta" in pretrained_model_name_or_path
            or "distilbert" in pretrained_model_name_or_path
        ):
            raise ValueError("Only the Bert model is currently supported.")

        model = super(Model2Model, cls).from_pretrained(
            encoder_pretrained_model_name_or_path=pretrained_model_name_or_path,
            decoder_pretrained_model_name_or_path=pretrained_model_name_or_path,
            *args,
            **kwargs
        )

        return model

    def _get_vocab_index_by_decoder_input_ids(self,decoder_input_ids,tokenizer,fsa):
        decoder_ids=decoder_input_ids.cpu().numpy().tolist()
        vocab_indexes_batch=[]
        for i in range(len(decoder_ids)):
            token_ids=decoder_ids[i]
            tokens=[]
            print(token_ids)
            for j in range(len(token_ids)):
                tokens.append(tokenizer.ids_to_tokens.get(token_ids[j], tokenizer.unk_token))
            #print('IN CONVERSION PROCESS: input = {}'.format(tokens))
            fsa_states = fsa.convert_seq_to_states(tokens)
            #print('IN CONVERSION PROCESS: states = {}'.format(fsa_states))
            vocab_indexes = []
            for state in fsa_states:
                vocab_indexes.append(fsa.get_vocab_index(state))
            vocab_indexes=fit_to_block_size(vocab_indexes,len(token_ids), tokenizer.pad_token_id)
            vocab_indexes_batch.append(vocab_indexes)
        vocab_indexes_batch=torch.tensor(vocab_indexes_batch).to(device=decoder_input_ids.device)
        return vocab_indexes_batch



    def decoding(self, encoder_input_ids, decoder_input_ids, tokenizer=None, fsa=None, fdebug=None, **kwargs):
        """ The forward pass on a seq2eq depends what we are performing:

        - During training we perform one forward pass through both the encoder
          and decoder;
        - During prediction, we perform one forward pass through the encoder,
          and then perform several forward passes with the encoder's hidden
          state through the decoder to decode a full sequence.

        Therefore, we skip the forward pass on the encoder if an argument named
        `encoder_hidden_state` is passed to this function.

        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.
        kwargs_common = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("encoder_")
            and not argument.startswith("decoder_")
        }
        #fdebug=kwargs['fdebug']
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder = kwargs_common.copy()
        kwargs_encoder.update(
            {
                argument[len("encoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        kwargs_decoder.update(
            {
                argument[len("decoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("decoder_")
            }
        )

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        if encoder_hidden_states is None:
            encoder_outputs = self.encoder(encoder_input_ids, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[
                2
            ]  # output the last layer hidden state
        else:
            encoder_outputs = ()

        #print('encoder input ids')
        #print(encoder_input_ids)
        #print('encoder hidden states')
        #print(encoder_hidden_states)
        torch.set_printoptions(profile="full")
        if fdebug is not None:
            fdebug.write('{}'.format(encoder_hidden_states) + '\n')
        # Decode
        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get(
            "attention_mask", None
        )
        #add attention_msk to kwarfs_decoder
        #decoder_input_ids=decoder_input_ids[:,:1]
        decoder_input_shape=decoder_input_ids.size()
        #print('debug decoder_input_ids={}'.format(decoder_input_shape))

        vocab_mask_index=None
        for step in range(64):
            produced_decoder_attn_mask=torch.cat([torch.ones([decoder_input_shape[0],step+1],dtype=torch.int32, device=decoder_input_ids.device)
                                                     ,torch.zeros([decoder_input_shape[0],decoder_input_shape[1]-(step+1)], dtype=torch.int32, device=decoder_input_ids.device)],dim=1)
            #print('produced_decoder_attn_mask = {}'.format(produced_decoder_attn_mask))

            decoder_input_ids=decoder_input_ids[:,:step+1]
            produced_decoder_attn_mask=produced_decoder_attn_mask[:,:step+1]
            #print('Debug decoder_input_ids ###########################')
            #print('decoder_input_ids size {}'.format(decoder_input_ids.size()))
            vocab_mask_index=self._get_vocab_index_by_decoder_input_ids(decoder_input_ids,tokenizer,fsa)
            kwargs_decoder["attention_mask"]=produced_decoder_attn_mask
            kwargs_decoder["vocab_mask_index"]=vocab_mask_index
            decoder_outputs = self.decoder(decoder_input_ids, **kwargs_decoder)



            content_decoder_ids=decoder_outputs[0][0].argmax(dim=-1)
            content_decoder_ids=content_decoder_ids[:,step]
            pointer_decoder_ids=decoder_outputs[0][1].argmax(dim=-1)
            pointer_decoder_ids=pointer_decoder_ids[:,step]

            '''
            torch.set_printoptions(profile="full")
            print('BEFORE {}'.format(pointer_decoder_ids))
            
            print('AFTER {}'.format(pointer_decoder_ids))
            torch.set_printoptions(profile="default")
            '''
            pointer_decoder_ids = pointer_decoder_ids + 5

            res_vocab_mask=vocab_mask_index[:,step]
            content_mask=(res_vocab_mask==1)
            pointer_mask=(res_vocab_mask!=1)
            content_decoder_ids=content_decoder_ids.masked_fill(content_mask,0)
            pointer_decoder_ids=pointer_decoder_ids.masked_fill(pointer_mask,0)
            decoder_ids=content_decoder_ids+pointer_decoder_ids
            decoder_ids=decoder_ids.unsqueeze(-1)



            #print('decoder_input_ids shape = {}'.format(decoder_input_ids.size()))
            #print('decoder_output_ids shape = {}'.format(decoder_ids.size()))
            #print(decoder_input_ids[:,step+1])
            #print('########################################')
            #print(decoder_ids)
            #print('decoder_input_ids shape= {}, decoder_ids={}'.format(decoder_input_ids.size(),decoder_ids.size()))
            decoder_input_ids=torch.cat([decoder_input_ids,decoder_ids],dim=1)
            #print('decoder input ids:')
            #print(decoder_input_ids)
            '''
            print('########################decoder_input_ids##############################')
            print('decoder_input_ids size {}'.format(decoder_input_ids.size()))
            torch.set_printoptions(profile="full")
            print('{}'.format(decoder_input_ids))
            torch.set_printoptions(profile="default")
            '''
        #exit(-1)
        return decoder_input_ids,vocab_mask_index


class Model2Models(PreTrainedEncoderDecoder):
    r"""
        :class:`~transformers.Model2Model` instantiates a Seq2Seq2 model
        where both of the encoder and decoder are of the same family. If the
        name of or that path to a pretrained model is specified the encoder and
        the decoder will be initialized with the pretrained weight (the
        cross-attention will be intialized randomly if its weights are not
        present).

        It is possible to override this behavior and initialize, say, the decoder randomly
        by creating it beforehand as follows

            config = BertConfig.from_pretrained()
            decoder = BertForMaskedLM(config)
            model = Model2Model.from_pretrained('bert-base-uncased', decoder_model=decoder)
    """

    def __init__(self, encoder, decoders):
        super(PreTrainedEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
    '''
    def to(self,device):
        self.encoder.to(device)
        for decoder in self.decoders:
            decoder.to(device)
    '''
    @classmethod
    def from_pretrained(
        cls,
        encoder_pretrained_model_name_or_path=None,
        decoder_pretrained_model_name_or_path=None,
        decoder_num=2,
        *model_args,
        **kwargs
    ):
        r""" Instantiates an encoder and a decoder from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you need to first set it back in training mode with `model.train()`

        Params:
            encoder_pretrained_model_name_or_path: information necessary to initiate the encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path: information necessary to initiate the decoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/decoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

                You can specify kwargs sepcific for the encoder and decoder by prefixing the key with `encoder_` and `decoder_` respectively. (e.g. ``decoder_output_attention=True``). The remaining kwargs will be passed to both encoders and decoders.

        Examples::

            model = PreTrainedEncoderDecoder.from_pretained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert
        """

        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as a whole.
        # We let the specific kwargs override the common ones in case of conflict.
        kwargs_common = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("encoder_")
            and not argument.startswith("decoder_")
        }
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder = kwargs_common.copy()
        kwargs_encoder.update(
            {
                argument[len("encoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        kwargs_decoder.update(
            {
                argument[len("decoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("decoder_")
            }
        )

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            encoder = AutoModel.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder
            )
        encoder.config.is_decoder = False

        decoders = kwargs_decoder.pop("model", None)
        if decoders is None:
            decoders=[None]*decoder_num
        for i in range(len(decoders)):

            if decoders[i] is None:
                decoders[i] = AutoModelWithLMHead.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder
                )
            decoders[i].config.is_decoder = True
        model = cls(encoder, decoders)

        return model

    def save_pretrained(self, save_directory):
        """ Save a Seq2Seq model and its configuration file in a format such
        that it can be loaded using `:func:`~transformers.PreTrainedEncoderDecoder.from_pretrained`

        We save the encoder' and decoder's parameters in two separate directories.
        """
        if not os.path.exists(os.path.join(save_directory, "encoder")):
            os.makedirs(os.path.join(save_directory, "encoder"))
        self.encoder.save_pretrained(os.path.join(save_directory, "encoder"))

        for i in range(len(self.decoders)):
            if not os.path.exists(os.path.join(save_directory, "decoder_{}".format(i))):
                os.makedirs(os.path.join(save_directory, "decoder_{}".format(i)))
            self.decoders[i].save_pretrained(os.path.join(save_directory, "decoder_{}".format(i)))


    def forward(self, encoder_input_ids, decoder_input_ids, fdebug=None, **kwargs):
        """ The forward pass on a seq2eq depends what we are performing:

        - During training we perform one forward pass through both the encoder
          and decoder;
        - During prediction, we perform one forward pass through the encoder,
          and then perform several forward passes with the encoder's hidden
          state through the decoder to decode a full sequence.

        Therefore, we skip the forward pass on the encoder if an argument named
        `encoder_hidden_state` is passed to this function.

        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.
        kwargs_common = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("encoder_")
            and not argument.startswith("decoder_")
        }
        #print(kwargs_common)
        #fdebug=kwargs_common['fdebug']
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder = kwargs_common.copy()
        kwargs_encoder.update(
            {
                argument[len("encoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        kwargs_decoder.update(
            {
                argument[len("decoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("decoder_")
            }
        )

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        #print('encoder_hidden_states={}'.format(encoder_hidden_states))
        if fdebug is not None:
            torch.set_printoptions(profile="full")
            fdebug.write('{}'.format(encoder_hidden_states)+'\n')

        if encoder_hidden_states is None:
            encoder_outputs = self.encoder(encoder_input_ids, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[
                0
            ]  # output the last layer hidden state
        else:
            encoder_outputs = ()

        # Decode
        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get(
            "attention_mask", None
        )
        assert len(self.decoders)==len(decoder_input_ids)
        decoder_outputs=[None]*len(self.decoders)
        decoder_attention_masks=kwargs_decoder.pop("attention_mask",None)
        decoder_lm_labels=kwargs_decoder.pop("lm_labels",None)
        for i in range(len(self.decoders)):
            decoder_outputs[i] = self.decoders[i](decoder_input_ids[i], attention_mask=decoder_attention_masks[i],
                                                  lm_labels=decoder_lm_labels[i],**kwargs_decoder)
            print('debug loss[{}]={}'.format(i,decoder_outputs[i][0]))

        return decoder_outputs + [encoder_outputs]

    def decoding(self, encoder_input_ids, decoder_input_ids, decoder, fdebug=None, **kwargs):
        """ The forward pass on a seq2eq depends what we are performing:

        - During training we perform one forward pass through both the encoder
          and decoder;
        - During prediction, we perform one forward pass through the encoder,
          and then perform several forward passes with the encoder's hidden
          state through the decoder to decode a full sequence.

        Therefore, we skip the forward pass on the encoder if an argument named
        `encoder_hidden_state` is passed to this function.

        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.
        kwargs_common = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("encoder_")
            and not argument.startswith("decoder_")
        }
        #fdebug=kwargs['fdebug']
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder = kwargs_common.copy()
        kwargs_encoder.update(
            {
                argument[len("encoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        kwargs_decoder.update(
            {
                argument[len("decoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("decoder_")
            }
        )

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        if encoder_hidden_states is None:
            encoder_outputs = self.encoder(encoder_input_ids, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[
                0
            ]  # output the last layer hidden state
        else:
            encoder_outputs = ()

        #print('encoder input ids')
        #print(encoder_input_ids)
        #print('encoder hidden states')
        #print(encoder_hidden_states)
        torch.set_printoptions(profile="full")
        if fdebug is not None:
            fdebug.write('{}'.format(encoder_hidden_states) + '\n')
        # Decode
        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get(
            "attention_mask", None
        )
        #add attention_msk to kwarfs_decoder
        #decoder_input_ids=decoder_input_ids[:,:1]
        decoder_input_shape=decoder_input_ids.size()
        #print('debug decoder_input_ids={}'.format(decoder_input_shape))


        for step in range(64):
            produced_decoder_attn_mask=torch.cat([torch.ones([decoder_input_shape[0],step+1],dtype=torch.int32, device=decoder_input_ids.device)
                                                     ,torch.zeros([decoder_input_shape[0],decoder_input_shape[1]-(step+1)], dtype=torch.int32, device=decoder_input_ids.device)],dim=1)
            #print('produced_decoder_attn_mask = {}'.format(produced_decoder_attn_mask))

            kwargs_decoder["attention_mask"]=produced_decoder_attn_mask
            decoder_outputs = decoder(decoder_input_ids, **kwargs_decoder)
            decoder_ids=decoder_outputs[1].argmax(dim=-1)
            decoder_ids=decoder_ids[:,step]
            #print('decoder_input_ids shape = {}'.format(decoder_input_ids.size()))
            #print('decoder_output_ids shape = {}'.format(decoder_ids.size()))
            #print(decoder_input_ids[:,step+1])
            #print('########################################')
            #print(decoder_ids)
            decoder_input_ids[:,step+1]=decoder_ids
            #print('decoder input ids:')
            #print(decoder_input_ids)


        return decoder_input_ids

    def eval(self):
        self.encoder.eval()
        for decoder in self.decoders:
            decoder.eval()

class Model2FSADecoder(PreTrainedEncoderDecoder):
    r"""
        :class:`~transformers.Model2Model` instantiates a Seq2Seq2 model
        where both of the encoder and decoder are of the same family. If the
        name of or that path to a pretrained model is specified the encoder and
        the decoder will be initialized with the pretrained weight (the
        cross-attention will be intialized randomly if its weights are not
        present).

        It is possible to override this behavior and initialize, say, the decoder randomly
        by creating it beforehand as follows

            config = BertConfig.from_pretrained()
            decoder = BertForMaskedLM(config)
            model = Model2Model.from_pretrained('bert-base-uncased', decoder_model=decoder)
    """

    def __init__(self, *args, **kwargs):
        super(Model2Model, self).__init__(*args, **kwargs)
        self.fsa=kwargs['fsa']
        self.tie_weights()


    def tie_weights(self):
        """ Tying the encoder and decoders' embeddings together.

       We need for each to get down to the embedding weights. However the
        different model classes are inconsistent to that respect:
        - BertModel: embeddings.word_embeddings
        - RoBERTa: embeddings.word_embeddings
        - XLMModel: embeddings
        - GPT2: wte
        - BertForMaskedLM: bert.embeddings.word_embeddings
        - RobertaForMaskedLM: roberta.embeddings.word_embeddings

        argument of the XEmbedding layer for each model, but it is "blocked"
        by a model-specific keyword (bert, )...
        """
        # self._tie_or_clone_weights(self.encoder, self.decoder)
        self.decoder.bert.embeddings=self.encoder.embeddings

    def decoding(self, encoder_input_ids, decoder_input_ids, fdebug=None, **kwargs):
        """ The forward pass on a seq2eq depends what we are performing:

        - During training we perform one forward pass through both the encoder
          and decoder;
        - During prediction, we perform one forward pass through the encoder,
          and then perform several forward passes with the encoder's hidden
          state through the decoder to decode a full sequence.

        Therefore, we skip the forward pass on the encoder if an argument named
        `encoder_hidden_state` is passed to this function.

        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.
        kwargs_common = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("encoder_")
            and not argument.startswith("decoder_")
        }
        #fdebug=kwargs['fdebug']
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder = kwargs_common.copy()
        kwargs_encoder.update(
            {
                argument[len("encoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        kwargs_decoder.update(
            {
                argument[len("decoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("decoder_")
            }
        )

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        if encoder_hidden_states is None:
            encoder_outputs = self.encoder(encoder_input_ids, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[
                0
            ]  # output the last layer hidden state
        else:
            encoder_outputs = ()

        #print('encoder input ids')
        #print(encoder_input_ids)
        #print('encoder hidden states')
        #print(encoder_hidden_states)
        torch.set_printoptions(profile="full")
        if fdebug is not None:
            fdebug.write('{}'.format(encoder_hidden_states) + '\n')
        # Decode
        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get(
            "attention_mask", None
        )
        #add attention_msk to kwarfs_decoder
        #decoder_input_ids=decoder_input_ids[:,:1]
        decoder_input_shape=decoder_input_ids.size()
        #print('debug decoder_input_ids={}'.format(decoder_input_shape))


        for step in range(10):
            produced_decoder_attn_mask=torch.cat([torch.ones([decoder_input_shape[0],step+1],dtype=torch.int32, device=decoder_input_ids.device)
                                                     ,torch.zeros([decoder_input_shape[0],decoder_input_shape[1]-(step+1)], dtype=torch.int32, device=decoder_input_ids.device)],dim=1)
            #print('produced_decoder_attn_mask = {}'.format(produced_decoder_attn_mask))

            kwargs_decoder["attention_mask"]=produced_decoder_attn_mask
            decoder_outputs = self.decoder(decoder_input_ids, **kwargs_decoder)
            decoder_ids=decoder_outputs[1].argmax(dim=-1)
            decoder_ids=decoder_ids[:,step]
            #print('decoder_input_ids shape = {}'.format(decoder_input_ids.size()))
            #print('decoder_output_ids shape = {}'.format(decoder_ids.size()))
            #print(decoder_input_ids[:,step+1])
            #print('########################################')
            #print(decoder_ids)
            decoder_input_ids[:,step+1]=decoder_ids
            #print('decoder input ids:')
            #print(decoder_input_ids)


        return decoder_input_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):

        if (
            "bert" not in pretrained_model_name_or_path
            or "roberta" in pretrained_model_name_or_path
            or "distilbert" in pretrained_model_name_or_path
        ):
            raise ValueError("Only the Bert model is currently supported.")

        model = super(Model2Model, cls).from_pretrained(
            encoder_pretrained_model_name_or_path=pretrained_model_name_or_path,
            decoder_pretrained_model_name_or_path=pretrained_model_name_or_path,
            *args,
            **kwargs
        )

        return model

class Model2LSTM(PreTrainedEncoderDecoder):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if kwargs.get("decoder_model", None) is None:
            # We will create a randomly initilized LSTM model as decoder
            if "decoder_config" not in kwargs:
                raise ValueError(
                    "To load an LSTM in Encoder-Decoder model, please supply either: "
                    "    - a torch.nn.LSTM model as `decoder_model` parameter (`decoder_model=lstm_model`), or"
                    "    - a dictionary of configuration parameters that will be used to initialize a"
                    "      torch.nn.LSTM model as `decoder_config` keyword argument. "
                    "      E.g. `decoder_config={'input_size': 768, 'hidden_size': 768, 'num_layers': 2}`"
                )
            kwargs["decoder_model"] = torch.nn.LSTM(kwargs.pop("decoder_config"))
        model = super(Model2LSTM, cls).from_pretrained(*args, **kwargs)
        return model
