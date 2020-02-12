# coding=utf-8
# Copyright 2019 The HuggingFace Inc. team.
# Copyright (c) 2019 The HuggingFace Inc.  All rights reserved.
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
""" Finetuning seq2seq models for sequence generation."""

import argparse
import functools
import logging
import os
import random
import sys
import uuid
sys.path.append(r'../')
os.environ["CUDA_VISIBLE_DEVICES"]="5"

import numpy as np
import json
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    BertConfig,
    PreTrainedEncoderDecoder,
    Model2Models,
    Model2Model,
    BertForMaskedLMVocabMask,
)
from finite_state_automata import FiniteStateAutomata
from utils_summarization import (
    CNNDailyMailDataset,
    encode_for_summarization,
    fit_to_block_size,
    build_lm_labels,
    build_mask,
    compute_token_type_ids,
)

from utils_chemistry import (ChemistryDataset,ChemistryProcessor,)
'''
class InputExample(object):
    def __init__(self,example_id,question_input,question_varible_output=None,condition_output=None):
        self.example_id=example_id
        self.question_input=question_input
        self.question_varible_output=question_varible_output
        self.condition_output=condition_output
'''


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# ------------
# Load dataset
# ------------


def load_and_cache_examples(args, tokenizer, prefix="train",fsa=None):
    dataset = ChemistryDataset(tokenizer, prefix=prefix, data_dir=args.data_dir,version=args.decoder_version,fsa_or_config=fsa)
    return dataset

def translate_tokenindex_to_subtokenindex(example,indexes,vocabs,states,clsoffset=1):
    #print('INDEXES: {}'.format(indexes))
    #print('VOCABS: {}'.format(vocabs))
    #print('STATES: {}'.format(states))
    new_indexes=[]
    for i,index in enumerate(indexes):
        if i>0 and vocabs[i-1] == 1:
            index=int(index)+clsoffset
            sub_index=example.orig_to_tok_index[index]
            if states[i].endswith('_end'):
                tmp_sub_index=sub_index
                j=sub_index+1
                while j < len(example.tok_to_orig_index) and example.tok_to_orig_index[j] == index:
                    sub_index=j
                    j+=1
                #print('Diff sub index by end: {} \t {}'.format(tmp_sub_index,sub_index))
            new_indexes.append(str(sub_index))
        else:
            new_indexes.append(index)
    return new_indexes

def translate_subtokenindex_backto_tokenindex(example,indexes,vocabs,clsoffset=1):
    new_indexes=[]
    for i,index in enumerate(indexes):
        if i>0 and vocabs[i-1] == 1:
            index=int(index)
            whole_index=example.tok_to_orig_index[index]-clsoffset

            new_indexes.append(str(whole_index))
        else:
            new_indexes.append(index)
    return new_indexes


def collate(data, encoder_tokenizer,decoder_tokenizer, input_block_size,output_block_size):
    """ List of tuple as an input. """
    inputs=[]
    outputs=[]
    vocabs=[]
    example_buffer=[]
    for i,example in enumerate(data):
        #input=encoder_tokenizer.encode(example.input)
        example_buffer.append(example)
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        input_tokens=['[CLS]']+example.input.split()+['SEP']
        for (i, token) in enumerate(input_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = encoder_tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        input=encoder_tokenizer.convert_tokens_to_ids(all_doc_tokens)
        example.tok_to_orig_index=tok_to_orig_index
        example.orig_to_tok_index=orig_to_tok_index
        input=fit_to_block_size(input, input_block_size, encoder_tokenizer.pad_token_id)
        inputs.append(input)

        if example.output is not None:
            #output=tokenizer.encode(example.output)
            output_tokens=example.output.split()
            #print('Before Whole Index: {}'.format(output_tokens))
            #print('encoder input: {}'.format(all_doc_tokens))
            output_tokens=translate_tokenindex_to_subtokenindex(example,output_tokens,example.vocab_indexes,example.fsa_states)
            #print('After Sub Index: {}'.format(output_tokens))
            output=decoder_tokenizer.convert_tokens_to_ids(output_tokens)
            output_states=example.fsa_states

        else:
            #output=decoder_tokenizer.build_inputs_with_special_tokens(['start'])
            output = decoder_tokenizer.convert_tokens_to_ids(['start'])

        output_vocab_indexes = example.vocab_indexes


        output=fit_to_block_size(output, output_block_size, decoder_tokenizer.pad_token_id)
        output_vocab_indexes=fit_to_block_size(output_vocab_indexes, output_block_size, decoder_tokenizer.pad_token_id)
        outputs.append(output)
        vocabs.append(output_vocab_indexes)
        #print('debug output={}'.format(example.output.split()))
        #print('debug output_states={}'.format(output_states))
        #print('debug output_vocab_indexes={}'.format(output_vocab_indexes))
        #print('debug outputid={}'.format(output))


        #if example.vocab_indexes is not None:
        #    vocab=example.vocab_indexes
        #else:
        #    vocab=[1]
        #vocabs.append(vocab)

    #print(tokenizer.vocab)
    #exit(-1)




    inputs = torch.tensor(inputs)
    outputs = torch.tensor(outputs)
    vocabs = torch.tensor(vocabs)

    inputs_mask = build_mask(inputs, encoder_tokenizer.pad_token_id)
    outputs_mask = build_mask(outputs, decoder_tokenizer.pad_token_id)
    vocabs_mask = build_mask(vocabs, decoder_tokenizer.pad_token_id)

    outputs_mask_lm_labels = build_lm_labels(outputs, decoder_tokenizer.pad_token_id)
    vocabs_mask_lm_labels = build_lm_labels(vocabs, decoder_tokenizer.pad_token_id)

    return (
        inputs,
        outputs,
        vocabs,
        inputs_mask,
        outputs_mask,
        vocabs_mask,
        outputs_mask_lm_labels,
        vocabs_mask_lm_labels,
        example_buffer,
    )




# ----------
# Optimizers
# ----------


class BertSumOptimizer(object):
    """ Specific optimizer for BertSum.

    As described in [1], the authors fine-tune BertSum for abstractive
    summarization using two Adam Optimizers with different warm-up steps and
    learning rate. They also use a custom learning rate scheduler.

    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    """

    def __init__(self, model, lr, warmup_steps, beta_1=0.99, beta_2=0.999, eps=1e-8):
        self.encoder = model.encoder
        self.decoders = model.decoders
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.decoders_parameters=[]
        for decoder in model.decoders:
            self.decoders_parameters+=decoder.parameters()

        self.optimizers = {
            "encoder": Adam(
                model.encoder.parameters(),
                lr=lr["encoder"],
                betas=(beta_1, beta_2),
                eps=eps,
            ),
            "decoder": Adam(
                self.decoders_parameters,
                lr=lr["decoder"],
                betas=(beta_1, beta_2),
                eps=eps,
            ),
        }

        self._step = 0

    def _update_rate(self, stack):
        return self.lr[stack] * min(
            self._step ** (-0.5), self._step * self.warmup_steps[stack] ** (-0.5)
        )

    def zero_grad(self):
        self.optimizer_decoder.zero_grad()
        self.optimizer_encoder.zero_grad()

    def step(self):
        self._step += 1
        for stack, optimizer in self.optimizers.items():
            new_rate = self._update_rate(stack)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_rate
            optimizer.step()


# ------------
# Train
# ------------


def train(args, model, tokenizer):
    """ Fine-tune the pretrained model on the corpus. """
    set_seed(args)

    # Load the data
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples(args, tokenizer, "train")
    train_sampler = RandomSampler(train_dataset)
    model_collate_fn = functools.partial(collate, tokenizer=tokenizer,
                                         input_block_size=args.input_block_size,output_block_size=args.output_block_size)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=model_collate_fn,
    )

    # Training schedule
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = t_total // (
            len(train_dataloader) // args.gradient_accumulation_steps + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare the optimizer
    #lr = {"encoder": 0.002, "decoder": 0.2}
    lr = {"encoder": args.encoder_lr, "decoder": args.decoder_lr}
    #warmup_steps = {"encoder": 20000, "decoder": 10000}
    warmup_steps = {"encoder": args.encoder_warmup, "decoder": args.decoder_warmup}
    optimizer = BertSumOptimizer(model, lr, warmup_steps)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps
        # * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch", disable=False)

    global_step = 0
    tr_loss = 0.0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            source, target, encoder_mask, decoder_mask, lm_labels = batch
            #print('source: {}'.format(source))
            #print('target: {}'.format(target))

            feed_source=None
            feed_targets=[None]*len(target)
            feed_encoder_mask=None
            feed_decoder_masks=[None]*len(decoder_mask)
            feed_lm_labels=[None]*len(lm_labels)

            feed_source = source.to(args.device)
            for i in range(len(target)):
                feed_targets[i] = target[i].to(args.device)


            feed_encoder_mask = encoder_mask.to(args.device)
            for i in range(len(decoder_mask)):
                feed_decoder_masks[i] = decoder_mask[i].to(args.device)
            for i in range(len(lm_labels)):
                feed_lm_labels[i] = lm_labels[i].to(args.device)

            model.train()
            #print('debug by zhuoyu: source = {}'.format(source))
            #print('debug by zhuoyu: target = {}'.format(target))
            #print('debug by zhuoyu, device:')
            #print('feed source {}'.format(feed_source.device))
            #print('feed target {}'.format([str(feed_target.device) for feed_target in feed_targets]))
            #print('feed encoder mask {}'.format(feed_encoder_mask.device))
            #print('feed decoder masks {}'.format([str(feed_decoder_mask.device) for feed_decoder_mask in feed_decoder_masks]))
            #print('feed lm labels {}'.format([str(feed_lm_label.device) for feed_lm_label in feed_lm_labels]))
            outputs = model(
                feed_source,
                feed_targets,
                encoder_attention_mask=feed_encoder_mask,
                decoder_attention_mask=feed_decoder_masks,
                decoder_lm_labels=feed_lm_labels,
            )

            loss=0
            for i in range(len(model.decoders)):
                #print('outputs[{}][0] type: {}'.format(i,type(outputs[i][0])))
                loss += outputs[i][0]
            #print(loss)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


# ------------
# Train
# ------------


def evaluate(args, model, tokenizer, prefix=""):
    set_seed(args)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset = load_and_cache_examples(args, tokenizer, prefix="dev")
    #for example in eval_dataset.examples:
    #    print(example.example_id)
    #    print(example.question_input)
    #    print(example.question_varible_output)
    #    print(example.condition_output)
    #exit(-1)
    eval_sampler = SequentialSampler(eval_dataset)
    model_collate_fn = functools.partial(collate, tokenizer=tokenizer,
                                         input_block_size=args.input_block_size,output_block_size=args.output_block_size)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,collate_fn=model_collate_fn,
    )

    # multi-gpu evaluate
    #if args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    fout=open(os.path.join(args.output_dir,"dev.res"),'w',encoding='utf-8')
    fdebug=open(os.path.join(args.output_dir,"dev.debug.res"),'w',encoding='utf-8')
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        source, target, encoder_mask, decoder_mask, lm_labels = batch
        #print('[SOURCE]: {}'.format(source))
        #print('[TARGET]: {}'.format(target))

        #source = source.to(args.device)
        #target = target.to(args.device)

        #encoder_mask = encoder_mask.to(args.device)
        #decoder_mask = decoder_mask.to(args.device)
        #lm_labels = lm_labels.to(args.device)

        feed_source = None
        feed_targets = [None] * len(target)
        feed_encoder_mask = None
        feed_decoder_masks = [None] * len(decoder_mask)
        feed_lm_labels = [None] * len(lm_labels)

        feed_source = source.to(args.device)
        for i in range(len(target)):
            feed_targets[i] = target[i].to(args.device)

        feed_encoder_mask = encoder_mask.to(args.device)
        for i in range(len(decoder_mask)):
            feed_decoder_masks[i] = decoder_mask[i].to(args.device)
        for i in range(len(lm_labels)):
            feed_lm_labels[i] = lm_labels[i].to(args.device)

        with torch.no_grad():

            if args.decoding_type=='decoding':
                tokens_roles=[]
                for i in range(len(feed_targets)):
                    outputs_ids=model.decoding(
                        feed_source,
                        feed_targets[i],
                        encoder_attention_mask=feed_encoder_mask,
                        decoder_attention_mask=feed_decoder_masks[i],
                        decoder_lm_labels=feed_lm_labels[i],
                        decoder=model.decoders[i]
                        #fdebug=fdebug,
                    )
                    print('outputs size: {}'.format(outputs_ids.size()))
                    outputs_ids =outputs_ids.cpu().numpy()


                    batch_tokens=[]
                    for idx in outputs_ids:
                        tokens = []
                        for id in idx:
                            #print('{}\t{}'.format(id,type(id)))
                            tokens.append(tokenizer.ids_to_tokens.get(int(id), tokenizer.unk_token))

                        batch_tokens.append(tokens)

                    tokens_roles.append(batch_tokens)

                def subtoken2token(subtokens):
                    token=""
                    tokens=[]
                    for subtoken in subtokens:
                        if subtoken.startswith("##"):
                            token+=subtoken[2:]
                        else:
                            if token!="":
                                tokens.append(token)
                            token=subtoken
                    if token!="":
                        tokens.append(token)
                    return tokens
                for i in range(len(tokens_roles[0])):
                    fout.write('\t'.join([' '.join(subtoken2token(tokens_roles[0][i]))
                                             ,' '.join(subtoken2token(tokens_roles[1][i]))]) + '\n')

            else:
                print('debug eva input:')
                print('feed_source={}'.format(feed_source))
                print('feed_targets={}'.format(feed_targets))
                print('feed_encoder_mask={}'.format(feed_encoder_mask))
                print('feed_decoder_masks={}'.format(feed_decoder_masks))
                print('feed_lm_labels={}'.format(feed_lm_labels))
                outputs = model(
                    feed_source,
                    feed_targets,
                    encoder_attention_mask=feed_encoder_mask,
                    decoder_attention_mask=feed_decoder_masks,
                    decoder_lm_labels=feed_lm_labels,
                    #fdebug=fdebug,
                )

                ans_seqs=[[],[]]
                for i in range(len(model.decoders)):
                    print(outputs[i][1].size())
                    predicted_scores=outputs[i][1].argmax(-1).cpu().numpy().tolist()
                    for idx in predicted_scores:
                        tokens = []
                        for id in idx:
                            tokens.append(tokenizer.ids_to_tokens.get(id, tokenizer.unk_token))
                        ans_seqs[i].append(tokens)

                for i in range(len(ans_seqs[0])):
                    fout.write('\t'.join([' '.join(ans_seqs[0][i]),' '.join(ans_seqs[1][i])]) + '\n')





                # print('debug by zhuoyu, predicted_scores size={}'.format(predicted_scores.size()))
                #eval_loss += lm_loss.mean().item()




        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    # Save the evaluation's results
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    #with open(os.path.join(args.output_dir,"dev.res"),'w',encoding='utf-8') as fout:
    fout.flush()
    fout.close()
    fdebug.flush()
    fdebug.close()
    return result

def init():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Optional parameters
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--do_evaluate",
        type=bool,
        default=True,
        help="Run model evaluation on out-of-sample data.",
    )
    parser.add_argument("--do_train", type=bool, default=False, help="Run training.")
    parser.add_argument(
        "--do_overwrite_output_dir",
        type=bool,
        default=False,
        help="Whether to overwrite the output dir.",
    )
    parser.add_argument(
        "--encoder_model_name_or_path",
        default="/data/zhuoyu/semantic_parsing_v3/serving/2_bert_output/encoder",
        type=str,
        help="The model checkpoint to initialize the encoder's weights with.",
    )
    parser.add_argument(
        "--decoder_model_name_or_path",
        default="/data/zhuoyu/semantic_parsing_v3/serving/2_bert_output/decoder",
        type=str,
        help="The model checkpoint to initialize the decoder's weights with.",
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="The decoder architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--to_cpu", default=False, type=bool, help="Whether to force training on CPU."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for eval.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--input_block_size",
        default=256,
        type=int,
        help="Max seq length for input",
    )
    parser.add_argument(
        "--output_block_size",
        default=64,
        type=int,
        help="Max seq length for output",
    )

    parser.add_argument(
        "--trained_checkpoints",
        default="/data/zhuoyu/semantic_parsing_v3/serving/2_bert_output",
        type=str,
        help="trained_checkpoints",
    )

    parser.add_argument(
        "--decoding_type",
        default="decoding",
        type=str,
        help="",
    )

    parser.add_argument(
        "--encoder_lr",
        default=5e-4,
        type=float,
        help="encoder's learning rate",
    )

    parser.add_argument(
        "--decoder_lr",
        default=5e-4,
        type=float,
        help="encoder's learning rate",
    )

    parser.add_argument(
        "--encoder_warmup",
        default=10,
        type=int,
        help="encoder's learning rate",
    )

    parser.add_argument(
        "--decoder_warmup",
        default=100,
        type=int,
        help="encoder's learning rate",
    )

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--decoder_version",
        default="v2",
        type=str,
        help="",
    )

    parser.add_argument(
        "--fsa_config",
        default="chemistry_fsa.json",
        type=str,
        help="",
    )

    args = parser.parse_args()



    # Set up training device
    if args.to_cpu or not torch.cuda.is_available():
        args.device = torch.device("cpu")
        args.n_gpu = 0
    else:
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        print(args.n_gpu)

    # Load pretrained model and tokenizer. The decoder's weights are randomly initialized.
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name_or_path)
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_name_or_path)
    tokenizers=[encoder_tokenizer,decoder_tokenizer]
    # config = BertConfig.from_pretrained(args.model_name_or_path)
    # config.num_hidden_layers=3
    # config.is_decoder=True
    # decoder_model = BertForMaskedLM(config)

    decoder_model = BertForMaskedLMVocabMask.from_pretrained(args.decoder_model_name_or_path)
    # print(decoder_model)
    # exit(-1)
    model = Model2Model.from_pretrained(
        args.encoder_model_name_or_path, decoder_model=decoder_model
    )
    # model = Model2Model.from_pretrained(
    #    args.model_name_or_path, decoder_model=None
    # )

    fsa = FiniteStateAutomata.init_from_config(os.path.join(args.decoder_model_name_or_path, args.fsa_config))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        0,
        args.device,
        args.n_gpu,
        False,
        False,
    )

    logger.info("Training/evaluation parameters %s", args)


    checkpoint=args.trained_checkpoints

    encoder_checkpoint = os.path.join(checkpoint, "encoder")
    decoder_checkpoint = os.path.join(checkpoint, "decoder")

    decoder_model = BertForMaskedLMVocabMask.from_pretrained(decoder_checkpoint)
    model = Model2Model.from_pretrained(
        encoder_checkpoint, decoder_model=decoder_model
    )

    model.to(args.device)
    model.decoder.to_for_other(args.device)
    model.eval()

    processor=ChemistryProcessor()

    return args,model,tokenizers,processor

def preprocess_line(line):
    line='\t'.join([str(uuid.uuid1()),line.strip()])
    print('debug in preprocess_line line=[{}]'.format(line))
    return line

def rconvert_c2cs(s):
    seqs=s.split('[SEP]')
    cjobj=[]
    for seq in seqs:
        jobj={}
        seq=seq.strip(' ')
        tokens=seq.split(' ')
        if tokens[0] == '0':
            jobj['type'] = 'physical unit'
            jobj['value'] = ' '.join(tokens[1:]).replace('[unused3]', '[OF]').replace('[unused1]', '[IN]').replace('[unused2]', '[=]').replace('[UNK]','').strip(' ')
        elif tokens[0] == '1':
            jobj['type'] = 'chemical equation'
            jobj['value'] = ' '.join(tokens[1:]).replace('[unused3]', '[OF]').replace('[unused1]', '[IN]').replace('[unused2]', '[=]').replace('[UNK]','').strip(' ')
        elif tokens[0] == '2':
            jobj['type'] = 'other'
            jobj['value'] = ' '.join(tokens[1:]).replace('[unused3]', '[OF]').replace('[unused1]', '[IN]').replace('[unused2]', '[=]').replace('[UNK]','').strip(' ')
        else:
            print('c wrong = [{}] in [{}]'.format(seq,s))
        cjobj.append(jobj)
    return cjobj

def rconvert_q2qjson(q):
    q=q.split(' [SEP] ')[0]
    tokens=q.split(' ')
    qobjs=[]
    qobj={}
    if tokens[0] == '0':
        qobj['type']='physical unit'
        qobj['value'] = ' '.join(tokens[1:]).replace('[unused3]', '[OF]').replace('[unused1]', '[IN]').replace('[unused2]', '[=]')
    elif tokens[0]=='1':
        qobj['type'] = 'other'
        qobj['value'] = ' '.join(tokens[1:])
    else:
        print('q wrong = {}'.format(q))
    qobjs.append(qobj)
    return qobjs



def build_jobj_from_oneline(question,line):
    #if is_bert_tokenize:
    #    question_tokens=tokenizing(question)
    #else:
    #    question_tokens=question.split()
    question_tokens = question.split()
    qc_blocks=line.split(' qc_end ')
    for i in range(len(qc_blocks)):
        qc_blocks[i]=qc_blocks[i].split()
    if qc_blocks[0][0]=='start':
        qc_blocks[0]=qc_blocks[0][1:]
    if qc_blocks[-1][-1] == 'end':
        qc_blocks=qc_blocks[:-1]

    jobj={}

    #question varible
    q_jobj={}
    if qc_blocks[0][0] == 'physical_unit':
        q_jobj['type'] = qc_blocks[0][0].replace('_',' ')
        subject=question_tokens[int(qc_blocks[0][1]):int(qc_blocks[0][2])+1]
        if subject == None or subject == []:
            subject="None"
        else:
            subject=' '.join(subject)
        property = qc_blocks[0][3]
        unit=qc_blocks[0][4]
        q_jobj['value']="{} [OF] {} [IN] {}".format(property,subject,unit)

    elif qc_blocks[0][0] == 'chemical_equation' or \
            qc_blocks[0][0] == 'chemical_formula':
        q_jobj['type']='other'
        q_jobj['value']='balance_equation'
    jobj["question_variable"] = [q_jobj]
    #conditions
    c_objs=[]
    for i in range(1,len(qc_blocks)):
        c_jobj={}
        if qc_blocks[i][0] == 'physical_unit':
            c_jobj['type'] = qc_blocks[i][0].replace('_', ' ')
            subject =question_tokens[int(qc_blocks[i][1]):int(qc_blocks[i][2]) + 1]
            if subject == None or subject == []:
                subject = "None"
            else:
                subject = ' '.join(subject)
            value = question_tokens[int(qc_blocks[i][3]):int(qc_blocks[i][4]) + 1]
            if value == None or value == []:
                value = "None"
            else:
                value = ' '.join(value)
            predicate=qc_blocks[i][5]
            c_jobj['value']='{} [OF] {} [=] '.format(predicate,subject) + r'\\pu{' +str(value) + '}'
        elif qc_blocks[i][0] == 'chemical_equation' or \
            qc_blocks[i][0] == 'chemical_formula' or \
            qc_blocks[i][0] == 'substance':
            c_jobj['type'] = qc_blocks[i][0].replace('_', ' ')
            subject = question_tokens[int(qc_blocks[i][1]):int(qc_blocks[i][2]) + 1]
            if subject == None or subject == []:
                subject = "None"
            else:
                subject = ' '.join(subject)
            c_jobj['value'] = subject
        elif  qc_blocks[i][0] == 'c_other':
            c_jobj['type'] = 'other'
            c_jobj['value'] = qc_blocks[i][1]

        c_objs.append(c_jobj)

    jobj["conditions"]=c_objs

    return jobj


def parse_oneline(line,args,model,tokenizers,processor):

    model_line=preprocess_line(line)

    encoder_tokenizer=tokenizers[0]
    decoder_tokenizer=tokenizers[1]

    examples=processor.get_examples_from_tsvlines([model_line])
    inputs, outputs, vocabs, inputs_mask, outputs_mask, vocabs_mask,\
    outputs_mask_lm_labels, vocabs_mask_lm_labels, example_buffer = \
        collate(examples, encoder_tokenizer=encoder_tokenizer,decoder_tokenizer=decoder_tokenizer,
                                         input_block_size=args.input_block_size,output_block_size=args.output_block_size)

    inputs = inputs.to(args.device)
    outputs = outputs.to(args.device)
    vocabs = vocabs.to(args.device)

    inputs_mask = inputs_mask.to(args.device)
    outputs_mask = outputs_mask.to(args.device)
    vocabs_mask = vocabs_mask.to(args.device)

    outputs_mask_lm_labels = outputs_mask_lm_labels.to(args.device)
    vocabs_mask_lm_labels = vocabs_mask_lm_labels.to(args.device)
    jobj=None
    with torch.no_grad():

        if args.decoding_type == 'decoding':
            tokens_roles = []
            outputs_ids, vocab_mask_index = model.decoding(
                encoder_input_ids=inputs,
                decoder_input_ids=outputs,
                decoder_vocab_mask_index=vocabs,
                encoder_attention_mask=inputs_mask,
                decoder_attention_mask=outputs_mask,
                decoder_lm_labels=None,
                tokenizer=decoder_tokenizer, fsa=fsa
            )
            print('outputs size: {}'.format(outputs_ids.size()))
            outputs_ids = outputs_ids.cpu().numpy()
            vocab_mask_index = vocab_mask_index.cpu().numpy()

            for i, idx in enumerate(outputs_ids):
                tokens = []
                for id in idx:
                    token = decoder_tokenizer.ids_to_tokens.get(id, decoder_tokenizer.unk_token)
                    tokens.append(token)
                    if token == 'end':
                        break
                # print("idx={}".format(idx))
                # print("tokens={}".format(tokens))
                # print('Before Whole Index: {}'.format(tokens))
                tokens = translate_subtokenindex_backto_tokenindex(example_buffer[i], tokens, vocab_mask_index[i])
                jobj=build_jobj_from_oneline(line.strip(), ' '.join(tokens))

        else:
            pass


    return jobj

def main():

    #test
    args, model, tokenizer, processor=init()
    line="How many moles of sodium carbonate are present in 6.80 grams of sodium carbonate?"
    res=parse_oneline(line,args,model,tokenizer,processor)
    print('{}\t{}'.format(line,json.dumps(res, ensure_ascii=False)))


def web_serving():
    args, model, tokenizer, processor = init()
    import flask
    server=flask.Flask(__name__)



    @server.route('/parse', methods=['get', 'post'])
    def parse():
        line = flask.request.values.get('q')
        print('debug by zhuoyu: q={}'.format(line))
        json_res=parse_oneline(line, args, model, tokenizer, processor)
        return json.dumps(json_res, ensure_ascii=False)

    server.run(debug=True,host='0.0.0.0',port=36521)


if __name__ == "__main__":
    main()
    web_serving()
