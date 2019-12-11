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
sys.path.append(r'../')


import numpy as np
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
)

from utils_summarization import (
    CNNDailyMailDataset,
    encode_for_summarization,
    fit_to_block_size,
    build_lm_labels,
    build_mask,
    compute_token_type_ids,
)

from utils_chemistry import (ChemistryDataset,)
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


def load_and_cache_examples(args, tokenizer, prefix="train"):
    dataset = ChemistryDataset(tokenizer, prefix=prefix, data_dir=args.data_dir)
    return dataset


def collate(data, tokenizer, input_block_size,output_block_size):
    """ List of tuple as an input. """
    question_inputs=[]
    question_varible_outputs=[]
    condition_outputs=[]
    for i,example in enumerate(data):
        question_input=tokenizer.encode(example.question_input)
        question_input=fit_to_block_size(question_input, input_block_size, tokenizer.pad_token_id)
        question_inputs.append(question_input)

        if example.question_varible_output is not None:
            question_varible_output=tokenizer.encode(example.question_varible_output)
        else:
            question_varible_output=tokenizer.build_inputs_with_special_tokens([])
        question_varible_output=fit_to_block_size(question_varible_output, output_block_size, tokenizer.pad_token_id)
        question_varible_outputs.append(question_varible_output)

        if example.condition_output is not None:
            condition_output=tokenizer.encode(example.condition_output)
        else:
            condition_output=tokenizer.build_inputs_with_special_tokens([])
        condition_output=fit_to_block_size(condition_output, output_block_size, tokenizer.pad_token_id)
        condition_outputs.append(condition_output)

    question_inputs = torch.tensor(question_inputs)
    question_varible_outputs = torch.tensor(question_varible_outputs)
    condition_outputs = torch.tensor(condition_outputs)

    question_inputs_mask = build_mask(question_inputs, tokenizer.pad_token_id)
    question_varible_outputs_mask = build_mask(question_varible_outputs, tokenizer.pad_token_id)
    condition_outputs_mask = build_mask(condition_outputs, tokenizer.pad_token_id)

    question_varible_outputs_mask_lm_labels = build_lm_labels(question_varible_outputs_mask, tokenizer.pad_token_id)
    condition_outputs_mask_lm_labels = build_lm_labels(condition_outputs_mask, tokenizer.pad_token_id)

    return (
        question_inputs,
        [question_varible_outputs,condition_outputs],
        question_inputs_mask,
        [question_varible_outputs_mask,condition_outputs_mask],
        [question_varible_outputs_mask_lm_labels,condition_outputs_mask_lm_labels],
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
    train_iterator = trange(args.num_train_epochs, desc="Epoch", disable=True)

    global_step = 0
    tr_loss = 0.0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
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

        source = source.to(args.device)
        target = target.to(args.device)

        encoder_mask = encoder_mask.to(args.device)
        decoder_mask = decoder_mask.to(args.device)
        lm_labels = lm_labels.to(args.device)

        with torch.no_grad():

            if args.decoding_type=='decoding':
                outputs_ids=model.decoding(
                    source,
                    target,
                    encoder_attention_mask=encoder_mask,
                    decoder_attention_mask=decoder_mask,
                    decoder_lm_labels=lm_labels,
                    #fdebug=fdebug,
                )
                print('outputs size: {}'.format(outputs_ids.size()))
                outputs_ids =outputs_ids.cpu().numpy()
                for idx in outputs_ids:
                    #print(idx)
                    #print('###')
                    tokens = []
                    for id in idx:
                        #print('{}\t{}'.format(id,type(id)))
                        tokens.append(tokenizer.ids_to_tokens.get(int(id), tokenizer.unk_token))
                    fout.write(' '.join(tokens) + '\n')

            else:
                outputs = model(
                    source,
                    target,
                    encoder_attention_mask=encoder_mask,
                    decoder_attention_mask=decoder_mask,
                    decoder_lm_labels=lm_labels,
                    #fdebug=fdebug,
                )

                lm_loss = outputs[0]
                predicted_scores = outputs[1].argmax(-1).cpu().numpy().tolist()
                for idx in predicted_scores:
                    tokens = []
                    for id in idx:
                        tokens.append(tokenizer.ids_to_tokens.get(id, tokenizer.unk_token))
                    fout.write(' '.join(tokens) + '\n')

                # print('debug by zhuoyu, predicted_scores size={}'.format(predicted_scores.size()))
                eval_loss += lm_loss.mean().item()




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


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
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
        default=False,
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
        default="bert-base-cased",
        type=str,
        help="The model checkpoint to initialize the encoder's weights with.",
    )
    parser.add_argument(
        "--decoder_model_name_or_path",
        default="/data/zhuoyu/semantic_parsing/models",
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
        default=4,
        type=int,
        help="Batch size per GPU/CPU for eval.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
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
        default="",
        type=str,
        help="trained_checkpoints",
    )

    parser.add_argument(
        "--decoding_type",
        default="pnt",
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
    args = parser.parse_args()

    if (
                        os.path.exists(args.output_dir)
                    and os.listdir(args.output_dir)
                and args.do_train
            and not args.do_overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --do_overwrite_output_dir to overwrite.".format(
                args.output_dir
            )
        )



    # Set up training device
    if args.to_cpu or not torch.cuda.is_available():
        args.device = torch.device("cpu")
        args.n_gpu = 0
    else:
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        print(args.n_gpu)

    # Load pretrained model and tokenizer. The decoder's weights are randomly initialized.
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name_or_path)
    #config = BertConfig.from_pretrained(args.model_name_or_path)
    #config.num_hidden_layers=3
    #config.is_decoder=True
    #decoder_model = BertForMaskedLM(config)
    decoder_models=[BertForMaskedLM.from_pretrained(args.decoder_model_name_or_path),
                    BertForMaskedLM.from_pretrained(args.decoder_model_name_or_path)]
    model = Model2Models.from_pretrained(
        args.encoder_model_name_or_path, decoder_model=decoder_models
    )
    #model = Model2Model.from_pretrained(
    #    args.model_name_or_path, decoder_model=None
    #)

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

    # Train the model
    model.to(args.device)
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_arguments.bin"))

    # Evaluate the model
    results = {}
    if args.do_evaluate:
        checkpoints = [args.trained_checkpoints]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            encoder_checkpoint = os.path.join(checkpoint, "encoder")
            decoder_checkpoint = os.path.join(checkpoint, "decoder")
            #model = PreTrainedEncoderDecoder.from_pretrained(
            #    encoder_checkpoint, decoder_checkpoint
            #)
            #model = Model2Model.from_pretrained(encoder_checkpoint)
            #model.to(args.device)
            results = "placeholder"

            evaluate(args,model,tokenizer,"test")

    return results


if __name__ == "__main__":
    main()
