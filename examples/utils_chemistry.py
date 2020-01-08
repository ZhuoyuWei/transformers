import os
import sys
import logging
import csv
import torch
from torch.utils.data import Dataset

from finite_state_automata import FiniteStateAutomata


'''
From tsv to example: Processor
From example to feature: convert_examples_to_features
'''

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self,example_id,question_input,question_varible_output=None,condition_output=None):
        self.example_id=example_id
        self.question_input=question_input
        self.question_varible_output=question_varible_output
        self.condition_output=condition_output

class InputExampleV2(object):
    def __init__(self,example_id,input,output=None,fsa=None):
        self.example_id=example_id
        self.input=input
        self.output=output

        self.fsa=fsa
        self.fsa_states=None
        self.vocab_indexes=None
        if self.fsa:
            self.fsa_states=self._get_fsa_states()
        if self.fsa_states:
            self.vocab_indexes=self.fsa.get_vocab_for_states(self.fsa_states)
            print("vocab_indexes = {}".format(self.vocab_indexes))
            exit(-1)


    def _get_fsa_states(self):
        if self.output:
            return self.fsa.convert_seq_to_states(self.output.split(' '))
        else:
            return []

class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()




class ChemistryProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def __init__(self,version='v1',fsa_or_config=None):
        self.version=version
        if version == 'v2':
            if type(fsa_or_config) == FiniteStateAutomata:
                self.fsa=fsa_or_config
            else:
                self.fsa=FiniteStateAutomata.init_from_config(fsa_or_config)

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        csv_buffer=self._read_csv(os.path.join(data_dir, "train.tsv"))
        if self.version == 'v1':
            return self._create_examples(csv_buffer, "train")
        else:
            return self._create_examples_V2(csv_buffer, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        csv_buffer=self._read_csv(os.path.join(data_dir, "dev.tsv"))
        if self.version == 'v1':
            return self._create_examples(csv_buffer, "dev")
        else:
            return self._create_examples_V2(csv_buffer, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        csv_buffer=self._read_csv(os.path.join(data_dir, "test.tsv"))
        if self.version == 'v1':
            return self._create_examples(csv_buffer, "test")
        else:
            return self._create_examples_V2(csv_buffer, "test")



    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8',newline='') as f:
            reader = csv.reader(f,delimiter='\t')
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        '''
        examples = [
            InputExample(
                example_id=line[0],
                input_text=line[1],
                output_text=line[2] if len(line) > 2 else None
            ) for line in lines
        ]'''
        examples=[]
        for line in lines:
            #print('debug line = {}'.format(len(line)))
            examples.append(InputExample(
                example_id=line[0],
                question_input=line[1],
                question_varible_output=line[2] if len(line) > 2 else None,
                condition_output=line[3] if len(line) > 3 else None,
            ))
        return examples

    def _create_examples_V2(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples=[]
        for line in lines:
            #print('debug line = {}'.format(len(line)))
            examples.append(InputExampleV2(
                example_id=line[0],
                input=line[1],
                output=line[2] if len(line) > 2 else None,
                fsa=self.fsa
            )
            )
        return examples


    def get_examples_from_tsvlines(self,lines):
        parsed_lines=[]
        for line in lines:
            parsed_line=line.strip().split('\t')
            parsed_lines.append(parsed_line)
            print('Debug parsed line = {}'.format(parsed_line))
        if self.version == 'v1':
            return self._create_examples(parsed_lines,"test")
        else:
            return self._create_examples_V2(parsed_lines,"test")

class ChemistryDataset(Dataset):
    """ Abstracts the dataset used to train seq2seq models.

    CNN/Daily News:

    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, untar both datasets in the same folder, and pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].

    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init__(self, tokenizer, prefix="train", data_dir="",version='v1',fsa_or_config=None):
        assert os.path.isdir(data_dir)
        self.tokenizer = tokenizer
        self.processor=ChemistryProcessor(version=version,fsa_or_config=fsa_or_config)
        self.examples=None
        if prefix == "train":
            self.examples=self.processor.get_train_examples(data_dir)
        elif prefix == "dev":
            self.examples=self.processor.get_dev_examples(data_dir)
        elif prefix == "test":
            self.examples=self.processor.get_test_examples(data_dir)



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example=self.examples[idx]
        return example
