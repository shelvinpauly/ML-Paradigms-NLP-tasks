import transformers
import numpy as np
import evaluate, torch, os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (DataCollatorWithPadding, 
                          DataCollatorForTokenClassification,
                          AutoTokenizer,
                          default_data_collator,
                          AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification,
                          AutoModelForQuestionAnswering,
                          PreTrainedTokenizerFast,
                          AdamW,
                          get_scheduler,
                          EvalPrediction)

from datasets import load_dataset, DatasetDict, load_metric, concatenate_datasets
from torch import nn
from tqdm.auto import tqdm
import logging
import collections
import math

# Number of epochs to train, we mostly keep this at 1.
num_epochs = 1

# Initial TaskSampler weights. Will be static if use_dyanmic_weight_avg is False.
# The correspondence of weights to tasks is [NLI, NER, MRC]. These will always
# add to 1. See sequential_mode for how to adjust for sequential orderings.
initial_weights = [1.0, 0.0, 0.0]

# Set this to use dynamic weight average (DWA), True to use DWA.
use_dynamic_weight_avg = False

# When DWA is enabled, this set the temperature hyperparameter, which controls the
# "softness" of task weighting. We keep it mostly at 2.
dwa_temp = 2 

# This controls whether or not we should train NER and NLI sequentially, set to
# true for sequential. If sequential mode is active, you need to make the initial
# weights [0.0, 1.0, 0.0] for NER first or [1.0, 0.0, 0.0] for NLI first.
sequential_mode = True

# This sets the portion of the training set to use. Adjust these to adjust the
# weight of NLI and NER in sequential mode. The third value (SQuAD/MRC) should
# always be 0.1, and none of these can be zero. [NLI, NER, MRC]
training_set_portions = [1.0, 1.0, 0.1]

# This controls whether or not we resume from a checkpoint
#    False: do not resume from checkpoint
#    "": resume from most recent checkpoint
#    "filename": resume from checkpoint file called filename
resume_from_checkpoint = False

# This controls batch size. We typically use 16.
batch_size = 16

# This controls the learning rate. We typically use 2e-5.
learning_rate = 2e-5

if sequential_mode:
    assert initial_weights == [0.0, 1.0, 0.0] or initial_weights == [1.0, 0.0, 0.0]
assert training_set_portions[2] == 0.1
for v in training_set_portions:
    assert v <= 1.0 and v != 0.0

class TaskSampler():
    """ 
    Class for sampling batches from a dictionary of dataloaders according to a weighted sampling scheme.

    Dynamic task weights can be externally computed and set using the set_task_weights method,
    or, this class can be extended with methods and state state to implement a more complex sampling scheme.

    You probably/shouldn't need to use this with multiple GPUs, but if you do, you'll may need
    to extend/debug it yourself since the current implementation is not distributed-aware.
    
    Args:
        dataloader_dict (dict[str, DataLoader]): Dictionary of dataloaders to sample from.
        task_weights (list[float], optional): List of weights for each task. If None, uniform weights are used. Defaults to None.
        max_iters (int, optional): Maximum number of iterations. If None, infinite. Defaults to None.
    """
    def __init__(self, *, dataloader_dict, task_weights, max_iters):
        assert dataloader_dict is not None, "Dataloader dictionary must be provided."
        self.dataloader_dict = dataloader_dict
        self.task_names = list(dataloader_dict.keys())
        self.dataloader_iterators = self._initialize_iterators()
        self.task_weights = task_weights if task_weights is not None else self._get_uniform_weights()
        self.max_iters = max_iters if max_iters is not None else float("inf")
    
    # Initialization methods
    def _get_uniform_weights(self):
        return [1/len(self.task_names) for _ in self.task_names]
    
    def _initialize_iterators(self):
        return {name:iter(dataloader) for name, dataloader in self.dataloader_dict.items()}
    
    # Weight getter and setter methods (NOTE can use these to dynamically set weights)
    def set_task_weights(self, task_weights):
        #assert sum(self.task_weights) == 1, "Task weights must sum to 1."
        self.task_weights = task_weights
    
    def get_task_weights(self):
        return self.task_weights

    # Sampling logic
    def _sample_task(self):
        return np.random.choice(self.task_names, p=self.task_weights)
    
    def _sample_batch(self, task):
        try:
            return self.dataloader_iterators[task].__next__()
        except StopIteration:
            print(f"Restarting iterator for {task}")
            self.dataloader_iterators[task] = iter(self.dataloader_dict[task])
            return self.dataloader_iterators[task].__next__()
        except KeyError as e:
            print(e)
            raise KeyError("Task not in dataset dictionary.")
    
    # Iterable interface
    def __iter__(self):
        self.current_iter = 0
        return self
    
    def __next__(self):
        if self.current_iter >= self.max_iters:
            raise StopIteration
        else:
            self.current_iter += 1
        task = self._sample_task()
        batch = self._sample_batch(task)
        return task, batch

model_checkpoint = "distilbert-base-uncased"

ner_datasets = load_dataset("Babelscape/wikineural")
nli_datasets = load_dataset("multi_nli")
squad_datasets = load_dataset("squad_v2")

ner_label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
ner_labels_vocab = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ner_labels_vocab_reverse = {v:k for k,v in ner_labels_vocab.items()}
#labels_vocab_reverse

ner_train_dataset = concatenate_datasets([ner_datasets["train_en"]])
ner_train_dataset = ner_train_dataset.select(range(round(training_set_portions[1]*len(ner_train_dataset))))
ner_val_dataset = concatenate_datasets([ner_datasets["val_en"]]) 
ner_test_dataset = concatenate_datasets([ner_datasets["test_en"]])

nli_train_dataset = nli_datasets["train"].select(range(round(training_set_portions[0]*len(nli_datasets["train"]))))
nli_val_dataset = nli_datasets["validation_matched"]

squad_train_dataset = squad_datasets["train"].select(range(round(0.1*len(squad_datasets["train"]))))
squad_val_dataset = squad_datasets["validation"]

#print(f"min({len(nli_train_dataset)}, {len(ner_train_dataset)}) + {len(squad_train_dataset)} = {min(len(nli_train_dataset), len(ner_train_dataset)) + len(squad_train_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
label_all_tokens = False

max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"

def prepare_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

train_features = prepare_features(squad_train_dataset[:5])

squad_train_tokenized = squad_train_dataset.map(prepare_features, batched=True, num_proc=4, remove_columns=squad_train_dataset.column_names)

# This is also only for Squad...
def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

squad_val_tokenized = squad_val_dataset.map(
    prepare_validation_features,
    batched=True, num_proc=4,
    remove_columns=squad_val_dataset.column_names
)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

ner_train_tokenized = ner_train_dataset.map(tokenize_and_align_labels, batched=True, num_proc=4, remove_columns=ner_train_dataset.column_names)
ner_val_tokenized = ner_val_dataset.map(tokenize_and_align_labels, batched=True, num_proc=4, remove_columns=ner_val_dataset.column_names)
ner_test_tokenized = ner_test_dataset.map(tokenize_and_align_labels, batched=True, num_proc=4, remove_columns=ner_test_dataset.column_names)

sentence1_key = "premise"
sentence2_key = "hypothesis"
padding = "max_length"
max_seq_length = 128

def preprocess_function(examples):
    # Tokenize the texts
    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True)

    if "label" in examples:
        result["labels"] = examples["label"]
    return result

nli_train_tokenized = nli_train_dataset.map(preprocess_function, batched=True, num_proc=4, remove_columns=nli_train_dataset.column_names)
nli_val_tokenized = nli_val_dataset.map(preprocess_function, batched=True, num_proc=4, remove_columns=nli_val_dataset.column_names)

ner_model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, 
                                                            num_labels=len(ner_label_list), 
                                                            label2id=ner_labels_vocab, 
                                                            id2label=ner_labels_vocab_reverse)
nli_config = AutoConfig.from_pretrained(model_checkpoint, num_labels=3, finetuning_task="mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=nli_config)
squad_model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

model_dict = {"nli": nli_model, "ner": ner_model, "squad": squad_model}

accelerator = Accelerator()
device = accelerator.device

for model in model_dict.values():
    model.distilbert = nli_model.distilbert
    model.to(device)

nli_dataloader = DataLoader(nli_train_tokenized, collate_fn=default_data_collator, batch_size=batch_size)
ner_dataloader = DataLoader(ner_train_tokenized, collate_fn=DataCollatorForTokenClassification(tokenizer=tokenizer), batch_size=batch_size)
squad_dataloader = DataLoader(squad_train_tokenized, collate_fn=default_data_collator, batch_size=batch_size)
dataloader_dict = {"nli": nli_dataloader, "ner": ner_dataloader, "squad": squad_dataloader}

nli_val_dataloader = DataLoader(nli_val_tokenized, collate_fn=default_data_collator, batch_size=batch_size)
ner_val_dataloader = DataLoader(ner_val_tokenized, collate_fn=DataCollatorForTokenClassification(tokenizer=tokenizer), batch_size=batch_size)
squad_val_dataloader = DataLoader(squad_val_tokenized.remove_columns(["example_id", "offset_mapping"]), collate_fn=default_data_collator, batch_size=batch_size)
val_dataloader_dict = {"nli": nli_val_dataloader, "ner": ner_val_dataloader, "squad": squad_val_dataloader}

optimizers = {
    "nli": AdamW(nli_model.parameters(), lr=learning_rate),
    "ner": AdamW(ner_model.parameters(), lr=learning_rate),
    "squad": AdamW(squad_model.parameters(), lr=learning_rate)
}

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizers["ner"],
    num_warmup_steps=0,
    num_training_steps=min(len(nli_dataloader), len(ner_dataloader)) + len(squad_dataloader)
)

task_sampler = TaskSampler(dataloader_dict=dataloader_dict, task_weights=initial_weights,max_iters=None)

(   nli_model, ner_model, squad_model, 
    optimizers["nli"], optimizers["ner"], optimizers["squad"], 
    nli_dataloader, ner_dataloader, squad_dataloader,
    nli_val_dataloader, ner_val_dataloader, squad_val_dataloader,
    lr_scheduler, task_sampler
) = accelerator.prepare(
    nli_model, ner_model, squad_model,
    optimizers["nli"], optimizers["ner"], optimizers["squad"],
    nli_dataloader, ner_dataloader, squad_dataloader,
    nli_val_dataloader, ner_val_dataloader, squad_val_dataloader,
    lr_scheduler, task_sampler
)

ner_metric = evaluate.load("seqeval")
nli_metric = evaluate.load("accuracy")
squad_metric = evaluate.load("squad_v2")

# This function is for NER only
def ner_compute_metrics():
    results = ner_metric.compute()
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

metric_dict = {"ner": ner_metric, "nli": nli_metric, "squad": squad_metric}

# This is for NER only
def get_labels(predictions, references):
    y_pred = predictions.detach().cpu().clone().numpy()
    y_true = references.detach().cpu().clone().numpy()
    true_predictions = [
        [ner_label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [ner_label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels
 
 # Post-processing:
def post_processing_function(examples, features, predictions):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    version_2_with_negative = True
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        n_best_size = 20,
        max_answer_length = 385,
        null_score_diff_threshold = 1,
        version_2_with_negative = version_2_with_negative
    )
    # Format the result to the format the metric expects.
    if version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """
    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size
    return logits_concat
# This is for Squad only


def postprocess_qa_predictions(
        examples,
        features,
        predictions,
        version_2_with_negative = True,
        n_best_size = 20,
        max_answer_length = 385,
        null_score_diff_threshold = 1):
    # output_dir: Optional[str] = None,
    # prefix: Optional[str] = None

    output_dir = None
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        print(len(predictions[0]))
        print("features QA PRED")
        print(len(features))
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    # logger.setLevel(log_level)
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if (
            version_2_with_negative
            and min_null_prediction is not None
            and not any(p["offsets"] == (0, 0) for p in predictions)
        ):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    '''
    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        print(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        print(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            print(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
    '''
    
    return all_predictions


num_updates_per_epoch = min(len(nli_dataloader), len(ner_dataloader)) + len(squad_dataloader)
print(num_updates_per_epoch)
completed_steps = 0
completed_epochs = 0
progress_bar = tqdm(range(num_updates_per_epoch*num_epochs))

resume_step = None
starting_epoch = 0
checkpointing_steps = 5790
switched_to_mrc = False

if resume_from_checkpoint != False:
    if resume_from_checkpoint == "":
        # Get the most recent checkpoint
        dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
        dirs.sort(key=os.path.getctime)
        path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        accelerator.print(f"Resumed from checkpoint: {path}")
        accelerator.load_state(path)
    else:
        accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint)
        path = os.path.basename(resume_from_checkpoint)
    
    # Extract `epoch_{i}` or `step_{i}`
    training_difference = os.path.splitext(path)[0]

    if "epoch" in training_difference:
        starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        resume_step = None
    else:
        resume_step = int(training_difference.replace("step_", ""))
        starting_epoch = resume_step // num_updates_per_epoch*num_epochs
        resume_step -= starting_epoch * num_updates_per_epoch

for epoch in range(starting_epoch, num_epochs):
    for model in model_dict.values(): model.train()
    # TRAINING LOOP
    prev_losses_dict = {"nli": 1.0, "ner": 1.0}
    prev2_losses_dict = {"nli": 1.0, "ner": 1.0}
    for step, (task, batch) in enumerate(task_sampler):
        # skip to resume point if applicable
        if resume_from_checkpoint != False and epoch == starting_epoch and resume_step is not None and step < resume_step:
            completed_steps += 1
            if step % 10 == 0 and step > 0: progress_bar.update(10)
            if step == resume_step: progress_bar.update(step % 10)
            continue
        if step >= min(len(nli_dataloader), len(ner_dataloader)) and not switched_to_mrc:
            # Switch to only training the MRC head
            switched_to_mrc = True
            task_sampler.set_task_weights([0.0, 0.0, 1.0])
            for layer in model.distilbert.parameters():
                layer.requires_grad = False
            print("switched to training mrc only")

        if sequential_mode and not switched_to_mrc and completed_steps == min(len(nli_dataloader), len(ner_dataloader))//2:
            task_sampler.set_task_weights([task_sampler.get_task_weights()[1], task_sampler.get_task_weights()[0], 0.0])
            print("weights swapped for sequential mode")
                
        optimizers[task].zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model_dict[task](**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizers[task].step()
        lr_scheduler.step()
        
        if use_dynamic_weight_avg and not switched_to_mrc:
            curr_task_weights = task_sampler.get_task_weights()
            t1 = task
            t2 = "ner" if task == "nli" else "nli"
            r_i = {}
            r_i[t1] = loss / prev_losses_dict[t1]
            r_i[t2] = prev_losses_dict[t2] / prev2_losses_dict[t2]
            nli_weight = math.exp(r_i["nli"]/dwa_temp)/(math.exp(r_i["nli"]/dwa_temp) + math.exp(r_i["ner"]/dwa_temp))
            ner_weight = math.exp(r_i["ner"]/dwa_temp)/(math.exp(r_i["nli"]/dwa_temp) + math.exp(r_i["ner"]/dwa_temp))
            # weights need to sum to 1 so omitting the T in the numerator and doing an extra normalization pass
            weights = np.array([nli_weight, ner_weight, 0.0]).astype('float64')
            weights /= weights.sum()
            task_sampler.set_task_weights(weights.tolist())
            prev2_losses_dict[task] = prev_losses_dict[task]
            prev_losses_dict[task] = loss
        
        progress_bar.update(1)
        completed_steps += 1
        
        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps}"
                accelerator.save_state(output_dir)
        
        if completed_steps >= num_updates_per_epoch: break
    print("ready to EVAL")
    
    for task in ("ner", "nli", "squad"):        
        progress_bar_eval = tqdm(range(len(val_dataloader_dict[task])))
        model_dict[task].eval()
        samples_seen = 0
        all_start_logits = []
        all_end_logits = []
        # EVAL LOOP
        print(f"Val dataloader len: {len(val_dataloader_dict[task])}")
        batch_count = 0
        output_count = 0
        for step, batch in enumerate(val_dataloader_dict[task]):
            batch = {k: v.to(device) for k, v in batch.items()}
            if task == "squad":
                with torch.no_grad():
                    outputs = model(**batch)
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                    #print(f"start logits pre-pad: {len(start_logits)}")

                    # if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)
                    #print(f"start logits post-pad: {len(start_logits)}")

                    all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                    all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())
                    #print(f"start logits total: {len(all_start_logits)}")
            else:
                # print("HELLO IN ELSE")
                with torch.no_grad():
                    outputs = model_dict[task](**batch)
                preds = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                preds, labels = accelerator.gather((preds, labels))
                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(val_dataloader_dict[task]) - 1:
                        preds = preds[: len(val_dataloader_dict[task].dataset) - samples_seen]
                        labels = labels[: len(val_dataloader_dict[task].dataset) - samples_seen]
                    else:
                        samples_seen += labels.shape[0]
                if task == "ner": 
                    preds, labels = get_labels(preds, labels)
                if len(preds) != len(labels):
                    print("Skipping a bad test")
                    continue
                metric_dict[task].add_batch(
                    predictions=preds,
                    references=labels
                )
            progress_bar_eval.update(1)
        
        if task == "squad":
            max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
            
            # concatenate the numpy array
            start_logits_concat = create_and_fill_np_array(all_start_logits, squad_val_tokenized, max_len)
            end_logits_concat = create_and_fill_np_array(all_end_logits, squad_val_tokenized, max_len)
            
            print(f"start_logits_concat len: {len(start_logits_concat)}") #11873
            print(f"squad_val_tokenized len: {len(squad_val_tokenized)}") #12134

            # delete the list of numpy arrays
            del all_start_logits
            del all_end_logits

            outputs_numpy = (start_logits_concat, end_logits_concat)
            prediction = post_processing_function(squad_val_dataset, squad_val_tokenized, outputs_numpy)
            predict_metric = metric_dict[task].compute(predictions=prediction.predictions, references=prediction.label_ids)
            print(f"{task} epoch {epoch}: {predict_metric}")
        else:
            eval_metric = ner_compute_metrics() if task == "ner" else metric_dict[task].compute()
            print(f"{task} epoch {epoch}:", eval_metric)
    

    if checkpointing_steps == "epoch":
        output_dir = f"epoch_{epoch}"
        accelerator.save_state(output_dir)
