from datasets import load_from_disk
import torch

squad_train_file = "data/squad_train"
dolly_test_file = "data/closed"

def load_data():
    squad = load_from_disk(squad_train_file)
    dolly = load_from_disk(dolly_test_file)
    return squad, dolly

def collate_fn(batch, question_key, context_key, answer_key, qc_len, a_len, eos_idx, bos_idx, pad_idx, a_pad_idx, padding_function):
    """
    Each batch is a list of dictionaries, where each dictionary is a single
    datapoint. Each datapoint has a question, a context, and an answer, and
    each of these is a vector of variable length.

    Padding is done by prepending the bos_idx and appending the eos_idx and
    then padding such that final length is equal to the respective specified lengths

    If a given vector is already of maximum length or longer, it is truncated
    to length max_len - 1 and prepended with bos_idx and appended with eos_idx.

    For question and context, they are concatenated together and padded to the
    qc_len. The answer is padded to a_len.

    @param batch: list of dictionaries, where each dictionary is a datapoint
    @param question_key: key for accessing question in each datapoint
    @param context_key: key for accessing context in each datapoint
    @param answer_key: key for accessing answer in each datapoint
    @param qc_len: length to pad question and context to
    @param a_len: length to pad answer to
    @param eos_idx: index of end of sentence token
    @param bos_idx: index of beginning of sentence token
    @param pad_idx: index of padding token
    @param a_pad_idx: index of padding token for answers
    @param padding_function: accepts a tensor, max_len, eos_idx, bos_idx, padding_idx,
        and returns a padded tensor

    @return: tuple of tensors (padded question and context, padded answer)
    """
    inputs = [
        handleqc(elem, question_key, context_key, qc_len, eos_idx, bos_idx, pad_idx, padding_function) for elem in batch
    ]
    answers = [
        handlea(elem, answer_key, a_len, eos_idx, bos_idx, a_pad_idx, padding_function) for elem in batch
    ]

    # for elem in batch:
    #     question = elem[question_key]
    #     context = elem[context_key]
    #     ans = elem[answer_key]

    #     qc = torch.cat((question, context))
    #     qc = padding_function(qc, qc_len, eos_idx, bos_idx, pad_idx)
    #     inputs.append(qc)

    #     a = padding_function(ans, a_len, eos_idx, bos_idx, a_pad_idx)
    #     answers.append(a)

    return torch.stack(inputs), torch.stack(answers)

def handleqc(elem, question_key, context_key, qc_len, eos_idx, bos_idx, pad_idx, padding_function):
    question = elem[question_key]
    context = elem[context_key]
    qc = torch.cat((question, context))
    qc = padding_function(qc, qc_len, eos_idx, bos_idx, pad_idx)
    return qc

def handlea(elem, answer_key, a_len, eos_idx, bos_idx, a_pad_idx, padding_function):
    ans = elem[answer_key]
    a = padding_function(ans, a_len, eos_idx, bos_idx, a_pad_idx)
    return a

def dolly_collate_fn(batch, qc_len, a_len, eos_idx, bos_idx, pad_idx, a_pad_idx, padding_function):
    return collate_fn(batch, "instruction", "context", "response", qc_len, a_len, eos_idx, bos_idx, pad_idx, a_pad_idx, padding_function)

def squad_collate_fn(batch, qc_len, a_len, eos_idx, bos_idx, pad_idx, a_pad_idx, padding_function):
    return collate_fn(batch, "question", "context", "answers", qc_len, a_len, eos_idx, bos_idx, pad_idx, a_pad_idx, padding_function)

def prep_squad_collate_fn(qc_len, a_len, eos_idx, bos_idx, pad_idx, a_pad_idx, padding_function):
    return lambda batch: squad_collate_fn(batch, qc_len, a_len, eos_idx, bos_idx, pad_idx, a_pad_idx,padding_function)

def prep_dolly_collate_fn(qc_len, a_len, eos_idx, bos_idx, pad_idx, a_pad_idx, padding_function):
    return lambda batch: dolly_collate_fn(batch, qc_len, a_len, eos_idx, bos_idx, pad_idx, a_pad_idx,padding_function)