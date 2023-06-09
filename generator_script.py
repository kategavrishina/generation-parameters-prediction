import json
import random

from tqdm import tqdm

import torch
import numpy as np


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class GeneratorInference:
    _default_generation_parameters = {
        "do_sample": True,
    }

    def __init__(self, tokenizer, model, max_length=1024, **generation_parameters):

        self.tokenizer = tokenizer

        self.model = model

        self.max_length = max_length

        self.model.eval()

        self.generation_parameters = generation_parameters or self._default_generation_parameters

    @staticmethod
    def postprocessing(response):
        response = response.strip()
        return response

    def __call__(self, context, num_return_sequences=1):

        while True:

            if len(context) == 1:
                return None

            tokenized = self.tokenizer(" <turn> ".join(context))
            indices = tokenized.input_ids
            if len(indices) > self.max_length:
                context = context[1:]
            else:
                break

        indices = torch.tensor([indices], dtype=torch.long).to(self.model.device)

        with torch.inference_mode():
            generated_indices = self.model.generate(
                input_ids=indices,
                **self.generation_parameters,
                num_return_sequences=num_return_sequences,
                max_length=self.max_length,
            ).detach().cpu()

        generated_responses = self.tokenizer.batch_decode(generated_indices, skip_special_tokens=True)

        generated_responses = [self.postprocessing(response) for response in generated_responses]

        return generated_responses


def start_generator(generative_model_name):
    generative_tokenizer = AutoTokenizer.from_pretrained(generative_model_name)
    generative_model = AutoModelForSeq2SeqLM.from_pretrained(generative_model_name)

    generative_model.cuda()

    generator = GeneratorInference(generative_tokenizer, generative_model)

    return generator


def get_data(input_file):
    data = list()

    with open(input_file, 'r') as f:
        for dialog in f.readlines():
            dialog = json.loads(dialog)
            data.append(dialog)
    return data


def choose_params():
    params = {'top_p': 1.0, 'top_k': 50, 'typical_p': 1.0, 'temperature': 1.0, 'do_sample': True}

    top_k_range = [5, 25, 50, 75, 100, 150, 250, 350, 500, 750, 1000]

    top_p_range = np.arange(0.05, 1.01, 0.05)

    typical_p_range = top_p_range

    temperature_range = np.arange(0.05, 1.21, 0.05)

    change_type = random.choice(['top_p', 'typical_p', 'both'])

    params['temperature'] = random.choice(temperature_range)

    if change_type == 'top_p':
        params['top_p'] = random.choice(top_p_range)
        params['top_k'] = random.choice(top_k_range)
    elif change_type == 'typical_p':
        params['typical_p'] = random.choice(typical_p_range)
    else:
        params['top_p'] = random.choice(top_p_range)
        params['top_k'] = random.choice(top_k_range)
        params['typical_p'] = random.choice(typical_p_range)

    return change_type, params


def answers_generator(dialog: dict, generator, num_answers: int):

    answers = generator(dialog['dialog'], num_return_sequences=num_answers)
    if answers is None:
        return None
    else:
        dialog['predicted_answers'] = [{'answer': answer} for answer in list(answers)]
        dialog['params'] = generator.generation_parameters
        return dialog


def generate_answers(input_file, output_file, generator, num_dialogs=None, num_answers=15, longest=True,
                     fixed=False, random_params=False, **generation_parameters):
    data = get_data(input_file)
    if num_dialogs is not None:
        data = random.sample(data, num_dialogs)
    if longest:
        data = sorted(data, key=lambda x: len(x['dialog']), reverse=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        if fixed:
            generator.generation_parameters = generation_parameters
            for dialog in tqdm(data):
                dialog = answers_generator(dialog, generator, num_answers)
                if dialog is None:
                    continue
                f.write(json.dumps(dialog))
                f.write('\n')
        else:
            default = {'top_p': 1.0, 'top_k': 50, 'typical_p': 1.0, 'temperature': 1.0, 'do_sample': True}
            for dialog in tqdm(data):
                if random_params:
                    change_type, generator.generation_parameters = choose_params()
                    dialog = answers_generator(dialog, generator, num_answers)
                    dialog['change_type'] = change_type
                else:
                    generator.generation_parameters = dialog['pred_params']
                    answer1 = generator(dialog['dialog'], num_return_sequences=num_answers)
                    generator.generation_parameters = default
                    answer2 = generator(dialog['dialog'], num_return_sequences=num_answers)
                    dialog['answer_pred_params'] = answer1
                    dialog['answer_default_params'] = answer2
                if dialog is None:
                    continue
                f.write(json.dumps(dialog))
                f.write('\n')

    f.close()

    return output_file


if __name__ == "__main__":
    model_name = "allenai/cosmo-xl"

    generator = start_generator(model_name)
    # _ = generate_answers("emp_train.jsonl", "generated_random_params_big.jsonl", generator, num_answers=5, random_params=True)
    _ = generate_answers("final_embedded_pred_params_with_type.jsonl", "generated_pred_params.jsonl", generator, num_answers=3, fixed=False, random_params=False)


