import numpy as np

import torch
from torch import nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import json
import random
import tqdm


class DiscriminatorInference:

    def __init__(self, tokenizer, model, max_context=15, bot_prompt="Bot", human_prompt="Person"):

        self.tokenizer = tokenizer
        self.model = model.cuda()

        self.max_context = max_context

        self.bot_prompt = bot_prompt
        self.human_prompt = human_prompt

        self.model.eval()

    def prompting(self, dialog):

        prompted_dialog = list()

        for n, phrase in enumerate(dialog[::-1]):
            person_prompt = f"{self.bot_prompt}: " if n % 2 == 0 else f"{self.human_prompt}: "
            prompted_phrase = person_prompt + phrase
            prompted_dialog.insert(0, prompted_phrase)

        return prompted_dialog

    def __call__(self, dialogs):

        prompted_dialogs = ["\n".join(self.prompting(dialog[-self.max_context:])) for dialog in dialogs]

        tokenized = self.tokenizer(prompted_dialogs, padding=True, return_tensors="pt").to('cuda')

        with torch.inference_mode():
            logits = self.model(**tokenized).logits

        predictions = torch.sigmoid(logits).squeeze()

        return predictions


def start_discriminative(discriminative_model_name, pretrained_model):

    discriminative_tokenizer = AutoTokenizer.from_pretrained(discriminative_model_name, use_fast=False)
    discriminative_model = AutoModelForSequenceClassification.from_pretrained(discriminative_model_name)

    discriminative_model.classifier = nn.Linear(discriminative_model.config.hidden_size, 1)
    state_dict = torch.load(pretrained_model, map_location="cpu")

    discriminative_model.load_state_dict(state_dict)

    discriminative_model.cuda()

    discriminator = DiscriminatorInference(discriminative_tokenizer, discriminative_model)

    return discriminator


def evaluate_answers(input_file, output_file, discriminator):

    with open(output_file, 'w', encoding='utf-8') as g:
        with open(input_file, 'r', encoding='utf-8') as f:
            for dialog in tqdm.tqdm(f.readlines()):
                dialog = json.loads(dialog)
                scores = list()
                for answer in dialog['predicted_answers']:
                    score = discriminator(dialogs=[dialog['dialog'] + [answer['answer']]]).item()
                    answer['score'] = score
                    scores.append(score)
                dialog['mean_score'] = np.mean(scores)
                dialog['std_score'] = np.std(scores)
                g.write(json.dumps(dialog))
                g.write('\n')

    g.close()
    f.close()


def evaluate_answers_extra(input_file, output_file, discriminator):

    with open(output_file, 'w', encoding='utf-8') as g:
        with open(input_file, 'r', encoding='utf-8') as f:
            for dialog in tqdm.tqdm(f.readlines()):
                dialog = json.loads(dialog)
                pred_params_score = []
                default_params_score = []
                for answer in dialog['answer_pred_params']:
                    pred_params_score.append(discriminator(dialogs=[dialog['dialog'] + [answer]]).item())
                for answer in dialog['answer_default_params']:
                    default_params_score.append(discriminator(dialogs=[dialog['dialog'] + [answer]]).item())

                dialog['score_pred_params'] = np.mean(pred_params_score)
                dialog['score_default_params'] = np.mean(default_params_score)

                g.write(json.dumps(dialog))
                g.write('\n')

    g.close()
    f.close()


def count_stats(input_file, random_type='choice'):

    all_averages = list()
    with open(input_file, 'r', encoding='utf-8') as f:
        dialogs = f.readlines()
        for i in range(10):
            averages = list()

            for dialog in dialogs:
                dialog = json.loads(dialog)
                if random_type == 'sample':
                    responses = random.sample(dialog['predicted_answers'], 5)
                    response = sorted(responses, key=lambda x: x['score'], reverse=True)[0]
                elif random_type == 'choice':
                    response = random.choice(dialog['predicted_answers'])
                score = response['score']
                averages.append(score)

            av_averages = np.mean(averages)
            std_averages = np.std(averages)
            all_averages.append(av_averages)
            print(i, av_averages, std_averages)

        print('\nАгрегация агрегаций:', np.mean(all_averages), '\nStd по агрегациям', np.std(all_averages))


if __name__ == "__main__":
    discriminator = start_discriminative("microsoft/deberta-v3-xsmall", "drive/MyDrive/microsoft-deberta-v3-xsmall_rank_softmax.pt")
    # evaluate_answers("drive/MyDrive/generated_random_params_big.jsonl", "drive/MyDrive/generated_random_params_big_scores.jsonl", discriminator)
    evaluate_answers_extra('generated_pred_params.jsonl', 'generated_pred_params_scores.jsonl', discriminator)
    # count_stats("generated_scores.txt", "sample")
