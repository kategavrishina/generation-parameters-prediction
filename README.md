# Предсказание параметров генерации ответа диалоговой модели
Code and data provided for the masters's degree "Predicting Parameters For The Generative Dialog Model"\
HSE University, Faculty of Humanities, educational program "Computational Linguistics"\
Moscow, 2023

Код и данные для магистерской диссертации "Предсказание параметров генерации ответа диалоговой модели"\
НИУ ВШЭ, факультет гуманитарных наук, образовательная программа "Компьютерная лингвистика"\
Москва, 2023

## Генеративная модель
**generator_script.py**
- input_file: формат jsonl, каждый диалог в виде 
```json
{
  "dialog": [
      "turn 1",
      "turn 2",
      "turn 3"
  ],
  "real_answer": "turn 4"
}
```
- output_file: формат jsonl, каждый диалог в виде
```json
{
  "dialog": [
      "turn 1",
      "turn 2",
      "turn 3"
  ],
  "real_answer": "turn 4",
  "predicted_answers": [
      {"answer": "answer 1"},
      {"answer": "answer 2"}, 
      {"answer": "answer 3"}
  ],
  "params": {
      "top_p": "top_p",
      "top_k": "top_k",
      "typical_p": "typical_p",
      "temperature": "temperature",
      "do_sample": "True"
  },
  "change_type": "change_type"
}
```
- generator: инициализированная модель
- num_dialogs (int, None): количество диалогов (если не указать, будут обработаны все диалоги в input_file)
- num_answers (int, 15): значение параметра num_return_sequences для модели генерации
- longest (bool, True): отсортировать диалоги по количеству реплик, начиная с самых длинных
- fixed (bool, False): предопределенные параметры генерации на всю выборку, если True
- random_params (bool, False): сэмплировать значения параметров для каждого диалога, если True
- \*\*generation_parameters: параметры генерации, если fixed=True

**Пример вызова функции:**
```
generator = start_generator("allenai/cosmo-xl")
generate_answers("emp_train.jsonl", "generated_random_params_big.txt", generator, num_answers=5, random_params=True)
```
**Пример запуска скрипта:**
```
! pip install -r requirements.txt
! python generator_script.py
```

## Модель для оценивания сгенерированных ответов
**discriminator_script.py**

1) evaluate_answers()
- input_file: формат jsonl, каждый диалог в виде
```json
{
  "dialog": [
      "turn 1",
      "turn 2",
      "turn 3"
  ],
  "real_answer": "turn 4",
  "predicted_answers": [
      {"answer": "answer 1"},
      {"answer": "answer 2"}, 
      {"answer": "answer 3"}
  ],
  "params": {
      "top_p": "top_p",
      "top_k": "top_k",
      "typical_p": "typical_p",
      "temperature": "temperature",
      "do_sample": "True"
  },
  "change_type": "change_type"
}
```
- output_file: формат jsonl, каждый диалог в виде
```json
{
  "dialog": [
      "turn 1",
      "turn 2",
      "turn 3"
  ],
  "real_answer": "turn 4",
  "predicted_answers": [
      {"answer": "answer 1", "score": "score"},
      {"answer": "answer 2", "score": "score"}, 
      {"answer": "answer 3", "score": "score"}
  ],
  "params": {
      "top_p": "top_p",
      "top_k": "top_k",
      "typical_p": "typical_p",
      "temperature": "temperature",
      "do_sample": "True"
  },
  "change_type": "change_type",
  "mean_score": "mean_score",
  "std_score": "std_score"
}
```
- discriminator: инициализированная модель
2) evaluate_answers_extra() -- для оценивания ответов с дефолтными и предсказанными параметрами
- input_file: формат jsonl, каждый диалог в виде
```json
{
  "dialog": [
      "turn 1",
      "turn 2",
      "turn 3"
  ],
  "answer_pred_params": [
      "answer 1",
      "answer 2",
      "answer 3"
  ],
  "answer_default_params": [
      "answer 1",
      "answer 2",
      "answer 3"
  ],
  "pred_params": {
      "top_p": "top_p",
      "top_k": "top_k",
      "typical_p": "typical_p",
      "temperature": "temperature",
      "do_sample": "True"
  }
}
```
- output_file: формат jsonl, каждый диалог в виде
```json
{
  "dialog": [
      "turn 1",
      "turn 2",
      "turn 3"
  ],
  "answer_pred_params": [
      "answer 1",
      "answer 2",
      "answer 3"
  ],
  "answer_default_params": [
      "answer 1",
      "answer 2",
      "answer 3"
  ],
  "pred_params": {
      "top_p": "top_p",
      "top_k": "top_k",
      "typical_p": "typical_p",
      "temperature": "temperature",
      "do_sample": "True"
  },
  "score_pred_params": "score_pred_params",
  "score_default_params": "score_default_params"
}
```
- discriminator: инициализированная модель
3) count_stats()
- input_file: формат jsonl вида output_file функции evaluate_answers()
- random_type ('choice'/'sample'): либо среди предсказанных ответов выбирается один случайный (и его оценка), либо сначала сэмплируются случайные 5 и из них выбирается ответ с наибольшей оценкой

**Пример вызова функций:**
```
discriminator = start_discriminative("microsoft/deberta-v3-xsmall", "microsoft-deberta-v3-xsmall_rank_softmax.pt")
evaluate_answers("generated_random_params_big.jsonl", "generated_random_params_big_scores.jsonl", discriminator)
evaluate_answers_extra('generated_pred_params.jsonl', 'generated_pred_params_scores.jsonl', discriminator)
```
**Пример запуска скрипта:**
```
! pip install -r requirements.txt
! python discriminator_script.py
```

## Модель для предсказания параметров
### Предобработка
1) разбиение диалогов на контексты, деление на тренировочную и тестовую выборки
2) генерация ответов с рандомными параметрами из заданных массивов (generator_script.py: generate_answers(fixed=False, random_params=True))
3) оценивание ответов дискриминатором (discriminator_script.py: evaluate_answers())\
**preprocessing.ipynb**
4) преобразование значений параметров в номера классов
5) выбор примеров по ограничениям: качество >= 0.5-quantile & количество ответов >= 3 & std >= 0.03
### Обучение модели
**parameters_predictor.ipynb**
1) инициализация предиктора
2) оптимизация параметров предиктора при помощи Optuna
3) обучение с лучшими параметрами
4) предсказание параметров генерации на тестовой выборке
5) модификация модели с добавлением предсказания типа изменения значений параметров (change_type)
6) предсказание параметров генерации на тестовой выборке обновленной модели
7) преобразование номеров классов параметров в реальные значения
### Постобработка
1) генерация ответов для тестовой выборки 1) с дефолтными параметрами, 2) с предсказанными параметрами (generator_script.py: generate_answers(fixed=False, random_params=False)\
**postprocessing.ipynb**
2) сэмплирование примеров (200) и подготовка файла для экспертной оценки
3) оценивание результатов разметки
4) результаты оценивания ответов дискриминатором

## Данные
- emp_train.jsonl и emp_test.jsonl: обучающая и тестовая выборки (полученные в результате 1-го шага предобработки)
- generated_random_params_big_scores.jsonl: ответы, сгенерированные по рандомным наборам параметров, и их оценки, проставленные дискриминатором (полученные в результате 3-го шага предобработки)
- final_big_embedded_good.jsonl: датасет для обучения предиктора (полученный в результате предобработки)
- generated_pred_params.jsonl: ответы, сгенерированные по предсказанным и дефолтным параметрам (полученные в результате 1-го шага постобработки)
- generated_pred_params_scores.jsonl: оцененные ответы по предсказанным и дефолтным параметрам (полученные в результате постобработки)
