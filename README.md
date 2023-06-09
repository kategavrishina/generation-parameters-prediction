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
  "real_answer": "turn 4"
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

## Модель для предсказания параметров
### Предобработка
**preprocessing.ipynb**
- взять empathetic, разбить диалоги на контексты (контекст = 1 не считается!!), поделить на трейн и тест
- на трейне сгенерировать ответы с рандомными параметрами из заданных массивов (generator_script.py: fixed=False, random_params=True)
- оценить ответы дискриминатором (discriminator_script.py)
- выбрать лучшие 25% ответов = наборов параметров (качество > 0.25-quantile + условие на std и кол-во ответов) и преобразовать значения параметров в номера классов (preprocessing.ipynb) и их взять в качестве корпуса для обучения и валидации предиктора
### Обучение модели
**parameters_predictor.ipynb**
- обучить предиктор и предсказать параметры для тестовой выборки, которая не участвовала в генерации с рандомными параметрами
### Постобработка
**postprocessing.ipynb**
- сгенерировать ответы для тестовой выборки 1) с дефолтными параметрами, 2) с предсказанными параметрами (generator_script.py: fixed=False, random_params=False)
- оценить ответы 1) дискриминатором, 2) вручную, 3) gpt

## Данные
- emp_train.jsonl и emp_test.jsonl
- generated_random_params_big_scores.jsonl сгенерированные ответы по рандомным наборам параметров
- final_big_embedded_good.jsonl датасет для обучения предиктора
- generated_pred_params.jsonl сгенерированные ответы по предсказанным и дефолтным параметрам
- generated_pred_params_scores.jsonl оцененные ответы по предсказанным и дефолтным параметрам
