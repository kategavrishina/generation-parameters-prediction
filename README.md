# generation-parameters-prediction
Code and data provided for the masters's degree "Predicting Parameters For The Generative Dialog Model"


Generation model
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
- output_file: формат jsonl ИСПРАВИТЬ В КОДЕ, каждый диалог в виде
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
- num_dialogs (int): количество диалогов (если не указать, будут обработаны все диалоги в input_file)
- num_answers (int): значение параметра num_return_sequences для модели генерации
- longest (bool): отсортировать диалоги по количеству реплик, начиная с самых длинных
- fixed (bool): предопределенные параметры генерации на всю выборку
- \*\*generation_parameters: параметры генерации, если fixed=True


Discriminator model



Parameters prediction model
- взять empathetic, разбить диалога на контексты, поделить на трейн и тест
- на трейне сгенерировать ответы с рандомными параметрами из заданных массивов
- оценить ответы дискриминатором
- выбрать лучшие 25% ответов = наборов параметров (качество > 0.25-quantile)
- их взять в качестве корпуса для обучения и валидации предиктора
- обучить предиктор
- предсказать параметры для тестовой выборки (которая не участвовала в генерации с рандомными параметрами)
- сгенерировать ответы для тестовой выборки 1) с дефолтными параметрами, 2) с предсказанными параметрами
- оценить ответы 1) дискриминатором, 2) вручную, 3) gpt
