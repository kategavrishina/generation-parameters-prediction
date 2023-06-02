# generation-parameters-prediction
Code and data provided for the masters's degree "Predicting Parameters For The Generative Dialog Model"


Generation model
- input_file: формат jsonl, каждый диалог в виде 
```json
{
  'dialog': [
      'turn 1',
      'turn 2',
      'turn 3'
  ],
  'real_answer': 'turn 4' \\optional
}
```
- output_file: формат jsonl ИСПРАВИТЬ В КОДЕ, каждый диалог в виде
```json
{
  'dialog': [
      'turn 1',
      'turn 2',
      'turn 3'
  ],
  'real_answer': 'turn 4', \\optional
  'predicted_answers': [
      {'answer': 'answer 1'},
      {'answer': 'answer 2'}, 
      {'answer': 'answer 3'}
  ],
  'params': {
      'top_p': top_p,
      'top_k': top_k,
      'typical_p': typical_p,
      'temperature': temperature,
      'do_sample': True
  },
  'change_type': change_type
}
```
- generator
- num_dialogs=5000
- num_answers=15
- longest=True
- fixed=True, \**generation_parameters


Discriminator model



Parameters prediction model
- взять empathetic, разбить диалога на контексты, поделить на трейн и тест
- на трейне сгенерировать ответы с рандомными параметрами из заданных массивов
- оценить ответы дискриминатором
- выбрать лучшие 25% ответов = наборов параметров (качество > 0.25-quantile)
- 
