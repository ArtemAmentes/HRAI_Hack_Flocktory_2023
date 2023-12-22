# The house of the Rising AI Хакатон 2023 Flocktory Case

## Описание

Наш проект на хакатоне - это инновационное решение в области машинного обучения,
предназначенное для классификации пользователей по полу (мужской или женский)
на основе предоставленных данных. Мы разработали модель машинного обучения, которая
обучена на разнообразных данных и способна точно предсказывать класс пользователя.

![Иллюстрация Мужчина и Женщина](/data/malefemale.png)

## Проведенные эксперименты

Для решения задач проведен обзор международной практики реализации подобных решений. 
Выявлено, что возможно использовать поведение пользователей на сайте для определения 
пола. В то же время, команда проекта решила проверить связь товарных категорий с половой 
принадлежностью и также получила некоторый результат. 

В данном репозитории представлен блок экспериментов в Google Colab (/experiments). 

Основная модель находится hrai_model_transformer.py
Веса для модели находятся в файле best_model.pth. 

Таким образом обучен трансформер на малой выборке. 10% от общей, но показал хорошие результаты. 
Предлагаем эту модель к использованию. 

По ссылке в презентации есть пайплайн для обучения трансформера.


## Как это работает

Модель использует следующую структуру данных для каждого пользователя:

- История покупок
  ['food' 'electronics' 'bank' 'other' 'media' 'travel' 'shoes' 'fashion'
 'entertainment services' 'hypermarket' 'furniture' 'health' 'pets' 'kids'
 'education' 'insurance' 'cosmetics' nan 'gifts' 'luxury'
 'household appliances' 'deal of the day' 'sport' 'software']

- Активность на сайте
  - total_visits - всего посещений сайта
  - average_session - средняя сессия длительность
  - pages_count - количество просмотренных страниц
  - visit_interval - время между повторными визитами

  
На основе этих данных модель проводит анализ и определяет пол пользователя.

## Установка и Запуск

Для установки и запуска нашего решения следуйте приведенным ниже инструкциям:

```bash
git clone [ссылка на репозиторий]
cd [название репозитория]
pip install -r requirements.txt
python hrai_model_transformer.py
```

## Использование
Для использования модели отправьте запрос 
с данными пользователя в формате вектора pytorch
```bash
model_path = 'best_model.pth'  # Путь к сохраненной модели
input_size = 23
num_classes = 2
num_heads = 2
num_layers = 2
new_embed_dim = 24
model = load_model(model_path, input_size, num_classes, num_heads, num_layers, new_embed_dim)

# Пример входного вектора данных
example_input = np.random.rand(input_size)  # Случайный вектор

# Выполнение предсказания
prediction = predict(model, example_input)
print("Результат предсказания:", prediction)
```

## Команда & Контакты
Артем Аментес  artem.amentes@yandex.ru 

Анатолий Ржевский 

Артем Фомкин 






