
# Toxic_comments_analisys_tg_bot: 

Телеграмм бот, созданный для решения задачи бинарной классификации: определить, является ли (и насколько) пользовательское сообщение токсичным в отношении собеседника. 

## Как запустить данный проект:

**Для правильной работы бота необходимо создать виртуальное окружение с использованием requirements.txt**, для этого в терминале пишем:

```
git clone https://github.com/trojanof/toxic_comments_analisys_tg_bot #делаем клон этого репозитория себе на комп
cd toxic_comments_analisys_tg_bot #переходим в папку репозитория
python -m venv .venv #создаем новое виртуальное окружение в папке проекта 
pip install -r requirements.txt #устанавливаем необходимые библиотеки в это окружение
```
Создайте себе нового бота в телеграмме с помощью бота [BotFather](https://telegram.me/BotFather) и сохраните полученный токен в файл 
tg_settings.py

Для запуска вашего бота выполните команду в терминале: 
```
python bot.py
```

## Этапы подготовки текста и обучения модели:

- 1.Предобработка текста (удаление: знаков препинания, стоп-слов и пр.) 
- 2.Построение векторного представления текста с помощью TfIDF
- 3.Обучение модели бинарной классификации(LogReg)
- 4.Тестирование и релиз бота

### Папка 'models'
- __mod_vectorizer.pkl__  – файл с векторизатором (TfIDF)
- __model.pkl__  – файл с обученной моделью(LogReg)

### Папка 'jupyter_notebooks'
- __preprocessing.ipynb__ – файл с процессом предобработки датасета (очистка от знаков препинания, стемминг/лемматизиация и прочее)
- __modelling.ipynb__ – файл с процессом обучения моделей и сравнения их качества


