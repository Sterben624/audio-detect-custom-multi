## **ShahedNetDeepWide**
Модель містить: мелспектрограму, 4 згорткові шари з BatchNorm, MaxPool та Dropout, SE-блок уваги,
перетворення тензора для GRU, двонапрямлений GRУ і фінальний лінійний шар.

* MelSpectrogram + AmplitudeToDB
* 4 Conv2d шари з BatchNorm2d, ReLU, MaxPool2d і Dropout2d
* SE-блок (Squeeze-and-Excitation)
* Перетворення тензора для послідовного вводу в RNN
* Двонапрямлений GRU шар
* Dropout
* Лінійний (Fully Connected) шар для класифікації

Пов'язані файли:
* train_deep_wide.py - навчання + фокус на клас shahed
* test_model_deep.py - файл для тестування
* shahednet_best_deep_wide.pth - ваги після 400 епох

## ShahedNet
Модель містить: мелспектрограму, 3 згорткові шари з BatchNorm, ReLU, MaxPool, AdaptiveAvgPool, Dropout,
фінальний лінійний шар.

* MelSpectrogram + AmplitudeToDB
* 3 Conv2d шари з BatchNorm2d, ReLU, MaxPool2d (для перших двох шарів)
* AdaptiveAvgPool2d (GAP)
* Dropout
* Лінійний (Fully Connected) шар для класифікації

Пов'язані файли:
* train_focus_shahed.py - тренування + фокус на shahed
* train.py - тренування
* test_model.py - тестування
* shahednet_best_light.pth - ваги після 600 епох

## ShahedDataset
Клас ShahedDataset відповідає за завантаження аудіоданих і відповідних міток для навчання або валідації моделі. Він реалізує інтерфейс Dataset бібліотеки PyTorch.

Основна функціональність:
* Ініціалізація з шляхом до CSV-файлу з метаданими, директорією з аудіофайлами, словником відповідності міток до індексів та цільовою частотою дискретизації.
* Метод len повертає кількість зразків у датасеті.
* Метод getitem за індексом завантажує аудіофайл, за потреби виконує ресемплінг до цільової частоти дискретизації.
* Конвертує рядок міток (може містити кілька класів, розділених комами) у multi-hot вектор відповідної довжини.
* Повертає тензор аудіосигналу (одноканальний) та multi-hot вектор міток.

## Utils
* utils/audio_utils.py — функції завантаження, обрізання аудіо та передбачення класу (не використовується).
* utils/convert_44khz_to_16khz.py — конвертація аудіо з 44 кГц у 16 кГц і моно.
* utils/divide.py — розподіл датасету на тренувальний і валідаційний набори.
* utils/generator_augmentation_audio.py — генерація аугментацій аудіо з pitch shift, шумом і фільтрацією.
* utils/mixing_audio.py — створення міксованих аудіозаписів і формування CSV з мітками.
* utils/prepare_audio.py — нарізка аудіо на фрагменти з перекриттям та розподіл на train/valid (не використовується, залишилось з проєкту multi клас, зараз multilabel).
* utils/rename_v.py — перейменування файлів у валідаційних папках із додаванням символу «v».

## Директорії
* audio_for_test - аудіо для тестування.
* classes - класи ShahedNetDeepWide, ShahedNet, ShahedDataset.
* dataset/custom_16k - сирі аудіо в 16 кГц моно. (основний)
* dataset/custom_44k - сирі аудіо в 44 кГц мікс. (сміття)
* dataset/mixed_train - аудіо по 1 с для тренування multi labels + labels.csv.
* dataset/valid_train - аудіо по 1 с для валідації multi labels + labels.csv.
* dataset/to_train - також відносно сирі аудіо 16 кГц моно 1 с. (основний)

## Старт
`conda env activate audio-detect-custom`