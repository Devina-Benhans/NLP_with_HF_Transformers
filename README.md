<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Devina Benhans

## My todo : 

### Exercise 1 - Sentiment Analysis

```
# TODO :
from transformers import pipeline
specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
tweet = "I just finished my project using Hugging Face Transformers and I'm so happy with the result! #AI"
result = specific_model(tweet)
print(result)
```
Result : 

```
[{'label': 'LABEL_2', 'score': 0.990740180015564}]
```

Analysis on example 1 : 

Model cardiffnlp/twitter-roberta-base-sentiment berhasil mengklasifikasikan tweet dengan akurasi tinggi sebagai positif. Ini menunjukkan kemampuannya dalam memahami ekspresi emosional secara eksplisit di media sosial, terutama pada kalimat yang menunjukkan kepuasan atau kegembiraan.


### Exercise 2 - Topic Classification


```
# TODO :
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "I love travelling and learning new cultures",
    candidate_labels=["art", "education", "travel"],
)

```

Result : 

```
{'sequence': 'I love travelling and learning new cultures',
 'labels': ['travel', 'education', 'art'],
 'scores': [0.9902300238609314, 0.005778131075203419, 0.003991869743913412]}```
```

Analysis on example 2 : 

Dengan menggunakan model facebook/bart-large-mnli, teks tentang hobi dan budaya berhasil diklasifikasikan sebagai topik "travel" meskipun tidak ada pelatihan sebelumnya terhadap label tersebut. Ini membuktikan kemampuan zero-shot learning dalam mengaitkan teks dengan label yang bersifat semantik.

### Exercise 3 - Text Generation Models

```
# TODO :
generator("Hello, I'm a language model", max_length=30, num_return_sequences=3)
```

Result : 

```
[{'generated_text': 'Hello, I\'m a language model, I want to help you understand how languages work and I\'m thinking about it. So, here\'s a video of a tutorial I created called "The Language of a Language" and it\'s my favorite part of the tutorial.\nWhat started out as an experiment on my own was an idea I didn\'t even know how to write and what I wanted to do to achieve it. Then, I started thinking about how to write and writing.\nThen, I thought of an idea that I had written with a friend and a friend. It wasn\'t a complete solution, but it was easy to see why I wanted to do it. So, I created\nThe Language of a Language by David Campbell.\nDavid Campbell is a program manager, software engineer, and programmer. He has written over 10 million languages and is the founder of the Language of a Language.\nWhat is the language you see as your language?\nI\'m a language model. I want to help you understand how languages work and I\'m thinking about it. So, here\'s a video of a tutorial I created called "The Language of a Language" and it\'s my favorite part of the tutorial.\nWhat is the language you see as your language?\nI\'m a language model. I'}

```

Analysis on example 3 : 

Model teks mampu menghasilkan lanjutan kalimat yang koheren dan panjang dari prompt awal. Meskipun beberapa bagian terlihat repetitif atau tidak logis, hasilnya tetap menggambarkan potensi GPT-based models dalam menciptakan teks naratif atau kreatif secara otomatis.

### Exercise 4 - Name Entity Recognition

```
# TODO :
ner = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)
text = "Her name is Anjela and she lives in Seoul."
ner_results = ner(text)
print(ner_results)
```

Result : 

```
[{'entity_group': 'PER', 'score': np.float32(0.9481442), 'word': 'Anjela', 'start': 11, 'end': 18}, {'entity_group': 'LOC', 'score': np.float32(0.9986114), 'word': 'Seoul', 'start': 35, 'end': 41}]
```

Analysis on example 4 : 

Dengan model camembert-ner, entitas seperti nama orang dan lokasi dapat dikenali secara akurat dari kalimat pendek. Ini menunjukkan keandalan model dalam melakukan ekstraksi informasi dasar, cocok untuk aplikasi seperti analisis berita atau chatbot.

### Exercise 5 - Question Answering

```
# TODO :
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
qa({
    "question": "Which lake is one of the five Great Lakes of North America?",
    "context": "Lake Ontario is one of the five Great Lakes of North America. It is surrounded on the north, west, and southwest by the Canadian province of Ontario, and on the south and east by the U.S. state of New York, whose water boundaries, along the international border, meet in the middle of the lake."
})

```

Result : 

```
{'score': 0.9834363460540771, 'start': 0, 'end': 12, 'answer': 'Lake Ontario'}

```

Analysis on example 5 : 

Pipeline QA berbasis distilbert berhasil menjawab pertanyaan dengan tepat dari konteks yang diberikan, dengan skor keyakinan yang sangat tinggi. Ini menegaskan kemampuannya dalam memahami konteks dan menemukan jawaban spesifik secara efisien.


### Exercise 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", max_length=59)
summarizer(
    """
Lake Superior in central North America is the largest freshwater lake in the world by surface area and the third-largest by volume, holding 10% of the world's surface fresh water. The northern and westernmost of the Great Lakes of North America, it straddles the Canada–United States border with the province of Ontario to the north, and the states of Minnesota to the northwest and Wisconsin and Michigan to the south. It drains into Lake Huron via St. Marys River and through the lower Great Lakes to the St. Lawrence River and the Atlantic Ocean.
"""
```

Result : 

```
[{'summary_text': " Lake Superior is the largest freshwater lake in the world by surface area . It holds 10% of the world's surface fresh water . It straddles the Canada–U.S. border with the province of Ontario to the north . It drains into Lake Huron via St. Marys River and through the lower Great Lakes to the St. Lawrence River and the Atlantic Ocean ."}]
```

Analysis on example 6 :

Model distilbart merangkum paragraf panjang menjadi versi singkat tanpa kehilangan informasi penting seperti lokasi dan statistik utama. Hal ini menunjukkan efisiensi model dalam menyederhanakan teks informatif, berguna dalam dunia jurnalistik dan dokumentasi.

### 7. Example 7 - Translation

```
# TODO :
translator = pipeline("translation_en_to_de", model="t5-small")
print(translator("New York is my favourite city", max_length=40))
```

Result : 

```
[{'translation_text': 'New York ist meine Lieblingsstadt'}]

```

Analysis on example 7 :

Model t5-small mampu menerjemahkan kalimat bahasa Inggris ke bahasa Jerman dengan benar secara gramatikal dan konteks. Meski ringan, model ini efektif dalam komunikasi lintas bahasa yang cepat dan informal.


---

## Analysis on this project

This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems.
