from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
classifier = pipeline('sentiment-analysis')
x = classifier('We are not very happy to introduce pipeline to the transformers repository.')

print(x)

# [{'label': 'POSITIVE', 'score': 0.9996980428695679}]

# Allocate a pipeline for question-answering
question_answerer = pipeline('question-answering')

x = question_answerer({
    'question': 'What is the name of the repository ?',
    'context': 'Pipeline has been included in the huggingface/transformers repository'
})

print(x)

print("done")
