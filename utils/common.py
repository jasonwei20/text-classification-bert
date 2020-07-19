
def get_sentences_and_labels_from_txt(dataset_path):
    lines = open(dataset_path).readlines()
    sentences = []
    labels = []
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        sentence = parts[1]
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels