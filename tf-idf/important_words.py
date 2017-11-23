"""
TF-IDF stands for 'Term Frequency, Inverse Document Frequency'. It is a way to score the importance of words (or 'items) in a document based on how frequently
they appear accross multiple documents.

Intuitively....
1) If a word appears frequently in a document, it's important. Give the word a high score.
2) But if a word appears in many documents, it's not a unique identifier. Give the word a low score.
"""
import math
from textblob import TextBlob as tb

"""
Computes term frequency which is the number of times a word appears in a document, normalized by dividing by the total number of
words in document.
"""
def tf(word, document):
    return document.words.count(word) / len(document.words)

"""
Return the number of documents containing word.
"""
def n_containing(word, documents):
    return sum(1 for document in documents if word in document.words)

"""
Measures how common a word is among in all documents. The more common word is, the lower its idf.
"""
def idf(word, documents):
    return math.log(len(documents) / (1 + n_containing(word, documents)))

"""
Computes the IF-IDF score.
"""
def tfidf(word, document, documents):
    return tf(word, document) * idf(word, documents)

document1 = tb("""Python is a 2000 made-for-TV horror movie directed by Richard
Clabaugh. The film features several cult favorite actors, including William
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean
Whalen. The film concerns a genetically engineered snake, a python, that
escapes and unleashes itself on a small town. It includes the classic final
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles,
 California and Malibu, California. Python was followed by two sequels: Python
 II (2002) and Boa vs. Python (2004), both also made-for-TV films.""")

document2 = tb("""Python, from the Greek word (πύθων/πύθωνας), is a genus of
nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are
recognised.[2] A member of this genus, P. reticulatus, is among the longest
snakes known.""")

document3 = tb("""The Colt Python is a .357 Magnum caliber revolver formerly
manufactured by Colt's Manufacturing Company of Hartford, Connecticut.
It is sometimes referred to as a "Combat Magnum".[1] It was first introduced
in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued
Colt Python targeted the premium revolver market segment. Some firearm
collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy
Thompson, Renee Smeets and Martin Dougherty have described the Python as the
finest production revolver ever made.""")

documents = [document1, document2, document3]

for i, document in enumerate(documents):
    print("Top words in document {}".format(i+1))
    scores = {word : tfidf(word, document, documents) for word in document.words}
    sorted_words = sorted(scores.items(), key = lambda x: x[1], reverse=True)

    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score,5)))

