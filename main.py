import nltk
import pandas as pd
import matplotlib.pyplot as plt
import string
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

counter1 = 1

messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
messages['length'] = messages['message'].apply(len)

'''Most of the messages are ham'''
# sns.countplot(messages['label'])
# plt.show()

'''Most of the messages are very lengthy'''
# messages['length'].plot.hist(bins=50)
# plt.show()

'''Spam messages have an average length of 150 characters'''
'''Ham messages have a relatively small length'''
# messages.hist(column='length', by='label', bins=80, figsize=(12,4))
# plt.show()

# '''remove punctuations from the messages'''
# def remove_punc(message):
# 	global counter1
# 	print('Counter 1::', counter1)
# 	counter1 += 1
# 	return ''.join([word for word in message if word not in string.punctuation])

# messages['message'] = messages['message'].apply(remove_punc)


# '''remove filler words from the messages'''
# def remove_filler(message):
# 	return ' '.join([word for word in message.split() if word not in stopwords.words('english')])

# messages['message'] = messages['message'].apply(remove_filler)

# '''Remove Stemmers from messages'''
# def remove_stem(message):
# 	return ' '.join(list(set([SnowballStemmer('english', ignore_stopwords=True).stem(stem) for stem in message.split()])))


def filter_data(message):
    global counter1
    print('Counter::', counter1)
    counter1 += 1
    message = ''.join([word for word in message if word not in string.punctuation])
    message = ' '.join([word for word in message.split() if word.lower() not in stopwords.words('english')])
    message = [SnowballStemmer('english', ignore_stopwords=True).stem(stem) for stem in message.split()]
    return message


messages['message'].apply(filter_data)

counter1 = 1

bag_of_words = CountVectorizer(analyzer=filter_data).fit(messages['message'])

print('Length of Vocab::', len(bag_of_words.vocabulary_))

sent = bag_of_words.transform(['i am a test message and i also have a good grammar',
                               'me have bad english sorry 4 this mistake me try best ugandawordnomeaning'])

print(sent.shape)
print(sent)
