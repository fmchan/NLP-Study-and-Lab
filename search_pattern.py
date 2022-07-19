from pattern.web import cache, URL, PDF, plaintext, find_urls, asynchronous, time, Google, Twitter

print(find_urls('To search anything, go to www.google.com', unique=True))

asyn_req = asynchronous(Google().search, 'artificial intelligence', timeout=4)
while not asyn_req.done:
    time.sleep(0.1)
    print('searching...')

print(asyn_req.value)

print(find_urls(asyn_req.value, unique=True))

google = Google(license=None)
for search_result in google.search('artificial intelligence'):
    print(search_result.url)
    print(search_result.text)

twitter = Twitter()
index = None
for j in range(3):
    for tweet in twitter.search('artificial intelligence', start=index, count=3):
        print(tweet.text)
        index = tweet.id

html_content = URL('https://stackabuse.com/python-for-nlp-introduction-to-the-textblob-library/').download()
cleaned_page = plaintext(html_content.decode('utf-8'))
print(cleaned_page)

pdf_doc = URL('http://demo.clab.cs.cmu.edu/NLP/syllabus_f18.pdf').download()
print(PDF(pdf_doc.decode('utf-8')))

cache.clear()