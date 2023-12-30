import pandas as pd
import tiktoken, os, datetime
from openai import OpenAI
import random
import string

import numpy as np
#from openai.embeddings_utils import distances_from_embeddings
# Inspiration : https://github.com/openai/web-crawl-q-and-a-example/blob/main/web-qa.ipynb

#free:50, 
STATIC_FILE_DIR = '/tmp/'

client = OpenAI(
    api_key="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",
)

def generate_random_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    file_name = f"{timestamp}_{random_string}"

    return file_name

class ChatModel:

    def remove_newlines(self, serie):
        try:
            serie = serie.str.replace('\n', ' ')
            serie = serie.str.replace('\\n', ' ')
            serie = serie.str.replace('  ', ' ')
            serie = serie.str.replace('  ', ' ')

            return serie
        except Exception as e:
            print(serie)
            return serie

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_embedding(self, text, model="text-embedding-ada-002"): # model = "deployment_name"
        return client.embeddings.create(input = [text], model=model).data[0].embedding

        #res = search_docs(df_bills, "Can I get information on cable company tax revenue?", top_n=4)

    # def create_scrapped_csv(self, file_name):
    #     texts=[]
            
    #     with open(file_name, "r", encoding="UTF-8") as f:
    #         text = f.read()
    #         texts.append(('title', text))

    #     self.df = pd.DataFrame(texts, columns = ['fname', 'text'])

    #     self.df['text'] = self.remove_newlines(self.df.text)

    #     self.path_scrapped_data = STATIC_FILE_DIR + generate_random_filename()+".csv"
    #     self.df.to_csv(self.path_scrapped_data)
        
    #     # print(self.df.to_string(index=False))
    #     self.df.head()

    def create_and_save_df(self, file_name, words_per_chunk=1000):
        chunks = []

        with open(file_name, "r", encoding="UTF-8") as f:
            input_string = f.read()
            # chunks = [input_string[i:i + chunk_size] for i in range(0, len(input_string), chunk_size)]

            words = input_string.split()
            chunks = [words[i:i + words_per_chunk] for i in range(0, len(words), words_per_chunk)]
        
        texts = [('title', ' '.join(chunk), ' '.join(chunk)) for chunk in chunks]

        #texts = [('title', chunk) for chunk in chunks]
        
        self.df = pd.DataFrame(texts, columns=['fname', 'text', 'context_data'])

        self.df['text'] = self.remove_newlines(self.df.text)
        self.path_scrapped_data = STATIC_FILE_DIR + "scrapped_"+generate_random_filename()+".csv"

        self.df.to_csv(self.path_scrapped_data, index=False)
        self.df.head()

        return self.df

    def load_tokenizer(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.df = pd.read_csv(self.path_scrapped_data, index_col=0)
        # self.df.columns = ['fname', 'text']
        self.df['n_tokens'] = self.df.text.apply(lambda x: len(self.tokenizer.encode(x)))

        # self.df.n_tokens.hist()

    def split_into_many(self, text, max_tokens=1000):

        sentences = text.split('. ')

        n_tokens = [len(self.tokenizer.encode(" " + sentence)) for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        for sentence, token in zip(sentences, n_tokens):

            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

           
            if token > max_tokens:
                continue

            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks

    def generate_embeddings(self, text, model="text-embedding-ada-002"): # model = "deployment_name"
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    def dataframe_tokenization(self, max_tokens=1000):
        shortened = []
        context_data = self.df['context_data'].tolist()
        for row in self.df.iterrows():
            if row[1]['text'] is None:
                continue

            if row[1]['n_tokens'] > max_tokens:
                shortened += self.split_into_many(row[1]['text'])

            else:
                shortened.append(row[1]['text'])


        self.df = pd.DataFrame(shortened, columns = ['text'])
        # self.df['context_data'] = context_data
        self.df['n_tokens'] = self.df.text.apply(lambda x: len(self.tokenizer.encode(x)))
        # self.df.n_tokens.hist()

        self.df['embeddings'] = self.df.text.apply(lambda x : self.generate_embeddings (x, model = 'text-embedding-ada-002'))
        #self.df['context_data'])

        self.embedding_file_name = STATIC_FILE_DIR + generate_random_filename() + ".csv"
        self.df.to_csv(self.embedding_file_name)
        self.df.head()
        return self.embedding_file_name
    
    def read_embedding_file(self, embedding_file_path):
        self.df=pd.read_csv(embedding_file_path, index_col=0)
        self.df['embeddings'] = self.df['embeddings'].apply(eval).apply(np.array)

        self.df.head()
    
    def search_docs(self, user_query, top_n=4):
        embedding = self.get_embedding(
            user_query,
            model="text-embedding-ada-002" # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
        )
        self.df["similarities"] = self.df.embeddings.apply(lambda x: self.cosine_similarity(x, embedding))

        res = (
            self.df.sort_values("similarities", ascending=False)
            # .head(top_n)
        )
        return res
    
    def train_model(self, training_data_file_name):
        self.create_and_save_df(training_data_file_name)
        #self.create_scrapped_csv(training_data_file_name)
        self.load_tokenizer()
        return self.dataframe_tokenization()
    
    def ask_chatbot(self, token_file_path, question):
        self.read_embedding_file(token_file_path)
        self.search_docs(question)
        return self.create_context()
    
    def create_context(self, max_len=4000):

        returns = []
        cur_len = 0

        for i, row in self.df.sort_values('similarities', ascending=True).iterrows():

            cur_len += row['n_tokens'] + 4

            if cur_len > max_len:
                break

            returns.append(row["text"])

        return "\n".join(returns)
    
    def ask_chatgpt(self, context, question, model='gpt-3.5-turbo', max_tokens=150):
        try:
            print(context)
            messages=[
                    # {"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
                    {"role": "user", f"content": "Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
                ]
            # Create a chat completion using the question and context
            response = client.chat.completions.create(
                model=model,
                messages=messages,               
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message
        except Exception as e:
            print(e)
            return ""




#print(ChatModel().train_model('/Users/deepakkumar/Downloads/sync_backend/tmp_scrap.txt'))
# question = 'who are you'

# context = ChatModel().ask_chatbot('tmp/tr.csv', question)
# print(context)
# print(ChatModel().ask_chatgpt(context=context, question=question))

# answer_question(df, question="What day is it?", debug=False)

# answer_question(df, question="What is our newest embeddings model?")

# answer_question(df, question="What is ChatGPT?")


class OldChatGPT:
    def create_context(self, question, df, max_len=1800, size="ada"):
        q_embeddings = client.embeddings.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

        self.df['distances'] = distances_from_embeddings(q_embeddings, self.df['embeddings'].values, distance_metric='cosine')

        returns = []
        cur_len = 0

        for i, row in self.df.sort_values('distances', ascending=True).iterrows():

            cur_len += row['n_tokens'] + 4

            if cur_len > max_len:
                break

            returns.append(row["text"])

        return "\n\n###\n\n".join(returns)

    def answer_question(self, df, model="gpt-3.5-turbo",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=150,
        stop_sequence=None
    ):
        """
        Answer a question based on the most similar context from the dataframe texts
        """
        context = self.create_context(
            question,
            self.df,
            max_len=max_len,
            size=size,
        )
        # If debug, print the raw model response
        if debug:
            print("Context:\n" + context)
            print("\n\n")

        try:
            # Create a chat completion using the question and context
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
                    {"role": "user", f"content": "Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
                ],
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
            )
            return response.choices[0].message.strip()
        except Exception as e:
            print(e)
            return ""