import glob
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math
stop_words = set(stopwords.words('english'))
stop_words.remove("in")
stop_words.remove("where")
stop_words.remove("to")

term_list = {}
docs = 0
# term_list = {freq , number of docs containing term ( df ) ,[docs] ,{ every doc with positions } }


def tokenization():
    my_files = glob.glob('txt files/*.txt')
    my_files.append(my_files.pop(my_files.index('txt files\\10.txt')))
    first_tokens = []
    print(my_files)
    for file in my_files:
        read = open(file, "r")
        words = read.read()
        tokens = word_tokenize(words.lower())
        first_tokens.append(tokens)
    return first_tokens


def remove_stop_words(first_tokens):
    second_tokens = []
    sec_tokens = []
    for tokens in first_tokens:
        for word in tokens:
            if word.lower() not in stop_words:
                sec_tokens.append(word.lower())
        second_tokens.append(sec_tokens)
        sec_tokens = []
    return second_tokens


def positional_index_2(final_tokens_list):
    doc_id = 1
    for file in final_tokens_list:
        index = 1
        for word in file:
            if word in term_list:
                term_list[word][0] += 1
                if doc_id not in term_list[word][2]:
                    term_list[word][2].append(doc_id)
                    term_list[word][1] += 1
                    term_list[word][3][doc_id] = [index]
                else:
                    term_list[word][3][doc_id].append(index)
            else:
                term_list[word] = []
                term_list[word].append(1)
                term_list[word].append(1)
                term_list[word].append([doc_id])
                term_list[word].append({doc_id: [index]})
                # term_list[word].append([doc_id,index])
            index += 1
        doc_id += 1
    # print(term_list)
    return term_list


def compute_term_frequency(positional_index_list):
    my_files = glob.glob('txt files/*.txt')
    my_files.append(my_files.pop(my_files.index('txt files\\10.txt')))
    number_of_docs = 0
    for _ in my_files:
        number_of_docs += 1
    for term in positional_index_list:
        for key, value in positional_index_list[term][3].items():
            counter = 0
            for _ in value:
                counter += 1
            print("The frequency of : " + term + " in document : " + str(key) + " is → " + str(counter))
        idf = math.log10(number_of_docs / positional_index_list[term][1])
        print("IDF for : " + term + " is → " + str(idf))
        print("-------------------------------------------------------------------------------------")


def compute_tf_idf_matrix(positional_index_list):
    print("--------------------------------------------------------------------")
    my_files = glob.glob('txt files/*.txt')
    my_files.append(my_files.pop(my_files.index('txt files\\10.txt')))
    number_of_docs = 0
    values = []
    for _ in my_files:
        number_of_docs += 1
        values.append(0)
    print("term | ", end=" \t\t\t")
    for x in range(number_of_docs):
        print("doc " + str(x+1) + " | ", end=" \t")
    print(" ")
    print("-----------------------------------------------------------------")
    for term in positional_index_list:
        for x in range(number_of_docs):
            values[x] = 0
        print(term + " | ", end=" \t\t\t")
        # key is the number of the doc
        for key, value in positional_index_list[term][3].items():
            counter = 0
            for _ in value:
                counter += 1
            tf = 1 + math.log10(counter)
            idf = math.log10(number_of_docs / positional_index_list[term][1])
            tf_idf = tf * idf
            # format_tf_idf = "{:.2f}".format(tf_idf)
            values[key-1] = tf_idf
        for x in values:
            x = "{:.2f}".format(x)
            print(str(x) + " | ", end=" \t")
        print(" ")
        print("-----------------------------------------------------------------")


def ph_query(user_query, positional_index_list):
    final_shared_docs = []  # will contain the docs that contain all word of query in the same doc
    query_split = user_query.split()
    new_query = []  # will contain the query
    ans = []  # will contain all docs contain any word of the docs
    shared_ans = []
    real_ans = []  # will contain the docs that matching the query
    for word in query_split:
        if word.lower() not in stop_words:
            new_query.append(word.lower())   # tokenization and remove stop words of query
    for w in new_query:
        if w not in positional_index_list:
            print("NO DOCUMENTS FOUND MATCHING THIS QUERY")  # if word not in the positional index
            return 0
        else:
            for key, value in positional_index_list[w][3].items():  # loop for docs contain the current word
                if key in ans:  # check if we added this doc to ans before
                    if key not in shared_ans:  # if true check we didn't add it to shared ans
                        shared_ans.append(key)  # if true add it to shared ans
                else:
                    ans.append(key)  # add the doc to ans
    if len(new_query) == 1:
        real_ans = ans
    else:
        for doc in shared_ans:  # loop on shared docs
            not_found = 0  # will change to 1 if the current doc doesnt contain any word of the query
            for w in new_query:  # loop on the words of the query
                if doc not in positional_index_list[w][3]:  # check if the current doc not in the docs contain the word
                    not_found = 1  # if true change not_found to 1 and skip the rest of the loop
                    continue
            if not_found != 1:
                final_shared_docs.append(doc)  # add the doc to final shared docs

        for doc in final_shared_docs:  # loop on the docs in final shared docs
            counter = len(positional_index_list[new_query[0]][3][doc])  # number of positions of first word in query
            i = 0
            while i < counter:
                positions = [positional_index_list[new_query[0]][3][doc][i]]  # position of first word in query in the list
                not_in_ans = 0  # change to 1 if two words of query not next to each other in doc
                for w in new_query:  # loop for words in the query
                    if w == new_query[0]:  # if the word is the first word skip
                        continue
                    else:
                        for d in positional_index_list[w][3][doc]:  # if the word has many pos in the doc
                            if not_in_ans == 1:  # check if not_in_ans is 1 before enter the loop
                                break  # break if not_in_ans == 1
                            if d not in positions:
                                if not positions:  # [1] never happen
                                    positions.append(d)
                                else:
                                    if d == positions[-1] + 1:  # check if the pos of the current word = last pos in list+1
                                        positions.append(d)  # add the pos to the list = [1,2]
                                        break  # skip the test of the loop
                                    else:
                                        if d != positions[-1] + 1 and positions[-1] + 1 not in positional_index_list[w][3][doc]:
                                            not_in_ans += 1  # if the current pos not good check the last ele +1
                                            # if a pos for the word
                if len(positions) == len(new_query):  # if length of pos list = length of the query
                    break
                i += 1
            if not_in_ans == 0:  # the doc will be an answer if not_in_us remains 0
                real_ans.append(doc)  # add the doc to the answer list
    if not real_ans:
        print("NO DOCUMENTS FOUND MATCHING THIS QUERY")
        return 0
    else:
        return real_ans


def print_matched_docs(real_ans):
    for doc in real_ans:
        print("DOCUMENT : ", doc, " IS MATCHING YOUR QUERY . ")


def unit_query(u_query, real_ans, positional_index_list):
    my_files = glob.glob('txt files/*.txt')
    my_files.append(my_files.pop(my_files.index('txt files\\10.txt')))
    number_of_docs = 0
    for _ in my_files:
        number_of_docs += 1
    query_split = u_query.split()
    new_query = query_split  # will contain the query
    sim = {}
    for doc in real_ans:
        sim[doc] = []
        words_list = []
        tf_vector = []
        df_vector = []
        q_vector = []
        q_unit = []
        tf_vector_d = []
        d_vector = []
        d_unit = []
        index = 0
        index_d = 0
        dot_product_index = 0
        euclidean = 0
        euclidean_d = 0
        similarity = 0
        insert_index = 0
        for term in positional_index_list:
            if doc in positional_index_list[term][2]:
                if term not in words_list:
                    words_list.append(term)
        for word in words_list:
            if word in new_query:
                tf_vector.append(new_query.count(word))
            else:
                tf_vector.append(0)
        for term in positional_index_list:
            if doc in positional_index_list[term][2]:
                if term in words_list:
                    df_vector.append(positional_index_list[term][1])
        for df in df_vector:
            idf_weight = math.log10(number_of_docs / df)
            tf = tf_vector[index]
            if tf == 0:
                tf_weight = 0
            else:
                tf_weight = 1 + math.log10(tf_vector[index])
            tf_idf = tf_weight * idf_weight
            tf_idf = "{:.2f}".format(tf_idf)
            q_vector.append(tf_idf)
            index += 1
        for q in q_vector:
            euclidean = euclidean + pow(float(q), 2)
        euclidean = math.sqrt(euclidean)
        euclidean = "{:.2f}".format(euclidean)
        for q in q_vector:
            u_q = float(q) / float(euclidean)
            u_q = "{:.2f}".format(u_q)
            q_unit.append(u_q)
        for term in positional_index_list:
            if doc in positional_index_list[term][2]:
                for key, value in positional_index_list[term][3].items():
                    if doc == key:
                        counter = 0
                        for _ in value:
                            counter += 1
                tf_vector_d.append(counter)
        for df in df_vector:
            idf_weight = math.log10(number_of_docs / df)
            tf = tf_vector_d[index_d]
            if tf == 0:
                tf_weight = 0
            else:
                tf_weight = 1 + math.log10(tf_vector_d[index_d])
            tf_idf = tf_weight * idf_weight
            tf_idf = "{:.2f}".format(tf_idf)
            d_vector.append(tf_idf)
            index_d += 1
        for d in d_vector:
            euclidean_d = euclidean_d + pow(float(d), 2)
        euclidean_d = math.sqrt(euclidean_d)
        euclidean_d = "{:.2f}".format(euclidean_d)
        for d in d_vector:
            u_d = float(d) / float(euclidean_d)
            u_d = "{:.2f}".format(u_d)
            d_unit.append(u_d)
        for q in q_unit:
            num_1 = q
            num_2 = d_unit[dot_product_index]
            product = float(num_1) * float(num_2)
            similarity = similarity + product
            dot_product_index += 1
        similarity = "{:.2f}".format(similarity)
        sim[doc].append (similarity)
        print("SIM(", doc, ", Q) = ", similarity)
    print("--------------------------------------------------------------------")
    ranked_docs = dict(sorted(sim.items(), key=lambda item: item[1],reverse=True))
    for key, value in ranked_docs.items():
        print("SIM(", key, ", Q) = ", value[0])
    


def print_positional_index(positional_index_list):
    for term in positional_index_list:
        # print(term_list[term][3][1])
        print(term + " : number of docs containing this term is : " + str(positional_index_list[term][1]))
        for key, value in positional_index_list[term][3].items():
            # print(key , value)
            print("document : " + str(key), end=" ")
            for v in value:
                print(" position → " + str(v), end=" ")
            print(" ")
        print("-----------------------------------------------------------------------------")


if __name__ == '__main__':
    final_token_list = remove_stop_words(tokenization())
    positional_index_1 = positional_index_2(final_token_list)
    print("------------------------------part 1-------------------------------------")
    print("\n\n")
    print(final_token_list)
    print("\n\n")
    print("------------------------------part 2-------------------------------------")
    print("\n\n")
    print_positional_index(positional_index_1)

    query = input("PLEASE ENTER THE QUERY : ")
    matched_docs = ph_query(query, positional_index_1)
    if matched_docs != 0:
        print_matched_docs(matched_docs)
    print("\n\n")
    print("------------------------------part 3-------------------------------------")
    print("\n\n")
    compute_term_frequency(positional_index_1)
    print("\n\n")
    print("------------------------------TF-IDF MATRIX-------------------------------------")
    print("\n\n")
    compute_tf_idf_matrix(positional_index_1)
    print("\n\n")
    if matched_docs != 0:
        unit_query(query, matched_docs, positional_index_1)