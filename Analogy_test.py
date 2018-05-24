from gensim.models.word2vec import Word2Vec
import sys
def Analogy_test(embedding='vectors.txt',analogy_test='analogy-test.txt'):
    model_vec = Word2Vec.load_word2vec_format(embedding, binary=False)
    vec_sim=0
    count=0
    sum_line=0
    with open(analogy_test,'r') as f:
        for line in f:
            wordsArray = line.split()
            if len(wordsArray)<4:
                    count+=1
                    continue
            word1=wordsArray[0].lower()
            word2=wordsArray[1].lower()
            word3=wordsArray[2].lower()
            word4=wordsArray[3].lower()
            try:
                tuple1=model_vec.most_similar(positive=[word3, word2], negative=[word1],topn=1)
                if tuple1[0][0]==word4:
                    vec_sim+=1
            except KeyError:
                count+=1
                continue
            sum_line+=1
    print "ignored lines is "+str(count)
    print "precision is "+str(float(vec_sim)/sum_line)
if __name__ == "__main__":
    Analogy_test(embedding=sys.argv[1], analogy_test=sys.argv[2])
