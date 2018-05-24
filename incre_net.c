//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "compute-accuracy.h"

#include <unistd.h>

#define MAX_STRING 100
//#define MAX_CONTEXT_RECORD 100
#define MAX_CONTEXT_RECORD 50
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

// const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int vocab_hash_size = 60000000;  // Maximum 60 * 0.7 = 42M words in the vocabulary
// NOTE: in the NE type version, this also includes new ne types,
//       we reduce the vocab when it is larger than 70% of this size,
//	    so there should be sufficient location for ne types.
const int new_word_hash_size = 10000000;  // Maximum 30 * 0.7 = 21M words in the new word array

typedef float real;                    // Precision of float numbers

struct vocab_word {
    long long cn;
    // boolean value, if this word needs to be updated.
    char update;
    int *point;
    char *word, *code, codelen;

    // stores the index of neType of this word in vocab.
    // put -1 if not a Name Entity.
    // put neType_start if is a Name Entity but the type is unknow.
    // = itself if it is a name entity type
    long long neType;
};

struct new_vocab_word {
    char *context[MAX_CONTEXT_RECORD];
    char *word;
    int context_size;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
char embedding_file[MAX_STRING], update_list[MAX_STRING], type_word_list[MAX_STRING];
struct vocab_word *vocab;
struct new_vocab_word *new_words;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int init_method = 1, updateNew = 1, updateAll = 1, eval = 0;
int *vocab_hash;
int *new_word_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long max_new_word = 1000, num_of_new_words = 0, word_in_list = 0;
int num_of_types = 0;
long long neType_start = 0;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5, negTypes = 0;
const int table_size = 1e8;
int *table;
long long *typeIndex;

// For debug use
void PrintSentence(char* sentence[]) {
    int i = 0;
    while(strcmp(sentence[i],"")) {
        printf("%s%c%d%c ", sentence[i],91,i,93);
        i++;
    }
    printf("\n");
}

// For debug use
void PrintVector(int index) {
    if (index == -1) {
        printf("can not find word.\n");
        return;
    }
    printf("%s %c%d%c", vocab[index].word, 91, index, 93);
    int i;
    for(i = 0; i < layer1_size; i++) {
        printf("%f ", syn0[index * layer1_size + i]);
    }
    printf("\n");
}

void InitUnigramTable() {
    int a, i;
    double train_words_pow = 0;
    double d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
    i = 0;
    d1 = pow(vocab[i].cn, power) / train_words_pow;
    for (a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (double)table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }
}

void InitTypeIndexArray() {
    int a;
    typeIndex = (long long *)malloc(num_of_types * sizeof(long long));
    for (a = 0; a < num_of_types; a++) typeIndex[a] = neType_start + a;
}

int rand_interval(int min, int max)
{
    int r;
    const int range = 1 + max - min;
    const int buckets = (unsigned long long)25214903917 / range;
    const int limit = buckets * range;

    /* Create equal size buckets all in a row, then fire randomly towards
     * the buckets until you land in one of them. All buckets are equally
     * likely. If you land off the end of the line of buckets, try again. */
    do
    {
        r = rand();
    } while (r >= limit);

    return min + (r / buckets);
}

char inArray(int *array, int size, int element){
	int a;
	for (a = 0; a < size; a++){
		if(array[a] == element) return 1;
	}
	return 0;
}

/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator. */
void shuffle(long long *array, size_t n)
{
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / ((unsigned long long)25214903917 / (n - i) + 1);
            long long t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}

int ReadSentence(char** sentence, FILE *fin) {
    int sen_length = 0;
    char temp[MAX_STRING];
    while(sen_length < MAX_SENTENCE_LENGTH) {
        int a = 0, ch;
        while (1) {
            ch = fgetc(fin);
            if(feof(fin)) {
                if (sen_length > 0) return sen_length;
                else return -1;
            }
            if (ch == 13) continue;
            if ((ch == ' ') || (ch == '\t')) {
                if (a == 0) continue;
                else break;
            }
            if (ch == '\n') {
                if (a > 0) {
                    temp[a] = 0;
                    strncpy(sentence[sen_length], temp, a+1);
                    sen_length ++;
                }
                strcpy(temp, (char *)"</s>");
                strncpy(sentence[sen_length], temp, 5);
                sen_length ++;

                return sen_length;
            }
            temp[a] = ch;
            a++;
            if (a >= MAX_STRING - 1) a--;   // Truncate too long words
        }
        temp[a] = 0;

        strncpy(sentence[sen_length], temp, a+1);
        sen_length ++;
    }
    return sen_length;
}

// Returns hash value of a word
int GetWordHash(char *word, int hash_size) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word, vocab_hash_size);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Returns position of a word in the new_words list; if the word is not found, returns -1
int SearchNewWord(char *word) {
    unsigned int hash = GetWordHash(word, new_word_hash_size);
    while (1) {
        if (new_word_hash[hash] == -1) return -1;
        if (!strcmp(word, new_words[new_word_hash[hash]].word)) return new_word_hash[hash];
        hash = (hash + 1) % new_word_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab[vocab_size].update = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word, vocab_hash_size);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Adds a word to the new word array
int AddNewWord(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    int i = 0;
    if (length > MAX_STRING) length = MAX_STRING;
    new_words[num_of_new_words].word = (char *)calloc(length, sizeof(char));
    strcpy(new_words[num_of_new_words].word, word);
    // Allocate memory for context records
    for (i = 0; i < MAX_CONTEXT_RECORD; i++) new_words[num_of_new_words].context[i] = malloc(MAX_STRING);
    num_of_new_words++;
    // Reallocate memory if needed
    if (num_of_new_words + 2 >= max_new_word) {
        max_new_word += 1000;
        new_words = (struct new_vocab_word *)realloc(new_words, max_new_word * sizeof(struct new_vocab_word));
    }
    hash = GetWordHash(word, new_word_hash_size);
    while (new_word_hash[hash] != -1) hash = (hash + 1) % new_word_hash_size;
    new_word_hash[hash] = num_of_new_words - 1;
    return num_of_new_words - 1;
}

// params: index of word in new_words array
// 		   current index in this sentence
//		   the pointer ot current sentence
//		   the length of sentence
void AddContext(int word_index, int current_index, char** sentence, int sen_length) {
    int start = current_index - window;
    int stop = current_index + window;
    int i, j;
    for (i = start; i <= stop; i++) {
        if (i == current_index) continue;
        if (i >= 0 && i < sen_length && new_words[word_index].context_size < MAX_CONTEXT_RECORD) {
            int size = new_words[word_index].context_size;
            char add = 1;
            for (j = 0; j < size; j++) {
                add = strcmp(new_words[word_index].context[j], sentence[i]);
                if (add == 0) break;
            }
            if(add && size < MAX_CONTEXT_RECORD) strcpy(new_words[word_index].context[size], sentence[i]);
            new_words[word_index].context_size ++;
        }
    }
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    int a, size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    size = vocab_size;
    for (a = 0; a < size; a++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if ((vocab[a].cn < min_count) && (a != 0)) {
            vocab_size--;
            free(vocab[a].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetWordHash(vocab[a].word, vocab_hash_size);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = a;
        }
    }
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    // Allocate memory for the binary tree construction
    for (a = 0; a < vocab_size; a++) {
        vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}


// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
            vocab[b].cn = vocab[a].cn;
            vocab[b].word = vocab[a].word;
            vocab[b].update = vocab[a].update;
            b++;
        } else free(vocab[a].word);
    vocab_size = b;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    for (a = 0; a < vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetWordHash(vocab[a].word, vocab_hash_size);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
    for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
    pos1 = vocab_size - 1;
    pos2 = vocab_size;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++) {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            } else {
                min1i = pos2;
                pos2++;
            }
        } else {
            min1i = pos2;
            pos2++;
        }
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            } else {
                min2i = pos2;
                pos2++;
            }
        } else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++) {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == vocab_size * 2 - 2) break;
        }
        vocab[a].codelen = i;
        vocab[a].point[0] = vocab_size - 2;
        for (b = 0; b < i; b++) {
            vocab[a].code[i - b - 1] = code[b];
            vocab[a].point[i - b] = point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

// Also records the context words of new words
void LearnVocabFromTrainFile() {
    FILE *fin =fopen(train_file, "rb");
    char* sentence[MAX_SENTENCE_LENGTH];
    int j;
    for (j = 0; j < MAX_SENTENCE_LENGTH; j++) sentence[j] = malloc(MAX_STRING);
    long long a, i, b;
    int sen_length = 0;

    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }

    // Init the new word hash
    if (init_method == 2)for (a = 0; a < new_word_hash_size; a++) new_word_hash[a] = -1;
    while (sen_length != -1) {
        j = 0;
        sen_length = ReadSentence(sentence, fin);
        while(j < sen_length) {
            train_words++;
            if ((debug_mode > 1) && (train_words % 100000 == 0)) {
                printf("%lldK%c", train_words / 1000, 13);
                fflush(stdout);
            }

            i = SearchVocab(sentence[j]);
            if (i == -1) {
                a = AddWordToVocab(sentence[j]);
                vocab[a].cn = 1;
                vocab[a].neType = -1;
                if(updateNew == 1 || updateAll == 1) vocab[a].update = 1;
                if (init_method == 2) {
                    b = AddNewWord(sentence[j]);
                    AddContext(b, j, sentence, sen_length);
                }
            } else {
                vocab[i].cn++;
                // if in the new_word list, add context
                if (init_method == 2) {
                    i = SearchNewWord(sentence[j]);
                    if (i != -1) AddContext(i, j, sentence, sen_length);
                }
            }
            if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
            j++;
        }

        for (j = 0; j < MAX_SENTENCE_LENGTH; j++) strcpy(sentence[j], "");
    }
    SortVocab();

    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}


void SaveVocab(char* toFile) {
    printf("saving vocabulary to: %s\n", toFile);
    long long i;
    FILE *fo = fopen(toFile, "wb");
    for (i = 0; i < vocab_size; i++) {
        fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    }
    fclose(fo);
}

void SaveVectors(FILE *fo, char bin) {
    long a, b;
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (bin) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
        else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
}

// Please note put UNKOWN at the first place
void ReadTypeWordList() {
    long long word_index = 0;
    char word[MAX_STRING];
    char type[MAX_STRING];
    long long type_index;
    int j = 0;
    FILE *fin = fopen(type_word_list, "rb");
    if (fin == NULL) {
        printf("Type -> word list file not found\n");
        exit(1);
    }
    num_of_types = 0;
    while (1) {
        ReadWord(type, fin);
        if (feof(fin)) break;
        num_of_types++;
        type_index = AddWordToVocab(type);
        vocab[type_index].neType = type_index;
        if ((debug_mode > 1) && (num_of_types % 1000 == 0)) {
            printf("%dK%c", num_of_types / 1000, 13);
            fflush(stdout);
        }
        ReadWord(word, fin);
        int count = 0;
        while(strcmp(word, "</s>")) {
            word_index = SearchVocab(word);
            if (word_index != -1) {
                // assign the type to word
                vocab[word_index].neType = type_index;
                // the frequency of type = sum of frequency of words belong to this type
                vocab[type_index].cn += vocab[word_index].cn;
                // update type vectors
                vocab[type_index].update = 1;
                // sum word vectors
                for(j = 0; j < layer1_size; j++) syn0[type_index * layer1_size + j] += syn0[word_index * layer1_size + j];
                count++;
            } else printf("cannot find word %s in vocabulary\n", word);
            ReadWord(word, fin);
        }
        // calculate the average
        if (count != 0) {
            for(j = 0; j < layer1_size; j++) syn0[type_index * layer1_size + j] = syn0[type_index * layer1_size + j] / count;
        }
    }

    if (debug_mode > 0) {
        printf("Size of name entity types: %d\n", num_of_types);
    }

    fclose(fin);
}

void ReadVocab() {
    long long a = 0;
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        a = AddWordToVocab(word);
        fscanf(fin, "%lld%c", &vocab[a].cn, &c);
        if (updateAll == 1) vocab[a].update = 1;
        vocab[a].neType = -1;
    }
    if (debug_mode > 0) {
        printf("Loaded Vocab size: %lld\n", vocab_size);
    }
    fclose(fin);
}

// Read the first line: header of an embedding file
int ReadHeader(FILE *fin) {
    long long int result;
    fscanf(fin, "%lld", &result);
    fscanf(fin, "%lld", &result);
    return result;
}

void ReadUpdateList() {
    long long index = 0;
    char word[MAX_STRING];
    FILE *fin = fopen(update_list, "rb");
    if (fin == NULL) {
        printf("Update List file not found, will update all new words.\n");
        updateNew = 1;
        return;
    }
    printf("Loading words needs to be updated from file %s\n", update_list);
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        index = SearchVocab(word);
        if(index != -1) vocab[index].update = 1;
        word_in_list += 1;
    }
    if (debug_mode > 0) {
        printf("Update list loaded.\n");
    }
    fclose(fin);
}

void LoadEmbeddings() {
    long long index = 0, line = 0;
    char c = '\0';
    int i;
    char word[MAX_STRING];
    FILE *fin = fopen(embedding_file, "rb");
    if (fin == NULL) {
        printf("Embedding file not found\n");
        exit(1);
    }
    printf("Loading pre-trained word embeddings from file %s\n", embedding_file);

    layer1_size = ReadHeader(fin);
    posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    while (1) {
        fscanf(fin, "%s\n", word);
        line++;
        if ((debug_mode > 1) && (line % 10000 == 0)) {
            printf("%lldK%c", line / 1000, 13);
            fflush(stdout);
        }
        if (feof(fin)) break;
        index = SearchVocab(word);
        char find = 1;
        for(i = 0; i < layer1_size; i++) {
            float temp;
            fscanf(fin, "%f%c", &temp, &c);
            if (index != -1)syn0[index * layer1_size + i] = temp;
            else {
                find = 0;
            }
        }
        if(find == 0) 	printf("cannot find word %s in vocabulary\n", word);
    }
    fclose(fin);
    if (debug_mode > 0) {
        printf("Vector size: %lld\n", layer1_size);
    }
}

// get the average context words' embedding for the word of index a in the vocab
// return 1 if all good, otherwise, show warning and returen 0.
int AssignAverage(int a) {
    char* word = vocab[a].word;
    // do not init </s>
    if (!strcmp(word, "</s>")) return 1;
    int index = SearchNewWord(word);
    if (index == -1) {
        int length = strlen(word);
        if (length < 50) printf("Can not find word %s in the new word list.\n", word);
        return 0;
    }
    char** context = new_words[index].context;
    int size = new_words[index].context_size;
    int i, j;
    real* temp;
    // declare memory for temp, init temp
    posix_memalign((void **)&temp, 128, (long long)layer1_size * sizeof(real));
    for (i = 0; i < layer1_size; i++) temp[i] = 0;
    // add up over context words in each dimention
    for (i = 0; i < size; i++) {
        //
        char* c_word = context[i];
        int in_vocab = SearchVocab(c_word);
        if(in_vocab != -1) {
            for(j = 0; j < layer1_size; j++) temp[j] += syn0[in_vocab*layer1_size + j];
        }
    }

    float sum = 0.0;
    // average over context words in each dimention
    for (i = 0; i < layer1_size; i++) {
        temp[i] = temp[i] / size;
        sum += temp[i];
        syn0[a * layer1_size + i] = temp[i];
    }
    // after averaged, still ==0
    if(sum < 0.0000005) return 0;
    else return 1;
}

void InitNet() {
    LoadEmbeddings();
    if (eval == 2) {
        printf("Accuracy after load embeddings\n");
        FILE *temp;
        temp = fopen("./temp.bin", "wb");
        SaveVectors(temp, 1);
        evalEmbed("./temp.bin",30000,"./questions-words.txt");
        remove("./temp.bin");
        fclose(temp);
        printf("press any key to continue\n");
        getchar();
    }

    long long a, b;
    unsigned long long next_random = 1;
    if (hs) {
        a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if (syn1 == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
                syn1[a * layer1_size + b] = 0;
    }
    if (negative>0) {
        a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if (syn1neg == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
                syn1neg[a * layer1_size + b] = 0;
    }
    printf("Initializing word vectors ...\n");
    for (a = 0; a < vocab_size; a++) {
        float sum = 0.0;

        for (b = 0; b < layer1_size; b++) {
            sum += fabs(syn0[a * layer1_size + b]);
        }
        // vector not included in pre-trained
        if(sum < 0.0000005) {
            // Random
            if(init_method == 1) {
                for (b = 0; b < layer1_size; b++) {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
                }
            }
            // Averaged
            else if(init_method == 2) {
                int re = AssignAverage(a);
                // if after average, still == 0
                if (re == 0) {
                    for (b = 0; b < layer1_size; b++) {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
                    }
                }
            }
        } else continue;
    }
}

void FreeNewWord() {
    int i;
    for (i = 0; i < max_new_word; i ++) {
        int size = new_words[i].context_size;
        int j;
        for (j = 0; j < size; j ++) {
            free(new_words[i].context[j]);
        }
        free(new_words[i].word);
    }
    free(new_words);
    free(new_word_hash);
}

void *TrainModelThread(void *id) {
    long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label, local_iter = iter;
    unsigned long long next_random = (long long)id;
    long long type = -1;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(train_file, "rb");
    fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
    while (1) {
        if (word_count - last_word_count > 10000) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                       word_count_actual / (real)(iter * train_words + 1) * 100,
                       word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while (1) {
                word = ReadWordIndex(fi);
                if (feof(fi)) break;
                // not in vocab
                if (word == -1) continue;
                word_count++;
                // is the end of sentence
                if (word == 0) break;
                // The subsampling randomly discards frequent words while keeping the ranking same
                if (sample > 0) {
                    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi) || (word_count > train_words / num_threads)) {
            word_count_actual += word_count - last_word_count;
            local_iter--;
            if (local_iter == 0) break;
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
            continue;
        }
        word = sen[sentence_position];

        if (word == -1) continue;
        type = vocab[word].neType;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        if (cbow) {  //train the cbow architecture
            // in -> hidden
            cw = 0;
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                    c = sentence_position - window + a;
                    if (c < 0) continue;
                    if (c >= sentence_length) continue;
                    last_word = sen[c];
                    if (last_word == -1) continue;
                    for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
                    cw++;
                }
            // if word has entity type, make the type point as one of the context word as well
            if(type != -1) {
                for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + type * layer1_size];
                cw++;
            }

            if (cw) {
                for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
                if (hs) for (d = 0; d < vocab[word].codelen; d++) {
                        f = 0;
                        l2 = vocab[word].point[d] * layer1_size;
                        // Propagate hidden -> output
                        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
                        if (f <= -MAX_EXP) continue;
                        else if (f >= MAX_EXP) continue;
                        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        // 'g' is the gradient multiplied by the learning rate
                        g = (1 - vocab[word].code[d] - f) * alpha;
                        // Propagate errors output -> hidden
                        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                        // Learn weights hidden -> output
                        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
                    }
                // NEGATIVE SAMPLING
                if (negative > 0) for (d = 0; d < negative + 1; d++) {
                        if (d == 0) {
                            target = word;
                            label = 1;
                        } else {
                            next_random = next_random * (unsigned long long)25214903917 + 11;
                            target = table[(next_random >> 16) % table_size];
                            if (target == 0) target = next_random % (vocab_size - 1) + 1;
                            if (target == word) continue;
                            label = 0;
                        }
                        l2 = target * layer1_size;
                        f = 0;
                        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
                        if (f > MAX_EXP) g = (label - 1) * alpha;
                        else if (f < -MAX_EXP) g = (label - 0) * alpha;
                        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
                    }
                // hidden -> in
                for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                        c = sentence_position - window + a;
                        if (c < 0) continue;
                        if (c >= sentence_length) continue;
                        last_word = sen[c];
                        if (last_word == -1) continue;
                        // if this word not need to be updated
                        if (!vocab[last_word].update) continue;
                        for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
                    }
                // update type vectors
                if(type != -1) {
                    for (c = 0; c < layer1_size; c++) syn0[c + type * layer1_size] += neu1e[c];
                }
            }
        } else {  //train skip-gram
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                    c = sentence_position - window + a;
                    if (c < 0) continue;
                    if (c >= sentence_length) continue;
                    last_word = sen[c];
                    long long temp_type = vocab[last_word].neType;
                    if (last_word == -1) continue;
                    l1 = last_word * layer1_size;
                    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                    // HIERARCHICAL SOFTMAX
                    if (hs) for (d = 0; d < vocab[word].codelen; d++) {
                            f = 0;
                            l2 = vocab[word].point[d] * layer1_size;
                            // Propagate hidden -> output
                            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
                            if (f <= -MAX_EXP) continue;
                            else if (f >= MAX_EXP) continue;
                            else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                            // 'g' is the gradient multiplied by the learning rate
                            g = (1 - vocab[word].code[d] - f) * alpha;
                            // Propagate errors output -> hidden
                            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                            // Learn weights hidden -> output
                            for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
                        }
                    // NEGATIVE SAMPLING
                    if (negative > 0) for (d = 0; d < negative + 1; d++) {
                            if (d == 0) {
                                target = word;
                                label = 1;
                            } else {
                                next_random = next_random * (unsigned long long)25214903917 + 11;
                                target = table[(next_random >> 16) % table_size];
                                if (target == 0) target = next_random % (vocab_size - 1) + 1;
                                if (target == word) continue;
                                label = 0;
                            }
                            l2 = target * layer1_size;
                            f = 0;
                            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
                            if (f > MAX_EXP) g = (label - 1) * alpha;
                            else if (f < -MAX_EXP) g = (label - 0) * alpha;
                            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                            for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
                        }
                    // Learn weights input -> hidden
                    // if this word not need to be updated
                    if (!vocab[last_word].update) continue;
                    for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
                    if (temp_type != -1) for (c = 0; c < layer1_size; c++) syn0[c + temp_type * layer1_size] += neu1e[c];
                }
        }

        // Add type 
        if(type != -1) {
			int randomIndex = 0; 
			int array[negTypes];
			
            l1 = word * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

            // for other types
            for(d = 0; d < negTypes + 1; d++) {
                if (d == 0) {
                    target = type;
                    label = 1;
                } else {
					randomIndex = rand_interval(0, num_of_types);
                    if (target == type || inArray(array, negTypes, randomIndex)) {
						d--;
						continue;
						} else target = neType_start + randomIndex;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
            }
            for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }

        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void TrainModel() {
    long a, b, c, d;
    FILE *fo;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    printf("Starting training using file %s\n", train_file);

    starting_alpha = alpha;
    if (read_vocab_file[0] != 0) {
        ReadVocab();
        neType_start = vocab_size;
    }
    LearnVocabFromTrainFile();
    // This is the end of real words
    neType_start = vocab_size;
    if (save_vocab_file[0] != 0) SaveVocab(save_vocab_file);
    if (output_file[0] == 0) return;

    if (update_list[0] != 0) ReadUpdateList();
    InitNet();

    //if (type_file[0] != 0) ReadEntityTypeList();
    //if (word_type_map[0] !=0 ) ReadWordTypeMap();
    // Append ne types at the end of vocabulary list
    if (type_word_list[0] !=0 ) {
        ReadTypeWordList();
        // for negtive sampling
        //InitTypeIndexArray();
        vocab_size = neType_start;
        if (negTypes == 0) negTypes = num_of_types;
    }

    CreateBinaryTree();

    if (eval == 2) {
        printf("Accuracy after Init embeddings\n");
        FILE *temp;
        temp = fopen("./temp.bin", "wb");
        SaveVectors(temp, 1);
        evalEmbed("./temp.bin",30000,"./questions-words.txt");
        remove("./temp.bin");
        fclose(temp);
        printf("press any key to continue\n");
        getchar();
    }


    if(init_method == 2) FreeNewWord();
    if (negative > 0) InitUnigramTable();

    if (updateAll == 1) printf("Updating all vocabs\n");
    else {
        if (updateNew ==  1) {
            if (num_of_new_words == 0) printf("Note: no new vocabs added\n");
            printf("Updating new vocabs\n");
        }
        if (update_list[0] != 0) {
            if (word_in_list == 0) printf("Note: no vocabs in update list\n");
            printf("Updating vocabs in update list\n");
        }

    }

    start = clock();
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    fo = fopen(output_file, "wb");
    if (classes == 0) {
        // Save the word vectors
        printf("saving vectors to %s\n", output_file);
        SaveVectors(fo, binary);
        if(eval >= 1) {
            printf("calculating accuracy on words questions\n");
            if (binary == 0) {
                FILE *temp;
                temp = fopen("./temp.bin", "wb");
                SaveVectors(temp, 1);
                evalEmbed("./temp.bin", 50000, "./questions-words.txt");
                remove("./temp.bin");
                fclose(temp);
            }

            else {
                evalEmbed(output_file, 50000, "./questions-words.txt");
            }
        }
    } else {
        // Run K-means on the word vectors
        int clcn = classes, iter = 10, closeid;
        int *centcn = (int *)malloc(classes * sizeof(int));
        int *cl = (int *)calloc(vocab_size, sizeof(int));
        real closev, x;
        real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
        for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
        for (a = 0; a < iter; a++) {
            for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
            for (b = 0; b < clcn; b++) centcn[b] = 1;
            for (c = 0; c < vocab_size; c++) {
                for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
                centcn[cl[c]]++;
            }
            for (b = 0; b < clcn; b++) {
                closev = 0;
                for (c = 0; c < layer1_size; c++) {
                    cent[layer1_size * b + c] /= centcn[b];
                    closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
                }
                closev = sqrt(closev);
                for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
            }
            for (c = 0; c < vocab_size; c++) {
                closev = -10;
                closeid = 0;
                for (d = 0; d < clcn; d++) {
                    x = 0;
                    for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
                    if (x > closev) {
                        closev = x;
                        closeid = d;
                    }
                }
                cl[c] = closeid;
            }
        }
        // Save the K-means classes
        for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
        free(centcn);
        free(cent);
        free(cl);
    }
    fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-embedding <file>\n");
        printf("\t\tPre-trained word embeddings <file>\n");
        printf("\t-update-list <file>\n");
        printf("\t\tA list of words <file> that needs to be updated\n");
        printf("\t-type-file <file>\n");
        printf("\t\tA list of entity types\n");
        printf("\t-word-type-map <file>\n");
        printf("\t\tA file that contains word -> entity type maps\n");
        printf("\t-updateNew <int>\n");
        printf("\t\tSet to 1: update all new words in training data. 0 only update words in the update list\n");
        printf("\t-updateAll <int>\n");
        printf("\t\tupdate all words in training data and pre-trained vocabulary. 0 only update words in the update list, Default 0\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
        printf("\t-classes <int>\n");
        printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-eval <int>\n");
        printf("\t\tevaluation embeddings mode (default = 0: donnot evaluate; 1: evaluate after training; 2: evaluate after load embedding)\n");
        printf("\t-init <int>\n");
        printf("\t\tNew word vectors initilize methond, 1 for random, 2 for averaged. (default = 1)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-locab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe pre-trained vocabulary <file>\n");
        printf("\t-cbow <int>\n");
        printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }
    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    embedding_file[0] = 0;
    update_list[0] = 0;
    type_word_list[0] = 0;

    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-embedding", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-update-list", argc, argv)) > 0) strcpy(update_list, argv[i + 1]);
    if ((i = ArgPos((char *)"-type-word-list", argc, argv)) > 0) strcpy(type_word_list, argv[i + 1]);
    if ((i = ArgPos((char *)"-updateNew", argc, argv)) > 0) updateNew = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-updateAll", argc, argv)) > 0) updateAll = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-init", argc, argv)) > 0) init_method = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-eval", argc, argv)) > 0) eval = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negTypes", argc, argv)) > 0) negTypes = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    if (init_method == 2) new_words = (struct new_vocab_word *)calloc(max_new_word, sizeof(struct new_vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    if (init_method == 2) new_word_hash = (int *)calloc(new_word_hash_size, sizeof(int));
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    TrainModel();
    printf("Done!\n");
    return 0;
}
