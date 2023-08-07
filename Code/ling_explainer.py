from spacy.language import Language
from spacy_lefff import LefffLemmatizer, POSTagger
import pandas as pd 
import json
import math
import spacy

def word_list():
    dict_word_list = {}
    dict_word_list['week_day'] =  ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    
    dict_word_list['prons_1'] = ["moi", "je", "me", "nous", "mien", "miens", "nos", "notre", "nôtres", "mon", "ma", "mes", "moi", "nous", "j'", "m'"]
    
    dict_word_list['prons_2'] = ["toi", "tu", "te", "vous", "tien", "tiens", "vos", "votre", "votres", "ton", "ta", "tes", "toi", "vous"]
    
    dict_word_list['pr_rel'] = ["qui", "que", "dont", "où", "lequel", "quoi", "qu'"]
    
    dict_word_list['conj_comp'] = ["or", "donc", "car", "pourtant", "cependant", "néanmoins", "toutefois", "malgré", "outre", "quoique", "tandis"]
    
    dict_word_list['adv_mod'] = ["vraiment", "réellement", "sûrement", "certainement", "assurément", "incontestablement", "sans doute", "sans aucun doute", "heureusement", "malheureusement", "peut-être"]
    
    dict_word_list['negs'] = ["ne", "ni", "n", "non", "aucun", "sans", "nul", "nulle", "n'"]
    
    with open("../NeededFiles/lexicons/Citing_verbs.txt") as f:
        dict_word_list['citing_verbs'] = [line.strip("\n") for line in f] 
        
    with open("../NeededFiles/lexicons/Thinking_verbs.txt") as f:
        dict_word_list['thinking_verbs'] = [line.strip("\n") for line in f]

    with open("../NeededFiles/lexicons/Discourse_markers.txt") as f:
        dict_word_list['discourse_markers'] = [line.strip("\n") for line in f]
        
    return dict_word_list

#Counts the number of token belonging to a list in a text
def count_tok(tokenList, sentence):

    if len(sentence) == 0:
        return 0
    
    cnt = 0
    for token in sentence:
        if token in tokenList:
            cnt += 1
            
    return cnt/len(sentence)
    
# takes list of tokens and returns count of tokens with more than 8 characters (float)
def count_long_words(tokens):
    
    if len(tokens) == 0:
        return 0
    
    cnt_long_words = 0
    for i in tokens:
        if len(i) > 8:
            cnt_long_words += 1
    
    return cnt_long_words / len(tokens)   

# Corrected type-token ratio 
def cttr(tokens):

    if len(tokens) == 0:
        return 0
    
    types = {}
    for token in tokens:
        if token not in types:
            types[token] = True
    
    cttr = len(types) / math.sqrt(2 * len(tokens))
    return cttr

def build_lexicons():
    lexique3_path = f"../NeededFiles/lexicons/open_lexicon.csv"
    nrc_path = f"../NeededFiles/lexicons/nrc_lexicon.json"

    # create spacy lemmatizer
    @Language.factory('french_lemmatizer')
    def create_french_lemmatizer(nlp, name):
        return LefffLemmatizer(after_melt=True, default=True)

    # create spacy POS-tagger
    @Language.factory('melt_tagger')
    def create_melt_tagger(nlp, name):
        return POSTagger()


    # load Lexique3 lexicon
    columns = ["Word", "Pol.Val.", "Catégories"]
    lexique3_df = pd.read_csv(lexique3_path, sep=';', usecols = columns)
    lexique3_df.set_index("Word", inplace=True)
    lexique3_dict = lexique3_df.to_dict()
    lexique3_dict = lexique3_dict["Pol.Val."]

    # load NRC lexicon
    with open(nrc_path,'r') as f:
        data = json.loads(f.read())

    nrc_df = pd.json_normalize(data, record_path =['LexicalEntry'])
    nrc_df = nrc_df[['Lemma.-writtenForm', 'Sense.Sentiment.-polarity']].copy()
    nrc_df.set_index("Lemma.-writtenForm", inplace=True)
    nrc_df_pos = nrc_df[nrc_df['Sense.Sentiment.-polarity']=="positive"]
    nrc_df_neg = nrc_df[nrc_df['Sense.Sentiment.-polarity']=="negative"]
    nrc_lex_df = pd.concat([nrc_df_pos, nrc_df_neg])
    nrc_dict = nrc_lex_df.to_dict()
    nrc_dict = nrc_dict["Sense.Sentiment.-polarity"]
    
    return lexique3_dict, nrc_dict

def associate_measures(feature_dict, tokens_lc, tags, lemmas, lexique3_dict, nrc_dict, dict_word_list):
    rule_set_list = [set() for _ in tokens_lc]
    
    for pos, (tkn, tag, lemma) in enumerate(zip(tokens_lc, tags, lemmas)):
        tkn = tkn.lower()
        for feature, tkn_list in feature_dict.items():
            if (tkn in tkn_list) or (tag in tkn_list):
                rule_set_list[pos].add(feature)
                
        if len(tkn) > 8:
            rule_set_list[pos].add("nb_long_words")

        if lemma in lexique3_dict and lexique3_dict[lemma] in ['neg', 'positif']:
            rule_set_list[pos].add("lexique3_sentiment")

        if lemma in nrc_dict and nrc_dict[lemma] in ['negative', 'positive']:
            rule_set_list[pos].add("nrc_sentiment")

        if lemma in dict_word_list['citing_verbs']:
            rule_set_list[pos].add("citing_verbs")
            
    return rule_set_list 

def ling_measures(tokens, rule_set_list, feature_list):
    measures_dict = {}
    
    counting_dict = {i: 0 for i in feature_list}
    for feature_set in rule_set_list:
        for elem in feature_set:
            counting_dict[elem] = counting_dict.get(elem, 0) + 1
    
    for feature, count in counting_dict.items():
        measures_dict[feature] = 100*count/len(tokens) if len(tokens) != 0 else 0
    
    measures_dict["text_length"] =  len(tokens)/512
    measures_dict["length_words"] = sum([len(tok) for tok in tokens])/len(tokens) if len(tokens) != 0 else 0
    measures_dict["cttr"] = cttr(tokens)
    
    return measures_dict

def compute_relevance(feature_list, rule_set_list):
    relevance_map = [0 for _ in rule_set_list]
    
    for pos, rule_set in enumerate(rule_set_list):
        for elem in rule_set:
            if elem in feature_list:
                val = feature_list[elem]
                relevance_map[pos] += val
    
    return relevance_map

# create linguistic attention map of article (type(article) = str)
def create_map(tokens_lc, rule_set_list, feature_list, logreg):
    
    ling_measures_dict = ling_measures(tokens_lc, rule_set_list, feature_list)
    ling_v = pd.DataFrame(ling_measures_dict, index=[0], columns=feature_list)
    
    pred = logreg.predict(ling_v)
    model_coefs = {feature:logreg.coef_[0][i] for i, feature in enumerate(feature_list)}
        
    if pred == 0:
        info_feat = {feature: val for feature, val in model_coefs.items() if val <0}
        ling_map = compute_relevance(info_feat, rule_set_list)
        
    if pred == 1:
        opin_feat = {feature: val for feature, val in model_coefs.items() if val >0}
        ling_map = compute_relevance(opin_feat, rule_set_list)
            

    xmin = min([abs(i) for i in ling_map]) 
    xmax = max([abs(i) for i in ling_map])
    for i, x in enumerate(ling_map):
        x = abs(x)
        ling_map[i] = (x-xmin) / (xmax-xmin)
        if pred == 1:
            ling_map[i] = -ling_map[i]
        
    return tokens_lc, ling_map, logreg.predict_proba(ling_v)