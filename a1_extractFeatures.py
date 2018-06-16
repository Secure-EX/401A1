import numpy as np
import argparse
import os
import json
import re
import csv
import time

reg_uni = r"(.*)(/)(.*)"

dict_slang = "/u/cs401/Wordlists/Slang"
dict_bristol = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
dict_warriner = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"
dict_alt = "/u/cs401/A1/feats/Alt_IDs.txt"
dict_center = "/u/cs401/A1/feats/Center_IDs.txt"
dict_right = "/u/cs401/A1/feats/Right_IDs.txt"
dict_left = "/u/cs401/A1/feats/Left_IDs.txt"

alt_npy = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")
center_npy = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")
left_npy = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
right_npy = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")

class_integer = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}


def extract1(comment, steps=range(1, 30)):
    """ This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features
    """

    features = np.zeros(174)  # 1-29 and rest be 0

    comment_split = comment.split()

    if 1 in steps:
        # Number of first-person pronouns
        # All 1st-person pronoun are stop words and all stop words are removed
        features[0] = 0
        print("Ex1 Step1 DONE")
    if 2 in steps:
        # Number of second-person pronouns
        # All 2nd-person pronoun are stop words and all stop words are removed
        features[1] = 0
        print("Ex1 Step2 DONE")
    if 3 in steps:
        # Number of third-person pronouns
        # All 3rd-person pronoun are stop words and all stop words are removed
        features[2] = 0
        print("Ex1 Step3 DONE")
    if 4 in steps:
        # Number of coordinating conjunctions
        cc_reg = re.compile(r"/CC")
        match = cc_reg.findall(comment)
        features[3] = len(match)
        print("Ex1 Step4 DONE")
    if 5 in steps:
        # Number of past-tense verbs
        past_reg = re.compile(r"/VBD|/VBN")
        match = past_reg.findall(comment)
        features[4] = len(match)
        print("Ex1 Step5 DONE")
    if 6 in steps:
        # Number of future-tense verb
        future_reg = re.compile(r"'ll/MD|will/MD|gonna/DT|going/VBG|shall/MD|wo/MD")
        match = future_reg.findall(comment)
        features[5] = len(match)
        print("Ex1 Step6 DONE")
    if 7 in steps:
        # Number of commas
        comma_reg = re.compile(r",/,")
        match = comma_reg.findall(comment)
        features[6] = len(match)
        print("Ex1 Step7 DONE")
    if 8 in steps:
        # Number of multi-character punctuation tokens
        char_reg = r"^[\W](\W+$)"
        count = 0
        for word in comment_split:
            if re.match(char_reg, re.match(reg_uni, word).group(1)):
                count += 1
        features[7] = count
        print("Ex1 Step8 DONE")
    if 9 in steps:
        # Number of common nouns
        com_non = re.compile(r"/NNS|/NN")
        match = com_non.findall(comment)
        features[8] = len(match)
        print("Ex1 Step9 DONE")
    if 10 in steps:
        # Number of proper nouns
        prop_non = re.compile(r"/NNP|/NNPS")
        match = prop_non.findall(comment)
        features[9] = len(match)
        print("Ex1 Step10 DONE")
    if 11 in steps:
        # Number of adverbs
        adver = re.compile(r"/RBS|/RBR|/RB")
        match = adver.findall(comment)
        features[10] = len(match)
        print("Ex1 Step11 DONE")
    if 12 in steps:
        # Number of wh- words
        # WDT, WP, WP$. WRB
        wh_words = re.compile(r"/WDT|/WRB|/WP\$|/WP")
        match = wh_words.findall(comment)
        features[11] = len(match)
        print("Ex1 Step12 DONE")
    if 13 in steps:
        # Number of slang acronyms
        count = 0
        for word in comment_split:
            if re.match(reg_uni, word):
                if re.match(reg_uni, word).group(1) in slang:
                    count += 1
        features[12] = count
        print("Ex1 Step13 DONE")
    if 14 in steps:
        # Number of words in uppercase (≥3 letters long)
        features[13] = 0
        print("Ex1 Step14 DONE")
    if 15 in steps:
        # Average length of sentences, in tokens
        count = len(comment_split)
        size = len(comment.split("\n"))
        if size == 0:
            features[14] = 0
        else:
            features[14] = count/size
        print("Ex1 Step15 DONE")
    if 16 in steps:
        # Average length of tokens, excluding punctuation-only tokens, in characters
        punc_reg = r"^\W*$"
        total_num = 0
        total_length = 0
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                n = re.match(punc_reg, m.group(1))
                if not n:
                    total_length += len(m.group(1))
                    total_num += 1
        if total_num == 0:
            features[15] = 0
        else:
            features[15] = total_length/total_num
        print("Ex1 Step16 DONE")
    if 17 in steps:
        # Number of sentences.
        size = len(comment.split("\n"))
        features[16] = size
        print("Ex1 Step17 DONE")
    if 18 in steps:
        # Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
        # must convert str to int
        total_num = 0
        total_aoa = 0
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in bristol:
                    total_aoa += int(bristol[m.group(1)][0])
                    total_num += 1
        if total_num == 0:
            features[17] = 0
        else:
            features[17] = total_aoa/total_num
        print("Ex1 Step18 DONE")
    if 19 in steps:
        # Average of IMG from Bristol, Gilhooly, and Logie norms
        total_num = 0
        total_img = 0
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in bristol:
                    total_img += int(bristol[m.group(1)][1])
                    total_num += 1
        if total_num == 0:
            features[18] = 0
        else:
            features[18] = total_img/total_num
        print("Ex1 Step19 DONE")
    if 20 in steps:
        # Average of FAM from Bristol, Gilhooly, and Logie norm
        total_num = 0
        total_fam = 0
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in bristol:
                    total_fam += int(bristol[m.group(1)][2])
                    total_num += 1
        if total_num == 0:
            features[19] = 0
        else:
            features[19] = total_fam/total_num
        print("Ex1 Step20 DONE")
    if 21 in steps:
        # Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norm
        lst_aoa = []
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in bristol:
                    lst_aoa.append(int(bristol[m.group(1)][0]))
        if len(lst_aoa) == 0:
            features[20] = 0
        else:
            features[20] = np.std(lst_aoa)
        print("Ex1 Step21 DONE")
    if 22 in steps:
        # Standard deviation of IMG from Bristol, Gilhooly, and Logie norm
        lst_img = []
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in bristol:
                    lst_img.append(int(bristol[m.group(1)][1]))
        if len(lst_img) == 0:
            features[21] = 0
        else:
            features[21] = np.std(lst_img)
        print("Ex1 Step22 DONE")
    if 23 in steps:
        # Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
        lst_fam = []
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in bristol:
                    lst_fam.append(int(bristol[m.group(1)][2]))
        if len(lst_fam) == 0:
            features[22] = 0
        else:
            features[22] = np.std(lst_fam)
        print("Ex1 Step23 DONE")
    if 24 in steps:
        # Average of V.Mean.Sum from Warriner norm
        # must convert str to float
        total_num = 0
        total_vms = 0.0
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in warriner:
                    total_vms += float(warriner[m.group(1)][0])
                    total_num += 1
        if total_num == 0:
            features[23] = 0
        else:
            features[23] = total_vms/total_num
        print("Ex1 Step24 DONE")
    if 25 in steps:
        # Average of A.Mean.Sum from Warriner norms
        total_num = 0
        total_ams = 0.0
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in warriner:
                    total_ams += float(warriner[m.group(1)][3])
                    total_num += 1
        if total_num == 0:
            features[24] = 0
        else:
            features[24] = total_ams/total_num
        print("Ex1 Step25 DONE")
    if 26 in steps:
        # Average of D.Mean.Sum from Warriner norm
        total_num = 0
        total_dms = 0.0
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in warriner:
                    total_dms += float(warriner[m.group(1)][6])
                    total_num += 1
        if total_num == 0:
            features[25] = 0
        else:
            features[25] = total_dms/total_num
        print("Ex1 Step26 DONE")
    if 27 in steps:
        # Standard deviation of V.Mean.Sum from Warriner norms
        lst_vms = []
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in warriner:
                    lst_vms.append(float(warriner[m.group(1)][0]))
        if len(lst_vms) == 0:
            features[26] = 0
        else:
            features[26] = np.std(lst_vms)
        print("Ex1 Step27 DONE")
    if 28 in steps:
        # Standard deviation of A.Mean.Sum from Warriner norms
        lst_ams = []
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in warriner:
                    lst_ams.append(float(warriner[m.group(1)][3]))
        if len(lst_ams) == 0:
            features[27] = 0
        else:
            features[27] = np.std(lst_ams)
        print("Ex1 Step28 DONE")
    if 29 in steps:
        # Standard deviation of D.Mean.Sum from Warriner norms
        lst_dms = []
        for word in comment_split:
            m = re.match(reg_uni, word)
            if m:
                if m.group(1) in warriner:
                    lst_dms.append(float(warriner[m.group(1)][6]))
        if len(lst_dms) == 0:
            features[28] = 0
        else:
            features[28] = np.std(lst_dms)
        print("Ex1 Step29 DONE")

    return features


def extract2(ID, cat):
    """Auto generate the 144 feature that using LIWC

    Parameter ID:
        ID name
    Parameter cat:
        Category name
    Return:
        The 144 features
    """
    # Step 29 - 173
    arr = np.zeros(174)
    if cat == "Alt":
        id_index = altid[ID]
        arr[29:173] = alt_npy[id_index]
        print("Ex2 Alt")
    elif cat == "Center":
        id_index = centerid[ID]
        arr[29:173] = center_npy[id_index]
        print("Ex2 Center")
    elif cat == "Left":
        id_index = leftid[ID]
        arr[29:173] = left_npy[id_index]
        print("Ex2 Left")
    elif cat == "Right":
        id_index = rightid[ID]
        arr[29:173] = right_npy[id_index]
        print("Ex2 Right")
    return arr


def slang_dict():
    """Generate a dictionary with all slang words in file slang
    Returns:
        The dictionary with the content of all slang words
    """
    slang_dict = []
    if os.path.exists(dict_slang):
        with open(dict_slang, "r") as f:
            for line in f:
                key = line.split()
                if key:
                    slang_dict.append([key[0]])
        return slang_dict
    else:
        print("File in this path " + dict_slang + " does not exist.")


def bristol_dict():
    """Generate a dictionary with all bristol words in file BristolNorms
    Returns:
        The dictionary with the content of all bristol words
    """
    if os.path.exists(dict_bristol):
        reader = csv.reader(open(dict_bristol))
        next(reader)
        bristol_dict = {rows[1]: rows[3:6] for rows in reader}
        return bristol_dict
    else:
        print("File in this path " + dict_bristol + " does not exist.")


def warriner_dict():
    """Generate a dictionary with all warriner words in file warriner
    Returns:
        The dictionary with the content of all warriner words
    """
    if os.path.exists(dict_warriner):
        reader = csv.reader(open(dict_warriner))
        next(reader)
        warriner_dict = {rows[1]: rows[2:9] for rows in reader}
        return warriner_dict
    else:
        print("File in this path " + dict_warriner + " does not exist.")


def dict_altID():
    """Generate a dictionary with all altID in file Alt_IDs.txt
    Returns:
        The dictionary with the content of all key altID and value be index of them
    """
    alt_dict = {}
    if os.path.exists(dict_alt):
        with open(dict_alt, "r") as f:
            i = 0
            for line in f:
                key = line.split()
                alt_dict[key[0]] = i
                i += 1
        return alt_dict
    else:
        print("File in this path " + dict_alt + " does not exist.")


def dict_leftID():
    """Generate a dictionary with all leftID in file Left_IDs.txt
    Returns:
        The dictionary with the content of all key leftID and value be index of them
    """
    left_dict = {}
    if os.path.exists(dict_left):
        with open(dict_left, "r") as f:
            i = 0
            for line in f:
                key = line.split()
                left_dict[key[0]] = i
                i += 1
        return left_dict
    else:
        print("File in this path " + dict_left + " does not exist.")


def dict_rightID():
    """Generate a dictionary with all rightID in file Right_IDs.txt
    Returns:
        The dictionary with the content of all key rightID and value be index of them
    """
    right_dict = {}
    if os.path.exists(dict_right):
        with open(dict_right, "r") as f:
            i = 0
            for line in f:
                key = line.split()
                right_dict[key[0]] = i
                i += 1
        return right_dict
    else:
        print("File in this path " + dict_right + " does not exist.")


def dict_centerID():
    """Generate a dictionary with all centerID in file Center_IDs.txt
    Returns:
        The dictionary with the content of all key centerID and value be index of them
    """
    center_dict = {}
    if os.path.exists(dict_center):
        with open(dict_center, "r") as f:
            i = 0
            for line in f:
                key = line.split()
                center_dict[key[0]] = i
                i += 1
        return center_dict
    else:
        print("File in this path " + dict_center + " does not exist.")


def main(args):

    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # TODO: your code here
    for i in range(len(data)):
        j = data[i]
        ID = j["id"]
        print(ID)
        body = j["body"]
        cat = j["cat"]
        print(cat)
        feats[i] = extract1(body) + extract2(ID, cat)
        feats[i][173] = class_integer[cat]

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__":

    slang = slang_dict()
    print("slang")
    bristol = bristol_dict()
    print("bristol")
    warriner = warriner_dict()
    print("warriner")
    altid = dict_altID()
    print("altid")
    centerid = dict_centerID()
    print("centerid")
    leftid = dict_leftID()
    print("leftid")
    rightid = dict_rightID()
    print("rightid")

    parser = argparse.ArgumentParser(description='Process each.')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    start_time = time.time()

    main(args)

    print("--- %s seconds to finish extracting features ---" % (time.time() - start_time))
