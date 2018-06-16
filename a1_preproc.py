#!/usr/bin/env python3
import sys
import argparse
import os
import json
import html
import re
import spacy
import time

indir = "/u/cs401/A1/data/"
dicta = "/u/cs401/Wordlists/abbrev.english"
dicts = "/u/cs401/Wordlists/StopWords"
# Du you know da we?

reg0 = r"(.*)(/)(.*)"

nlp = spacy.load('en', disable=['parser', 'ner'])


def preproc1(comment, steps=range(1, 11)):
    """ This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment 
    """

    modComm = ""
    if 1 in steps:
        # Remove all new line characters
        modComm = re.sub(r"\r?\n", " ", comment)
        print("Step 1: " + modComm)
    if 2 in steps:
        # Remove all HTML character codes replaced to equivalent ASCII
        modComm = html.unescape(modComm)
        print("Step 2: " + modComm)
    if 3 in steps:
        # Remove all URLs
        modComm = re.sub(r"http\S+", " ", modComm)
        modComm = re.sub(r"www\S+", " ", modComm)
        modComm = re.sub(r"ftp\S+", " ", modComm)
        modComm = re.sub(r"localhost\S+", " ", modComm)
        print("Step 3: " + modComm)
    if 4 in steps:
        modCom = []
        reg_pun = r"([\W]*)([\w^\']+)([\W]*)"  # try to match words with punctuations
        comment_split = modComm.split()
        for i in comment_split:
            if i in abbrev:  # Periods in abbreviations
                modCom.append(i)
            else:  # need to find Apostrophes and multiple punctuations
                # match multiple punctuation
                # eg: 1. bla -> bla; 2. bla??! -> bla ??!; 3.???bla -> ??? bla; 4. !!bla?? -> !!! bla ??
                # about apostrophy, do not match group(2) with '
                if re.match(reg_pun, i):
                    if re.match(reg_pun, i).group(1) is "" and re.match(reg_pun, i).group(3) is "":
                        modCom.append(re.match(reg_pun, i).group(2))
                    elif re.match(reg_pun, i).group(1) is "" and re.match(reg_pun, i).group(3) is not "":
                        modCom.append(re.match(reg_pun, i).group(2))
                        modCom.append(re.match(reg_pun, i).group(3))
                    elif re.match(reg_pun, i).group(1) is not "" and re.match(reg_pun, i).group(3) is "":
                        modCom.append(re.match(reg_pun, i).group(1))
                        modCom.append(re.match(reg_pun, i).group(2))
                    else:
                        modCom.append(re.match(reg_pun, i).group(1))
                        modCom.append(re.match(reg_pun, i).group(2))
                        modCom.append(re.match(reg_pun, i).group(3))
                else:
                    modCom.append(i)
        modComm = " ".join(modCom)
        print("Step 4: " + modComm)
    if 5 in steps:
        # There exist 7 clitics:
        # 'm(I'm); 're(you're); 's(she's, the girl next door's cat);
        # s' => s ' (dogs' => dogs ');
        # 'll(they'll); 've(they've); n't(couldn't)

        # 1 n't special case: r"(\w+)(n't)" => can't couldn't shouldn't won't
        # 2 'm 's 're 'll 've special caseL r"(\w+)('\w?\w)"
        #   => they're they'll they've I'm door's
        modCom = []
        comment_split = modComm.split()
        # reg_all = r"(\w+)('\w\w)|(\w+)(n't)|(\w+)('\w)|(\w+)(')"
        reg1 = r"(\w+)(n't)"
        reg2 = r"(\w+)('\w*)"
        for i in comment_split:
            if re.match(reg1, i):
                modCom.append(re.match(reg1, i).group(1))
                modCom.append(re.match(reg1, i).group(2))
            elif re.match(reg2, i):
                modCom.append(re.match(reg2, i).group(1))
                modCom.append(re.match(reg2, i).group(2))
            else:
                modCom.append(i)
        modComm = " ".join(modCom)
        print("Step 5: " + modComm)
    if 6 in steps:
        new_doc = []
        doc = spacy.tokens.Doc(nlp.vocab, words=modComm.split())
        doc = nlp.tagger(doc)
        for token in doc:
            tok = token.text + "/" + token.tag_
            new_doc.append(tok)
        modComm = " ".join(new_doc)
        print("Step 6: " + modComm)
    if 7 in steps:
        comment_split = modComm.split()
        nlst = []
        for i in comment_split:
            if re.match(reg0, i):
                # since all the words in the StopWords file is lower case
                # we need to match ONLY the lower case
                if re.match(reg0, i).group(1).lower() not in stop:
                    nlst.append(i)
        modComm = " ".join(nlst)
        print("Step 7: " + modComm)
    if 8 in steps:
        nls = []
        comment_split = modComm.split()
        for i in comment_split:
            doc = spacy.tokens.Doc(nlp.vocab, words=[re.match(reg0, i).group(1)])
            doc = nlp.tagger(doc)
            for token in doc:
                if token.lemma_.startswith("-") and token.text[0] is not "-":
                    nls.append(i)
                else:
                    tok = token.lemma_ + "/" + token.tag_
                    nls.append(tok)
        modComm = " ".join(nls)
        print("Step 8: " + modComm)
    if 9 in steps:
        reg3 = r"etc."
        reg4 = r"vs."
        modSen = ""
        comment_split = modComm.split()
        for i in comment_split:
            if i[-1] is ".":  # match tag .
                modSen += i + "\n"
                continue
            if re.match(reg3, re.match(reg0, i).group(1)):  # match etc. at the end of the sentences
                modSen += i + "\n"
                continue
            if re.match(reg4, re.match(reg0, i).group(1)):  # match vs. at the end of the sentences
                modSen += i + "\n"
                continue
            else:
                modSen += i + " "
        modComm = modSen.rstrip("\n")
        print("Step 9: " + modComm)
    if 10 in steps:
        nls1 = []
        nls2 = []
        comment_split = modComm.split("\n")
        for i in comment_split:
            word_split = i.split()
            for j in word_split:
                lower = re.match(reg0, j).group(1).lower() + "/" + re.match(reg0, j).group(3)
                nls2.append(lower)
            nls1.append(" ".join(nls2))
            nls2 = []
        modComm = "\n".join(nls1)
        print("Step 10: " + modComm)
        
    return modComm


def dict_abbrev():
    """Generate a dictionary with all abbreviations in file abbrev.english
    Returns:
        The dictionary with the content of all abbreviation words
    """
    abb_dict = {}
    if os.path.exists(dicta):
        with open(dicta, "r") as f:
            abb_dict["e.g."] = True
            for line in f:
                key = line.split()
                abb_dict[str(key[0])] = True
        return abb_dict
    else:
        print("File in this path " + dicta + " does not exist.")


def dict_stop():
    """Generate a dictionary with all stop words in file StopWords
    Returns:
        The dictionary with the content of all stop words
    """
    stop_dict = {}
    if os.path.exists(dicts):
        with open(dicts, "r") as f:
            for line in f:
                key = line.split()
                stop_dict[str(key[0])] = True
        return stop_dict
    else:
        print("File in this path " + dicts + " does not exist.")


def main(args):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            file_name = os.path.basename(fullFile)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            indexStart = args.ID[0] % len(data)
            indexEnd = args.max + indexStart
            # TODO: read those lines with something like `j = json.loads(line)`
            for i in range(indexStart, indexEnd):
                j = json.loads(data[i])
                # TODO: choose to retain fields from those lines that are relevant to you
                newInfo = {}
                newInfo["id"] = j["id"]
                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                # TODO: replace the 'body' field with the processed text
                newInfo["body"] = preproc1(j["body"])
                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                newInfo["cat"] = file_name
                # TODO: append the result to 'allOutput'
                allOutput.append(newInfo)

            # need to type "python a1_preproc.py 1001705933 -o preproc.json" in bash
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":

    abbrev = dict_abbrev()  # Dictionary with all abbrev words in file abbrev.english
    stop = dict_stop()  # Dictionary with all stop words in file StopWords

    parser = argparse.ArgumentParser(description='Process each.')
    parser.add_argument('ID', metavar='N', type=int, nargs=1, help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    # change 10000 to 10 to minimize the time, need to change back!
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if args.max > 200272:
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    start_time = time.time()

    main(args)

    print("--- %s seconds to finish pre-processing ---" % (time.time() - start_time))
