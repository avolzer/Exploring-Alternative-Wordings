import torch
from transformers import MarianMTModel, MarianTokenizer
import string

#english to spanish
en_ROMANCE_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
en_ROMANCE_tokenizer = MarianTokenizer.from_pretrained(en_ROMANCE_model_name)
en_ROMANCE = MarianMTModel.from_pretrained(en_ROMANCE_model_name)

#spanish to english
ROMANCE_en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
ROMANCE_en_tokenizer = MarianTokenizer.from_pretrained(ROMANCE_en_model_name)
ROMANCE_en = MarianMTModel.from_pretrained(ROMANCE_en_model_name)

def incremental_generation(english_only, english, start, prefix_only):
    if english_only:
        #translate english to spanish
        engbatch = en_ROMANCE_tokenizer.prepare_translation_batch(["<pad>" + english])
        eng_to_spanish = en_ROMANCE.generate(**engbatch)
        machine_translation = en_ROMANCE_tokenizer.decode(eng_to_spanish[0])

        #prepare spanish to be translated back to english
        tokenizer = ROMANCE_en_tokenizer
        model = ROMANCE_en
        batchstr = ">>en<<" + machine_translation.replace("<pad> ", '')
        tokenized_prefix = tokenizer.convert_tokens_to_ids(en_ROMANCE_tokenizer.tokenize(start))

    #prepare english to be translated to spanish
    else:
        tokenizer = en_ROMANCE_tokenizer
        model = en_ROMANCE
        batchstr = ">>es<<" + english
        tokenized_prefix = tokenizer.convert_tokens_to_ids(ROMANCE_en_tokenizer.tokenize(start))

    prefix = torch.LongTensor(tokenized_prefix)

    batch = tokenizer.prepare_translation_batch([batchstr])
    original_encoded = model.get_encoder()(**batch)
    decoder_start_token = model.config.decoder_start_token_id
    partial_decode = torch.LongTensor([decoder_start_token]).unsqueeze(0)
    past = (original_encoded, None)

    #machine translation for comparative purposes
    translation_tokens = model.generate(**batch)
    machine_translation = tokenizer.decode(translation_tokens[0]).split("<pad>")[1]

    num_tokens_generated = 0
    prediction_list = []
    MAX_LENGTH = 100
    total = 0

    #generate tokens incrementally 
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            partial_decode, past=past, attention_mask=batch['attention_mask'], use_cache=model.config.use_cache
        )
        with torch.no_grad():
            model_outputs = model(**model_inputs)

        next_token_logits = model_outputs[0][:, -1, :]
        past = model_outputs[1]
        
        #start with designated beginning
        if num_tokens_generated < len(prefix):
            next_token_to_add = prefix[num_tokens_generated]
        elif prefix_only == True:
            break
        else:
            next_token_to_add = next_token_logits[0].argmax()

        #calculate score
        next_token_logprobs = next_token_logits - next_token_logits.logsumexp(1, True)
        token_score = next_token_logprobs[0][next_token_to_add].item()
        total += token_score

        #append top 10 predictions for each token to list
        decoded_predictions = []
        for tok in next_token_logits[0].topk(10).indices:
            decoded_predictions.append(tokenizer.convert_ids_to_tokens(tok.item()).replace('\u2581', '\u00a0'))
        
        #list of lists of predictions
        prediction_list.append(decoded_predictions)

        #add new token to tokens so far
        partial_decode = torch.cat((partial_decode, next_token_to_add.unsqueeze(0).unsqueeze(0)), -1)
        num_tokens_generated += 1

        #stop generating at </s>, or when max num tokens exceded
        if next_token_to_add.item() == 0 or not (num_tokens_generated < MAX_LENGTH):
            break

    #list of tokens used to display sentence
    decoded_tokens = [sub.replace('\u2581', '\u00a0') for sub in tokenizer.convert_ids_to_tokens(partial_decode[0])]
    decoded_tokens.remove("<pad>")

    final = tokenizer.decode(partial_decode[0]).replace("<pad>", '').replace('\u2581', '\u00a0')
    score = round(total/(len(decoded_tokens)), 3)

    if english_only:
        new_english = final
    #back translate spanish into english
    else:
        batch2 = ROMANCE_en_tokenizer.prepare_translation_batch([">>en<< " + final])
        spanish_to_english = ROMANCE_en.generate(**batch2)
        new_english = ROMANCE_en_tokenizer.decode(spanish_to_english[0]).replace("<pad>", '')

    return {"translation": final,
                    "expected" : machine_translation,
                    "newEnglish" : new_english,
                    "tokens" : decoded_tokens,
                    "predictions" : prediction_list,
                    "score" : score
                }

def rearrange(english, start, auto):
    wordlist = [''.join(x for x in par if x not in string.punctuation) for par in english.split(' ')]
    # first_phrases = set([word.capitalize() for word in wordlist])

    #get most likely sentence or prefix and its score, given the word to move towards front
    def get_alt(start, prefix_only):
        if start in wordlist:
            pos = wordlist.index(start.lstrip())
        #if subword token is selected
        else:
            res = [i for i in wordlist if start.lstrip() in i]
            pos = wordlist.index(res[0])
        #word before selected word
        first_phrases = [wordlist[pos - 1].capitalize(), ' '.join(wordlist[pos-2: pos]).capitalize(), 'The']

        # first_phrases.add(wordlist[pos - 1].capitalize())
        # #2 words before selected word
        # first_phrases.add(' '.join(wordlist[pos-2: pos]).capitalize())
        # first_phrases.add('The')
        print(first_phrases)
        prefixes = [word + ' ' + start.lstrip() for word in first_phrases]
        prefixes.append(start.lstrip().capitalize())

        results = []
        scores = []

        #score each possible prefix/sentence
        for prefix in prefixes:
            data = incremental_generation(english_only=True, english=english, start=prefix, prefix_only=prefix_only)
            results.append(data["translation"])
            scores.append(data["score"])
            
        #select most likely sentence or prefix
        ind = scores.index(max(scores))
        winner = results[ind]
        winnerscore = scores[ind]
        return (winnerscore, winner)

    alternatives = []
    winner = ''

    #generate a list of alternatives
    if auto:
        print(wordlist)
        #skip first 3 words bc they all return the default sentence
        for word in wordlist[3:]:
            alt = get_alt(word, prefix_only=False)
            #avoid duplicate sentences
            if alt not in alternatives:
                alternatives.append(alt)
            sorted_scores = sorted(((score, result) for score, result in alternatives), reverse=True)
        alternatives = [pair[1] for pair in sorted_scores]
        return {"alternatives" : alternatives}

    else:
        #get most likely sentence
        winner = get_alt(start, prefix_only=False)[1]
        #get full data with predictions
        return incremental_generation(english_only=True, english=english, start=winner, prefix_only=False)

    