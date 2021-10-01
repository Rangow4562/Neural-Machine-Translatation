from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,TransfoXLTokenizer
PATH1 = 'D:/translation-service/Project-Language-Translator-using-Flask-master/model/Seq2Seq/'
tokenizer1 = AutoTokenizer.from_pretrained(PATH1, local_files_only=True)
model1 = AutoModelForSeq2SeqLM.from_pretrained(PATH1, local_files_only=True)
PATH2 = 'D:/translation-service/Project-Language-Translator-using-Flask-master/model/Seq2Seq_en_es/'
tokenizer2 = AutoTokenizer.from_pretrained(PATH2, local_files_only=True)
model2 = AutoModelForSeq2SeqLM.from_pretrained(PATH2, local_files_only=True)
PATH3 = 'D:/translation-service/Project-Language-Translator-using-Flask-master/model/Seq2Seq_en_hi/'
tokenizer3 = AutoTokenizer.from_pretrained(PATH3, local_files_only=True)
model3 = AutoModelForSeq2SeqLM.from_pretrained(PATH3, local_files_only=True)
PATH = 'D:/translation-service/Project-Language-Translator-using-Flask-master/model/Seq2Seq_en_zh/'
tokenizer4 = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
model4 = AutoModelForSeq2SeqLM.from_pretrained(PATH, local_files_only=True)

input_ids1 = tokenizer1('The <extra_id_0> de_DE in <extra_id_1> park', return_tensors='pt').input_ids
input_ids2 = tokenizer2('The <extra_id_0> es_XX in <extra_id_1> park', return_tensors='pt').input_ids
input_ids3 = tokenizer3('The <extra_id_0> hi_IN in <extra_id_1> park', return_tensors='pt').input_ids
input_ids4 = tokenizer4('The <extra_id_0> zh_CN in <extra_id_1> park', return_tensors='pt').input_ids
input_ids = {input_ids1,input_ids2,input_ids3,input_ids4}
model={[model1],[model2],[model3],[model4]}
attention_mask = input_ids.ne(model.config.pad_token_id).long()
decoder_input_ids = tokenizer('<pad> <extra_id_0> en_XX <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids

traced_model = torch.jit.trace(model, (input_ids, attention_mask, decoder_input_ids))
torch.jit.save(traced_model, "D:/translation-service/Project-Language-Translator-using-Flask-master/model/Seq2SeqBart")

name = 'traced_model'
tokenizer = TransfoXLTokenizer(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name)
tokenizer.save_pretrained(" D:/translation-service/Project-Language-Translator-using-Flask-master/model/Seq2SeqBart")
model.save_pretrained("D:/translation-service/Project-Language-Translator-using-Flask-master/model/Seq2SeqBart")